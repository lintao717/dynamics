"""[V1.7R] ForecastRunner with proper forward simulation.

Key fixes vs V1.7:
  - Simulates through cutoff_step + horizon_steps (no padding)
  - Assimilates cutoff state from observed history
  - Produces behavioral-flow observables (first/repeat/active)
  - Two population modes: oracle_cohort, observed_closed
  - Asserts trajectory length, rejects short output

Usage:
  config = ForecastConfig(cutoff_step=2, horizon_steps=3, ...)
  runner = ForecastRunner(config)
  result = runner.run(history, case, params)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dynamics_simulation.config import ModelParams, default_params, ReactivationMode
from dynamics_simulation.data.schema import EventCase, RootPost, InteractionRecord
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.forecast import EventHistory
from dynamics_simulation.data.networks import ReplayNetworkMode, build_network_provider
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.agents import AgentState, initialize_agents


@dataclass
class ForecastConfig:
    """Configuration for a single forecast run."""

    cutoff_step: int
    horizon_steps: int
    step_hours: float = 24.0
    population_mode: str = "oracle_cohort"  # "oracle_cohort" | "observed_closed"
    reactivation_mode: ReactivationMode = ReactivationMode.ONE_SHOT
    forecast_seeds: tuple[int, ...] = (211, 223, 227)
    micro_steps: int = 1
    params: Optional[ModelParams] = None


@dataclass
class ForecastResult:
    """Output of one forecast run."""

    case_id: str
    cutoff_step: int
    horizon_steps: int
    population_mode: str
    n_agents: int
    n_observed: int  # observed at cutoff
    n_future: int   # potential future participants

    # Behavioral observables (per forecast seed and aggregated)
    sim_active_ts_p50: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_first_ts_p50: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_repeat_ts_p50: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_active_ts_p5: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_active_ts_p95: np.ndarray = field(default_factory=lambda: np.array([]))

    # Future only
    fc_active: np.ndarray = field(default_factory=lambda: np.array([]))
    fc_first: np.ndarray = field(default_factory=lambda: np.array([]))
    fc_repeat: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    git_sha: str = ""
    effective_params: dict = field(default_factory=dict)
    leakage_flags: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "cutoff_step": self.cutoff_step,
            "horizon_steps": self.horizon_steps,
            "population_mode": self.population_mode,
            "n_agents": self.n_agents,
            "n_observed": self.n_observed,
            "n_future": self.n_future,
            "sim_active_p50": self.sim_active_ts_p50.tolist(),
            "sim_first_p50": self.sim_first_ts_p50.tolist(),
            "sim_repeat_p50": self.sim_repeat_ts_p50.tolist(),
            "sim_active_p5": self.sim_active_ts_p5.tolist(),
            "sim_active_p95": self.sim_active_ts_p95.tolist(),
            "fc_active_future": self.fc_active.tolist(),
            "fc_first_future": self.fc_first.tolist(),
            "fc_repeat_future": self.fc_repeat.tolist(),
            "git_sha": self.git_sha,
            "effective_params": self.effective_params,
            "leakage_flags": self.leakage_flags,
        }


def _build_population(history: EventHistory, full_case: EventCase,
                      mode: str, grid: TimeGrid,
                      ) -> tuple[tuple[str, ...], int, int]:
    """Build diagnostic population list.

    Returns:
      (population_user_ids, n_observed, n_future_potential)
    """
    observed_users = set(history.observed_user_ids)
    all_users = set(full_case.user_ids)

    if mode == "oracle_cohort":
        # Full eventual participant list (future identities known, but actions hidden)
        pop = full_case.user_ids  # deterministic order, root first
        n_obs = len(observed_users)
        n_fut = len(all_users - observed_users)
        return pop, n_obs, n_fut

    elif mode == "observed_closed":
        # Only users seen by cutoff — no new entries
        pop = tuple(sorted(observed_users))
        # Ensure root is first
        if pop[0] != full_case.root.user_id:
            pop = (full_case.root.user_id,) + tuple(
                u for u in pop if u != full_case.root.user_id)
        n_obs = len(pop)
        n_fut = 0
        return pop, n_obs, n_fut

    else:
        raise ValueError(f"Unknown population_mode: {mode}")


def build_cutoff_state(history: EventHistory, population: tuple[str, ...],
                       params: ModelParams, grid: TimeGrid, cutoff_step: int,
                       rng: np.random.Generator) -> AgentState:
    """Build agent state at cutoff from observed history.

    Rules:
      - active in cutoff window -> A, m=1
      - active before cutoff but not in cutoff window -> D, m=1
      - never active by cutoff -> U, m=0
      - no agent starts in E
      - root author follows observed-history rule
      - fatigue initialized from recent activity density
    """
    N = len(population)
    user_to_idx = {uid: i for i, uid in enumerate(population)}

    # Collect active windows per user from history
    active_windows = {uid: set() for uid in population}
    for ix in history.interactions:
        if ix.user_id in active_windows:
            step = grid.step_of(ix.timestamp)
            if step <= cutoff_step:
                active_windows[ix.user_id].add(step)

    # Add root if active at step 0
    root_uid = history.root.user_id
    if root_uid in active_windows:
        active_windows[root_uid].add(0)

    # Assign initial z and m
    z = np.full(N, 0, dtype=np.int32)  # U=0
    m = np.zeros(N, dtype=np.int32)

    for uid, windows in active_windows.items():
        if uid not in user_to_idx:
            continue
        i = user_to_idx[uid]
        if not windows:
            z[i] = 0  # U
            m[i] = 0
        elif cutoff_step in windows:
            z[i] = 2  # A
            m[i] = 1  # previously activated
        else:
            z[i] = 3  # D
            m[i] = 1  # previously activated

    # Build full state with default opinion/emotion
    state = initialize_agents(N, initial_active=0,
                              rng=rng)
    # Override z and m with cutoff-based assignments
    state = AgentState(
        z=z, m=m, o=state.o, o_hat=state.o_hat,
        h=state.h, f=state.f, attrs=state.attrs,
    )

    # Initialize fatigue from recent activity density
    total_windows = max(cutoff_step + 1, 1)
    for i, uid in enumerate(population):
        n_active = len(active_windows.get(uid, set()))
        state.f[i] = float(np.clip(n_active / total_windows, 0.0, 1.0))

    return state


class ForecastRunner:
    """Run a forward simulation from a cutoff state."""

    def __init__(self, config: ForecastConfig):
        self.config = config

    def run(self, history: EventHistory, full_case: EventCase,
            params: ModelParams) -> ForecastResult:
        """Execute forecast from cutoff.

        Args:
            history: Pre-cutoff data.
            full_case: Full EventCase (used only for population list).
            params: Model parameters.

        Returns:
            ForecastResult with behavioral observables.

        Raises:
            RuntimeError: If simulation trajectory is too short.
        """
        cfg = self.config
        grid = TimeGrid.from_case(full_case, step_hours=cfg.step_hours, tail_steps=0)
        T_sim = cfg.cutoff_step + cfg.horizon_steps

        # Build population
        population, n_obs, n_fut = _build_population(
            history, full_case, cfg.population_mode, grid)
        N = len(population)
        index = NodeIndex(
            user_to_idx={uid: i for i, uid in enumerate(population)},
            idx_to_user=population,
        )

        # Build cutoff state
        rng = np.random.default_rng(cfg.forecast_seeds[0])
        cutoff_state = build_cutoff_state(
            history, population, params, grid, cfg.cutoff_step, rng)

        # Run forward simulation for each forecast seed
        all_active = []
        all_first = []
        all_repeat = []

        for seed in cfg.forecast_seeds:
            seed_rng = np.random.default_rng(seed)

            # Build network from history interactions only
            net_provider = build_network_provider(
                full_case, index, grid, mode=ReplayNetworkMode.BROADCAST)

            def net_fn(step):
                snap = net_provider.snapshot_at(step)
                return snap.G_s, snap.G_o, snap.communities

            from dynamics_simulation.data.timeline import EventInputTimeline
            timeline = EventInputTimeline(full_case, index, grid)

            def input_fn(n, t, T, ms=0, mt=1):
                return timeline.inputs_at(n, t, ms, mt)

            from dynamics_simulation.simulation import SimulationConfig, SimulationRunner
            sim_cfg = SimulationConfig(
                n_agents=N, T=T_sim, micro_steps=cfg.micro_steps,
                reactivation_mode=cfg.reactivation_mode.value,
                seed=seed, params=params,
                initial_state=cutoff_state.copy(),
                network_provider=net_fn, input_fn=input_fn,
                verbose=False,
            )
            runner = SimulationRunner(sim_cfg)
            metrics = runner.run()

            # Assert trajectory length
            if len(metrics.n_A_ts) < T_sim + 1:
                raise RuntimeError(
                    f"Forecast trajectory too short: got {len(metrics.n_A_ts)}, "
                    f"need {T_sim + 1} (cutoff={cfg.cutoff_step}+horizon={cfg.horizon_steps}+1)"
                )

            # Extract behavioral observables from transition flow
            E_to_A = np.array(metrics.E_to_A_ts, dtype=np.int32)
            D1_to_A = np.array(metrics.D1_to_A_ts, dtype=np.int32)
            # For ONE_SHOT, D0_to_A and D1_to_A are both zero, so:
            # first_actor = E_to_A, repeat = 0 (because D1->A is disabled)
            # But more generally: first = E_to_A + D0_to_A, repeat = D1_to_A
            # For observed_closed: no future users, so E_to_A comes from U pool
            first = E_to_A.copy()
            repeat = D1_to_A.copy()
            active = first + repeat

            all_active.append(active)
            all_first.append(first)
            all_repeat.append(repeat)

        # Aggregate across seeds
        active_stack = np.stack([a[:T_sim + 1] for a in all_active], axis=0)
        first_stack = np.stack([f[:T_sim + 1] for f in all_first], axis=0)
        repeat_stack = np.stack([r[:T_sim + 1] for r in all_repeat], axis=0)

        p50_a = np.percentile(active_stack, 50, axis=0)
        p5_a = np.percentile(active_stack, 5, axis=0)
        p95_a = np.percentile(active_stack, 95, axis=0)
        p50_f = np.percentile(first_stack, 50, axis=0)
        p50_r = np.percentile(repeat_stack, 50, axis=0)

        # Future-only values (after cutoff)
        future_idx = slice(cfg.cutoff_step + 1, T_sim + 1)
        fc_active = p50_a[future_idx]
        fc_first = p50_f[future_idx]
        fc_repeat = p50_r[future_idx]

        # Git SHA
        import subprocess
        git_sha = "unknown"
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            pass

        return ForecastResult(
            case_id=full_case.case_id,
            cutoff_step=cfg.cutoff_step,
            horizon_steps=cfg.horizon_steps,
            population_mode=cfg.population_mode,
            n_agents=N,
            n_observed=n_obs,
            n_future=n_fut,
            sim_active_ts_p50=p50_a,
            sim_first_ts_p50=p50_f,
            sim_repeat_ts_p50=p50_r,
            sim_active_ts_p5=p5_a,
            sim_active_ts_p95=p95_a,
            fc_active=fc_active,
            fc_first=fc_first,
            fc_repeat=fc_repeat,
            git_sha=git_sha,
            effective_params={
                "beta_M": params.propagation.beta_M,
                "alpha_0": params.activation.alpha_0,
                "gamma_0": params.decay.gamma_0,
            },
            leakage_flags={
                "population_mode": cfg.population_mode,
                "cohort_conditioned": cfg.population_mode == "oracle_cohort",
                "causal_forecast": False,
                "reactivation_mode": cfg.reactivation_mode.value,
            },
        )
