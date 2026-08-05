"""[V1.7R.1] ForecastRunner with corrected time indexing and behavioral emission.

Fixes vs V1.7R:
  - T_sim = horizon_steps (not cutoff_step + horizon_steps)
  - future_idx = slice(1, horizon_steps+1) (first step IS first forecast)
  - Input timeline uses absolute step: cutoff_step + rel_step
  - User-level behavioral emission (not just E->A aggregation)
  - Saves full per-step transition traces
  - Cutoff-state A agents get valid o_hat
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dynamics_simulation.config import ModelParams, default_params, ReactivationMode
from dynamics_simulation.data.schema import EventCase
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.forecast import EventHistory
from dynamics_simulation.data.networks import ReplayNetworkMode, build_network_provider
from dynamics_simulation.agents import AgentState, initialize_agents, A


@dataclass
class ForecastConfig:
    cutoff_step: int
    horizon_steps: int
    step_hours: float = 24.0
    population_mode: str = "oracle_cohort"
    reactivation_mode: ReactivationMode = ReactivationMode.ONE_SHOT
    forecast_seeds: tuple[int, ...] = (211, 223, 227)
    micro_steps: int = 1
    params: Optional[ModelParams] = None


def _build_population(history: EventHistory, full_case: EventCase,
                      mode: str, grid: TimeGrid,
                      ) -> tuple[tuple[str, ...], int, int]:
    observed_users = set(history.observed_user_ids)
    all_users = set(full_case.user_ids)
    if mode == "oracle_cohort":
        pop = full_case.user_ids
        return pop, len(observed_users), len(all_users - observed_users)
    elif mode == "observed_closed":
        pop = tuple(sorted(observed_users))
        if pop[0] != full_case.root.user_id:
            pop = (full_case.root.user_id,) + tuple(
                u for u in pop if u != full_case.root.user_id)
        return pop, len(pop), 0
    else:
        raise ValueError(f"Unknown population_mode: {mode}")


def build_cutoff_state(history: EventHistory, population: tuple[str, ...],
                       params: ModelParams, grid: TimeGrid, cutoff_step: int,
                       rng: np.random.Generator) -> AgentState:
    """Build agent state at cutoff from observed history."""
    N = len(population)
    user_to_idx = {uid: i for i, uid in enumerate(population)}

    active_windows = {uid: set() for uid in population}
    for ix in history.interactions:
        if ix.user_id in active_windows:
            step = grid.step_of(ix.timestamp)
            if step <= cutoff_step:
                active_windows[ix.user_id].add(step)

    root_uid = history.root.user_id
    if root_uid in active_windows:
        active_windows[root_uid].add(0)

    z = np.full(N, 0, dtype=np.int32)  # U
    m = np.zeros(N, dtype=np.int32)
    for uid, windows in active_windows.items():
        if uid not in user_to_idx:
            continue
        i = user_to_idx[uid]
        if not windows:
            z[i] = 0; m[i] = 0
        elif cutoff_step in windows:
            z[i] = 2; m[i] = 1  # A
        else:
            z[i] = 3; m[i] = 1  # D

    state = initialize_agents(N, initial_active=0, rng=rng)
    # Fix o_hat for A-state agents: they must have valid public expressions
    a_mask = z == A
    state.o[a_mask] = np.clip(state.o[a_mask], -1.0, 1.0)
    state.o_hat[a_mask] = state.o[a_mask]

    state = AgentState(z=z, m=m, o=state.o, o_hat=state.o_hat,
                       h=state.h, f=state.f, attrs=state.attrs)
    total_windows = max(cutoff_step + 1, 1)
    for i, uid in enumerate(population):
        n_active = len(active_windows.get(uid, set()))
        state.f[i] = float(np.clip(n_active / total_windows, 0.0, 1.0))
    return state


@dataclass
class ForecastResult:
    case_id: str
    cutoff_step: int
    horizon_steps: int
    population_mode: str
    n_agents: int; n_observed: int; n_future: int
    sim_active_p50: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_first_p50: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_repeat_p50: np.ndarray = field(default_factory=lambda: np.array([]))
    fc_active: np.ndarray = field(default_factory=lambda: np.array([]))
    fc_first: np.ndarray = field(default_factory=lambda: np.array([]))
    fc_repeat: np.ndarray = field(default_factory=lambda: np.array([]))
    # V1.7R.1: full per-step transition traces (per seed, aggregated)
    traces: dict = field(default_factory=dict)
    git_sha: str = ""
    effective_params: dict = field(default_factory=dict)
    leakage_flags: dict = field(default_factory=dict)


class ForecastRunner:
    def __init__(self, config: ForecastConfig):
        self.config = config

    def run(self, history: EventHistory, full_case: EventCase,
            params: ModelParams) -> ForecastResult:
        cfg = self.config
        grid = TimeGrid.from_case(full_case, step_hours=cfg.step_hours, tail_steps=0)
        H = cfg.horizon_steps  # number of future steps to simulate

        population, n_obs, n_fut = _build_population(
            history, full_case, cfg.population_mode, grid)
        N = len(population)
        user_to_idx = {uid: i for i, uid in enumerate(population)}

        # Build cutoff state
        rng = np.random.default_rng(cfg.forecast_seeds[0])
        cutoff_state = build_cutoff_state(
            history, population, params, grid, cfg.cutoff_step, rng)

        all_active_steps = []
        all_first_steps = []
        all_repeat_steps = []
        all_traces = []

        for seed in cfg.forecast_seeds:
            # Build network provider
            net_provider = build_network_provider(
                full_case, NodeIndex(user_to_idx=user_to_idx,
                                     idx_to_user=population),
                grid, mode=ReplayNetworkMode.BROADCAST)

            def net_fn(step):
                snap = net_provider.snapshot_at(step)
                return snap.G_s, snap.G_o, snap.communities

            from dynamics_simulation.data.timeline import EventInputTimeline
            idx_obj = NodeIndex(user_to_idx=user_to_idx,
                                idx_to_user=population)
            timeline = EventInputTimeline(full_case, idx_obj, grid)

            def input_fn(n, t, T, ms=0, mt=1):
                # Use ABSOLUTE time offset for exposure
                abs_step = cfg.cutoff_step + t
                return timeline.inputs_at(n, abs_step, ms, mt)

            from dynamics_simulation.simulation import SimulationConfig, SimulationRunner
            # V1.7R.2: user-level behavioral emission via step_observer
            step_active = []   # per-step: count of agents active during step
            step_first = []    # first-ever activation this step (E->A from m=0)
            step_repeat = []   # repeat activity (already had m=1)
            prior_m = cutoff_state.m.copy()

            def observer(step_idx, state_copy, events):
                # Track who was in A at step start vs became A during step
                start_A = (cutoff_state.z == 2) if step_idx == 1 else None
                # Actually use the state_copy from observer
                a_now = state_copy.z == 2
                m_now = state_copy.m
                # First actor: m changed from 0 to 1 this step
                new_first = (m_now == 1) & (prior_m == 0)
                # Repeat: already had m=1 from before
                was_active_before = prior_m == 1
                # Active during step: in A at step end OR newly activated
                active_during = a_now | new_first
                step_active.append(int(active_during.sum()))
                step_first.append(int(new_first.sum()))
                step_repeat.append(int((active_during & ~new_first).sum()))
                prior_m[:] = m_now  # update for next step

            # Build sim config with observer from the start
            sim_cfg = SimulationConfig(
                n_agents=N, T=H,
                micro_steps=cfg.micro_steps,
                reactivation_mode=cfg.reactivation_mode.value,
                seed=seed, params=params,
                initial_state=cutoff_state.copy(),
                network_provider=net_fn, input_fn=input_fn,
                step_observer=observer, verbose=False,
            )
            runner = SimulationRunner(sim_cfg)
            metrics = runner.run()

            if len(metrics.n_A_ts) < H + 1:
                raise RuntimeError(f"Forecast too short: {len(metrics.n_A_ts)} < {H+1}")

            # Save full traces from this seed
            traces = {
                "n_U": [int(x) for x in metrics.n_U_ts[:H+1]],
                "n_A": [int(x) for x in metrics.n_A_ts[:H+1]],
                "n_D": [int(x) for x in metrics.n_D_ts[:H+1]],
                "U_to_E": [int(x) for x in metrics.U_to_E_ts[:H+1]],
                "E_to_A": [int(x) for x in metrics.E_to_A_ts[:H+1]],
                "E_to_D": [int(x) for x in metrics.E_to_D_ts[:H+1]],
                "A_to_D": [int(x) for x in metrics.A_to_D_ts[:H+1]],
            }
            all_active_steps.append(step_active)
            all_first_steps.append(step_first)
            all_repeat_steps.append(step_repeat)
            all_traces.append(traces)

        # Aggregate across seeds
        def _agg(arr_list):
            stack = np.stack([np.array(a) for a in arr_list], axis=0)
            return (np.percentile(stack, 50, axis=0),
                    np.percentile(stack, 5, axis=0),
                    np.percentile(stack, 95, axis=0))

        p50_a, p5_a, p95_a = _agg(all_active_steps)
        p50_f, _, _ = _agg(all_first_steps)
        p50_r, _, _ = _agg(all_repeat_steps)

        # step_active already has only forecast values (no cutoff snapshot)
        fc_active = p50_a[:H]
        fc_first = p50_f[:H]
        fc_repeat = p50_r[:H]

        # Aggregate traces
        agg_traces = {}
        for key in all_traces[0].keys():
            agg_traces[key] = np.median(
                np.stack([np.array(tr[key]) for tr in all_traces]), axis=0
            ).tolist()

        # Git SHA
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            git_sha = "unknown"

        return ForecastResult(
            case_id=full_case.case_id,
            cutoff_step=cfg.cutoff_step,
            horizon_steps=cfg.horizon_steps,
            population_mode=cfg.population_mode,
            n_agents=N, n_observed=n_obs, n_future=n_fut,
            sim_active_p50=p50_a, sim_first_p50=p50_f,
            sim_repeat_p50=p50_r,
            fc_active=fc_active, fc_first=fc_first,
            fc_repeat=fc_repeat,
            traces=agg_traces,
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
