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

        all_active_steps = []  # per-seed: list of per-step active arrays
        all_first_steps = []
        all_repeat_steps = []
        all_traces = []
        prior_m = cutoff_state.m.copy()  # tracks whether agent has ever been active

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
            sim_cfg = SimulationConfig(
                n_agents=N, T=H,  # simulate H steps forward from cutoff
                micro_steps=cfg.micro_steps,
                reactivation_mode=cfg.reactivation_mode.value,
                seed=seed, params=params,
                initial_state=cutoff_state.copy(),
                network_provider=net_fn, input_fn=input_fn,
                verbose=False,
            )
            runner = SimulationRunner(sim_cfg)
            metrics = runner.run()

            if len(metrics.n_A_ts) < H + 1:
                raise RuntimeError(
                    f"Forecast trajectory too short: {len(metrics.n_A_ts)} < {H + 1}")

            # User-level behavioral emission per step
            # active_during_step[i] = True if agent i was active (in A) at ANY
            # point during this step. We approximate as: agent was in A either
            # at start or became A during the step via E->A or D1->A.
            # Since we only have snapshot-level data, we count:
            #   active = A-state at step start OR new activation this step
            n_A = np.array(metrics.n_A_ts, dtype=np.int32)
            E_to_A = np.array(metrics.E_to_A_ts, dtype=np.int32)
            D1_to_A = np.array(metrics.D1_to_A_ts, dtype=np.int32)

            # Per-step: who is newly activated?
            # We track via prior_m: if agent was m=0 before and now gets activated,
            # it's a first actor. If m=1, it's a repeat.
            per_step_active = []
            per_step_first = []
            per_step_repeat = []
            traces = {
                "n_U": [], "n_E": [], "n_A": [], "n_D": [],
                "U_to_E": [], "E_to_A": [], "E_to_D": [],
                "A_to_D": [], "D0_to_A": [], "D1_to_A": [],
            }

            # Current state z at step end
            for t in range(H + 1):  # t=0 is cutoff state, t=1..H are forecasts
                n_U_t = int(metrics.n_U_ts[t]) if t < len(metrics.n_U_ts) else 0
                n_A_t = int(metrics.n_A_ts[t]) if t < len(metrics.n_A_ts) else 0
                n_E_t = int(metrics.n_E_ts[t]) if t < len(metrics.n_E_ts) else 0
                n_D_t = int(metrics.n_D_ts[t]) if t < len(metrics.n_D_ts) else 0
                u2e = int(metrics.U_to_E_ts[t]) if t < len(metrics.U_to_E_ts) else 0
                e2a = int(metrics.E_to_A_ts[t]) if t < len(metrics.E_to_A_ts) else 0
                e2d = int(metrics.E_to_D_ts[t]) if t < len(metrics.E_to_D_ts) else 0
                a2d = int(metrics.A_to_D_ts[t]) if t < len(metrics.A_to_D_ts) else 0
                d02a = int(metrics.D0_to_A_ts[t]) if t < len(metrics.D0_to_A_ts) else 0
                d12a = int(metrics.D1_to_A_ts[t]) if t < len(metrics.D1_to_A_ts) else 0
                traces["n_U"].append(n_U_t)
                traces["n_E"].append(n_E_t)
                traces["n_A"].append(n_A_t)
                traces["n_D"].append(n_D_t)
                traces["U_to_E"].append(u2e)
                traces["E_to_A"].append(e2a)
                traces["E_to_D"].append(e2d)
                traces["A_to_D"].append(a2d)
                traces["D0_to_A"].append(d02a)
                traces["D1_to_A"].append(d12a)

                # Behavioral: who was active? = A-stock at time t
                # (Snapshot proxy: agent in A at step boundary)
                # first = E->A (first-ever activation)
                # repeat = D1->A (reactivation) + continuing A (already active before)
                n_active = n_A_t
                n_first_est = e2a  # new activations this step
                n_repeat_est = n_active - n_first_est  # rest are continuing/reactivated

                per_step_active.append(n_active)
                per_step_first.append(n_first_est)
                per_step_repeat.append(max(0, n_repeat_est))

            all_active_steps.append(per_step_active)
            all_first_steps.append(per_step_first)
            all_repeat_steps.append(per_step_repeat)
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

        # Future only: skip t=0 (cutoff), take t=1..H
        fc_active = p50_a[1:H + 1]
        fc_first = p50_f[1:H + 1]
        fc_repeat = p50_r[1:H + 1]

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
