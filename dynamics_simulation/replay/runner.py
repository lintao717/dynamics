"""
Replay runner: orchestrates EventCase → simulation → ReplayResult.

Pipeline:
  EventCase → NodeIndex → TimeGrid → ObservedTrajectory
           → initial AgentState → network provider
           → input timeline → SimulationRunner → ReplayRun

Multi-seed aggregation computes mean, std, and 5th/50th/95th percentiles.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from dynamics_simulation.config import ModelParams
from dynamics_simulation.simulation import SimulationConfig, SimulationRunner
from dynamics_simulation.data.schema import EventCase
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.observations import (
    ObservedTrajectory, build_observed_trajectory,
)
from dynamics_simulation.data.state import build_initial_state
from dynamics_simulation.data.networks import (
    build_network_provider, ReplayNetworkMode,
)
from dynamics_simulation.data.timeline import EventInputTimeline
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.result import ReplayResult, ReplayRun


def _run_one_seed(
    case: EventCase,
    index: NodeIndex,
    grid: TimeGrid,
    params: ModelParams,
    config: ReplayConfig,
    seed: int,
) -> ReplayRun:
    """Run one seed of the replay pipeline."""
    rng = np.random.default_rng(seed)

    # Build initial state from case
    state = build_initial_state(case, index, params, rng)

    # Build temporal network provider
    net_provider = build_network_provider(
        case, index, grid, mode=config.network_mode,
    )

    def network_fn(step: int):
        snap = net_provider.snapshot_at(step)
        return snap.G_s, snap.G_o, snap.communities

    # Build input timeline — use custom BroadcastExposureConfig if provided
    timeline_cfg = getattr(config, 'broadcast_exposure_config', None)
    if timeline_cfg is not None:
        from dynamics_simulation.data.timeline import BroadcastExposureConfig
        if isinstance(timeline_cfg, dict):
            timeline_cfg = BroadcastExposureConfig(**timeline_cfg)
        timeline = EventInputTimeline(case, index, grid, timeline_cfg)
    else:
        timeline = EventInputTimeline(case, index, grid)

    def input_fn(n: int, t: int, T: int, micro_step: int = 0, micro_total: int = 1):
        # timeline uses fixed time constants — no future-data dependency.
        # root_shock is applied at micro-step level within macro-step 0.
        return timeline.inputs_at(n, t, micro_step, micro_total)

    # Configure and run simulation
    sim_cfg = SimulationConfig(
        n_agents=len(index),
        T=grid.final_step,
        micro_steps=config.micro_steps,  # V1.3: configurable micro-stepping
        seed=seed,
        params=params,
        initial_state=state,
        network_provider=network_fn,
        input_fn=input_fn,
        verbose=False,
    )
    runner = SimulationRunner(sim_cfg)
    metrics = runner.run()

    return ReplayRun(
        seed=seed,
        steps=metrics.steps.copy(),
        active_count=metrics.n_A_ts.copy(),
        cumulative_users=np.array([]),
        n_A_ts=metrics.n_A_ts.copy(),
        n_E_ts=metrics.n_E_ts.copy(),
        n_D_ts=metrics.n_D_ts.copy(),
        o_mean_ts=metrics.o_mean_ts.copy(),
        h_mean_ts=metrics.h_mean_ts.copy(),
        # V1.3: flow metrics
        actor_flow_ts=getattr(metrics, 'actor_flow_ts',
                              np.zeros_like(metrics.n_A_ts)).copy(),
        new_activation_ts=getattr(metrics, 'new_activation_ts',
                                  np.zeros_like(metrics.n_A_ts)).copy(),
        reactivation_ts=getattr(metrics, 'reactivation_ts',
                                np.zeros_like(metrics.n_A_ts)).copy(),
    )


def _aggregate_seeds(runs: list[ReplayRun]) -> dict[str, dict[str, np.ndarray]]:
    """Compute mean, std, and percentiles across seeds."""
    if not runs:
        return {}

    # Collect per-seed trajectories for each metric
    metric_names = [
        "active_count", "n_A_ts", "n_E_ts", "n_D_ts",
        "o_mean_ts", "h_mean_ts",
        # V1.3: flow metrics
        "actor_flow_ts", "new_activation_ts", "reactivation_ts",
    ]

    result: dict[str, dict[str, np.ndarray]] = {
        "mean": {}, "std": {}, "p5": {}, "p50": {}, "p95": {},
    }

    for name in metric_names:
        stacked = np.stack([getattr(r, name) for r in runs], axis=0)
        result["mean"][name] = stacked.mean(axis=0)
        result["std"][name] = stacked.std(axis=0)
        result["p5"][name] = np.percentile(stacked, 5, axis=0)
        result["p50"][name] = np.percentile(stacked, 50, axis=0)
        result["p95"][name] = np.percentile(stacked, 95, axis=0)

    return result


def run_replay(
    case: EventCase,
    params: ModelParams,
    config: ReplayConfig | None = None,
) -> ReplayResult:
    """Run a complete multi-seed historical replay.

    Args:
        case: Validated EventCase to replay.
        params: Model parameters for the simulation.
        config: Replay configuration (uses defaults if None).

    Returns:
        ReplayResult with observed trajectory, per-seed runs, and
        aggregated statistics.
    """
    if config is None:
        config = ReplayConfig()

    case.validate()

    # Truncate if needed
    truncation_count = 0
    if len(case.user_ids) > config.max_nodes:
        # Keep root + first (max_nodes-1) users by earliest interaction
        keep_users = set(case.user_ids[:config.max_nodes])
        truncation_count = len(case.user_ids) - config.max_nodes
        filtered = tuple(
            ix for ix in case.interactions
            if ix.user_id in keep_users
        )
        case = EventCase(
            case_id=case.case_id,
            source_dataset=case.source_dataset,
            root=case.root,
            interactions=filtered,
            metadata=dict(case.metadata, truncated_from=len(case.user_ids)),
        )

    # Build pipeline components (shared across seeds)
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(
        case, step_hours=config.step_hours, tail_steps=config.tail_steps,
    )
    observed = build_observed_trajectory(case, index, grid)

    # Run each seed
    per_seed: list[ReplayRun] = []
    for seed in config.seeds:
        run_result = _run_one_seed(case, index, grid, params, config, seed)
        per_seed.append(run_result)

    # Aggregate
    agg = _aggregate_seeds(per_seed)

    # Build full params dict (all fields, not just 6)
    from dataclasses import asdict
    params_dict = asdict(params)

    # Resolve git commit at replay time
    import subprocess
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    assumption_flags = {
        # ── Network-level guarantees ──
        "no_future_edge_leakage": (
            config.network_mode != ReplayNetworkMode.ORACLE_STATIC
        ),
        # ── Participant cohort — NodeIndex is built from all event
        #     participants, so the model knows the final cohort at t=0.
        "future_participant_cohort_known": True,
        "cohort_conditioned_replay": True,
        "causal_forecast": False,
        # ── Mode metadata ──
        "broadcast_primary": (
            config.network_mode == ReplayNetworkMode.BROADCAST
        ),
        "oracle_is_upper_bound": (
            config.network_mode == ReplayNetworkMode.ORACLE_STATIC
        ),
        # ── Latent variables ──
        "latent_E": True,
        "latent_private_opinion": True,
    }

    return ReplayResult(
        case_id=case.case_id,
        source_dataset=case.source_dataset,
        network_mode=config.network_mode.value,
        step_hours=config.step_hours,
        tail_steps=config.tail_steps,
        last_data_step=grid.last_data_step,
        seeds=config.seeds,
        node_count=len(index),
        interaction_count=len(case.interactions),
        truncation_count=truncation_count,
        observed=observed,
        per_seed=per_seed,
        simulated_mean=agg.get("mean", {}),
        simulated_std=agg.get("std", {}),
        simulated_p5=agg.get("p5", {}),
        simulated_p50=agg.get("p50", {}),
        simulated_p95=agg.get("p95", {}),
        params_dict=params_dict,
        assumption_flags=assumption_flags,
        git_sha=git_sha,
    )
