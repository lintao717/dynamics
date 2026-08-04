"""Replay configuration: time grid, network mode, seeds, node limits."""

from __future__ import annotations

from dataclasses import dataclass

from dynamics_simulation.data.networks import ReplayNetworkMode


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration for a single historical replay run.

    All fields are immutable for reproducibility.
    """

    step_hours: float = 1.0
    """Duration of each simulation step in wall-clock hours."""

    tail_steps: int = 4
    """Extra simulation steps appended after the last data step."""

    network_mode: ReplayNetworkMode = ReplayNetworkMode.BROADCAST
    """Network construction mode (broadcast/cumulative/oracle)."""

    seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    """Random seeds for multi-seed aggregation."""

    max_nodes: int = 1000
    """Maximum number of agents per cascade. Larger cases are truncated."""

    truncate_policy: str = "earliest_interactions"
    """Policy for truncation: 'earliest_interactions' keeps earliest users."""

    micro_steps: int = 1
    """V1.3: Sub-steps per 24h macro-step (1 = no micro-stepping, V1.2.1 default)."""

    reactivation_mode: str = "full"
    """V1.5.2: Reactivation mode (full/no_delayed_first/no_true_reactivation/one_shot).
    Controls D0->A and D1->A behaviour via explicit branching in TransitionEngine."""

    broadcast_exposure_config: object | None = None
    """Optional BroadcastExposureConfig override. If None, uses default.
    Allows experiments to inject root_shock or custom exposure profiles
    without modifying the core replay runner."""
