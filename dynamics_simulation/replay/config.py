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

    micro_steps: int = 4
    """V1.3: Sub-steps per 24h macro-step (1 = no micro-stepping)."""
