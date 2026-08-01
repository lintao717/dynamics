"""
Temporal network providers for historical replay.

Three modes with explicit no-leak guarantees:

  broadcast (PRIMARY):
    G_s = 0 everywhere; exposure enters through media_exposure.
    No future information.

  cumulative_interaction:
    Root->user edge only after that user's first observed action.
    The edge may influence later steps but cannot explain the first action.

  oracle_static (SENSITIVITY ONLY):
    All observed root->user edges exist from step 0.
    Upper-bound sensitivity run — NOT causal validation.

All networks follow G[dst, src] > 0 convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from dynamics_simulation.data.schema import EventCase
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid


class ReplayNetworkMode(str, Enum):
    BROADCAST = "broadcast"
    CUMULATIVE_INTERACTION = "cumulative_interaction"
    ORACLE_STATIC = "oracle_static"


@dataclass(frozen=True)
class NetworkSnapshot:
    """One time-step's adjacency matrices.

    G_s: propagation network (N×N, G[dst, src]).
    G_o: opinion influence network (N×N, row-normalized).
    communities: community_id → list of node indices.
    """

    G_s: np.ndarray
    G_o: np.ndarray
    communities: Mapping[int, list[int]] = field(default_factory=dict)

    def __post_init__(self):
        G_s = np.asarray(self.G_s, dtype=np.float64)
        G_o = np.asarray(self.G_o, dtype=np.float64)

        if G_s.ndim != 2 or G_o.ndim != 2:
            raise ValueError("G_s and G_o must be 2-D")
        if G_s.shape != G_o.shape:
            raise ValueError(
                f"Shape mismatch: G_s {G_s.shape} vs G_o {G_o.shape}"
            )
        if np.any(np.isnan(G_s)) or np.any(np.isinf(G_s)):
            raise ValueError("G_s contains NaN or Inf")
        if np.any(np.isnan(G_o)) or np.any(np.isinf(G_o)):
            raise ValueError("G_o contains NaN or Inf")
        if np.any(G_s < 0) or np.any(G_o < 0):
            raise ValueError("Network weights must be non-negative")


class TemporalNetworkProvider:
    """Provides NetworkSnapshots for each simulation step.

    Immutable after construction. snapshot_at(step) is deterministic.
    """

    def __init__(self, snapshots: list[NetworkSnapshot]):
        self._snapshots = tuple(snapshots)

    def snapshot_at(self, step: int) -> NetworkSnapshot:
        """Return the network snapshot for *step*.

        If step exceeds the stored range, returns the last snapshot
        (tail steps after the last data step).
        """
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        idx = min(step, len(self._snapshots) - 1)
        return self._snapshots[idx]

    @property
    def max_step(self) -> int:
        return len(self._snapshots) - 1


def _row_normalize(adj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-normalize so each row sums to 0 or 1."""
    row_sums = adj.sum(axis=1, keepdims=True)
    return adj / np.maximum(row_sums, eps)


def build_network_provider(
    case: EventCase,
    index: NodeIndex,
    grid: TimeGrid,
    mode: ReplayNetworkMode = ReplayNetworkMode.BROADCAST,
) -> TemporalNetworkProvider:
    """Build a TemporalNetworkProvider for *case* under *mode*.

    Args:
        case: Validated EventCase.
        index: NodeIndex for the case.
        grid: TimeGrid for the case.
        mode: Replay network construction mode.

    Returns:
        A TemporalNetworkProvider whose snapshot_at(step) returns the
        appropriate networks for that step with no future leakage.
    """
    case.validate()
    N = len(index)
    T = grid.final_step
    root_idx = 0  # guaranteed by NodeIndex

    snapshots: list[NetworkSnapshot] = []

    if mode == ReplayNetworkMode.BROADCAST:
        # G_s is always zero; G_o starts empty and stays empty
        for step in range(T + 1):
            snapshots.append(NetworkSnapshot(
                G_s=np.zeros((N, N), dtype=np.float64),
                G_o=np.zeros((N, N), dtype=np.float64),
                communities={0: list(range(N))},
            ))

    elif mode == ReplayNetworkMode.CUMULATIVE_INTERACTION:
        accumulated = np.zeros((N, N), dtype=np.float64)
        for step in range(T + 1):
            # Add edges for interactions that occur in this step
            for ix in case.interactions:
                ix_step = grid.step_of(ix.timestamp)
                if ix_step == step:
                    user_idx = index.user_to_idx[ix.user_id]
                    # Root → user edge (G[dst=user, src=root])
                    accumulated[user_idx, root_idx] = 1.0

            G_o = _row_normalize(accumulated.copy())
            snapshots.append(NetworkSnapshot(
                G_s=accumulated.copy(),
                G_o=G_o,
                communities={0: list(range(N))},
            ))

    elif mode == ReplayNetworkMode.ORACLE_STATIC:
        # All root→user edges from step 0
        full = np.zeros((N, N), dtype=np.float64)
        seen = {case.root.user_id}
        for ix in case.interactions:
            uid = ix.user_id
            if uid not in seen:
                user_idx = index.user_to_idx[uid]
                full[user_idx, root_idx] = 1.0
                seen.add(uid)

        G_o = _row_normalize(full.copy())
        snap = NetworkSnapshot(
            G_s=full.copy(),
            G_o=G_o,
            communities={0: list(range(N))},
        )
        snapshots = [snap] * (T + 1)

    else:
        raise ValueError(f"Unknown network mode: {mode}")

    return TemporalNetworkProvider(snapshots)
