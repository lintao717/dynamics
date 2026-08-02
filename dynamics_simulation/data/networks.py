"""
Temporal network providers for historical replay.

Three modes with explicit no-leak guarantees:

  broadcast (PRIMARY):
    G_s = 0 everywhere; exposure enters through media_exposure.
    G_o = 0 everywhere; opinion influence via observed interaction
    graph is not modelled in broadcast mode. (A future extension may
    derive G_o from comment relationships for opinion dynamics.)
    Single shared snapshot — no per-step allocation.

  cumulative_interaction:
    Root→user edge only after that user's first observed action.
    Computed on demand from an edge schedule — no O(T·N²) storage.

  oracle_static (SENSITIVITY ONLY):
    All observed root→user edges exist from step 0.
    Single shared snapshot — upper-bound sensitivity, NOT causal.

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
    """Abstract temporal network provider.

    Subclasses implement lazy evaluation so per-step snapshots
    are not pre-allocated O(T·N²).
    """

    def snapshot_at(self, step: int) -> NetworkSnapshot:
        """Return the network snapshot for *step*."""
        raise NotImplementedError


def _row_normalize(adj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-normalize so each row sums to 0 or 1."""
    row_sums = adj.sum(axis=1, keepdims=True)
    return adj / np.maximum(row_sums, eps)


# ── Lazy providers ──


class BroadcastProvider(TemporalNetworkProvider):
    """Broadcast mode: G_s = 0, G_o = 0 always. Single shared snapshot.

    G_o is kept at zero because the observed interaction graph
    (comments/reposts) is not used for opinion influence in broadcast
    mode. A future extension may derive G_o from comment relationships.
    Memory: O(N²) — one snapshot, regardless of T.
    """

    def __init__(self, n: int):
        self._snapshot = NetworkSnapshot(
            G_s=np.zeros((n, n), dtype=np.float64),
            G_o=np.zeros((n, n), dtype=np.float64),
            communities={0: list(range(n))},
        )

    def snapshot_at(self, step: int) -> NetworkSnapshot:
        """Same zero snapshot for every step."""
        return self._snapshot


class CumulativeProvider(TemporalNetworkProvider):
    """Cumulative interaction mode: edges accumulate on demand.

    Each call to snapshot_at(step) computes the network by replaying
    all edges up to and including *step*. No snapshot history is stored.

    Memory: O(N² + I) where I = number of interactions.
    """

    def __init__(
        self,
        n: int,
        edges: list[tuple[int, int, int]],  # (step, dst_idx, src_idx)
        communities: Mapping[int, list[int]],
    ):
        self._n = n
        self._edges = edges  # pre-sorted by step
        self._communities = communities or {0: list(range(n))}

    def snapshot_at(self, step: int) -> NetworkSnapshot:
        """Compute the accumulated network up to *step* on the fly."""
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")

        accumulated = np.zeros((self._n, self._n), dtype=np.float64)

        for edge_step, dst_idx, src_idx in self._edges:
            if edge_step <= step:
                accumulated[dst_idx, src_idx] = 1.0
            else:
                break  # edges are pre-sorted by step

        G_o = _row_normalize(accumulated.copy())
        return NetworkSnapshot(
            G_s=accumulated,
            G_o=G_o,
            communities=self._communities,
        )


class OracleStaticProvider(TemporalNetworkProvider):
    """Oracle mode: all edges from step 0. Single shared snapshot.

    Memory: O(N²) — one snapshot.
    """

    def __init__(self, snapshot: NetworkSnapshot):
        self._snapshot = snapshot

    def snapshot_at(self, step: int) -> NetworkSnapshot:
        """Same full-network snapshot for every step."""
        return self._snapshot


# ── Public builder ──


def build_network_provider(
    case: EventCase,
    index: NodeIndex,
    grid: TimeGrid,
    mode: ReplayNetworkMode = ReplayNetworkMode.BROADCAST,
) -> TemporalNetworkProvider:
    """Build a lazy TemporalNetworkProvider for *case* under *mode*.

    Args:
        case: Validated EventCase.
        index: NodeIndex for the case.
        grid: TimeGrid for the case (used only for validation, not storage).
        mode: Replay network construction mode.

    Returns:
        A TemporalNetworkProvider whose snapshot_at(step) returns the
        appropriate networks with no future edge leakage and O(N²) memory
        (not O(T·N²)).
    """
    case.validate()
    N = len(index)
    root_idx = 0  # guaranteed by NodeIndex

    if mode == ReplayNetworkMode.BROADCAST:
        return BroadcastProvider(N)

    elif mode == ReplayNetworkMode.CUMULATIVE_INTERACTION:
        # Build an edge schedule: (step, user_idx, root_idx) sorted by step
        edge_schedule: list[tuple[int, float]] = []
        seen: set[str] = set()

        for ix in case.interactions:
            if ix.user_id not in seen:
                seen.add(ix.user_id)
                user_idx = index.user_to_idx[ix.user_id]
                ix_step = grid.step_of(ix.timestamp)
                edge_schedule.append((ix_step, user_idx, root_idx))

        # Already sorted by step because interactions are sorted by timestamp
        # But ensure determinism
        edge_schedule.sort(key=lambda e: e[0])

        return CumulativeProvider(
            n=N,
            edges=edge_schedule,  # [(step, dst_idx, src_idx), ...]
            communities={0: list(range(N))},
        )

    elif mode == ReplayNetworkMode.ORACLE_STATIC:
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
        return OracleStaticProvider(snap)

    else:
        raise ValueError(f"Unknown network mode: {mode}")
