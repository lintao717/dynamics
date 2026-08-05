"""
MetricsCollector: tracks simulation state over time and computes derived metrics.

Collected metrics per time step:
  - State counts: n_U, n_E, n_A, n_D
  - Opinion statistics: mean, std, polarization index
  - Public opinion bias: B_obs = |ō_private - ō_public|
  - Cross-community flow: κ_ab per community pair
  - Emotion & fatigue: mean levels

At finalize(), computes:
  - Peak active time and magnitude
  - Opinion convergence/divergence
  - Effective propagation threshold R_eff (qualitative)
  - Public-private opinion gap trajectory
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

from dynamics_simulation.agents import AgentState, U, E, A, D


@dataclass
class StepSnapshot:
    """All metrics at one time step."""
    step: int

    # State counts
    n_U: int
    n_E: int
    n_A: int
    n_D: int
    n_aware: int   # n_E + n_A + n_D

    # Opinion (private)
    o_mean: float
    o_std: float
    o_median: float
    o_polarization: float   # bimodality index: high when two clusters exist
    o_positive_frac: float  # fraction with o > 0

    # Opinion (public, among A agents only)
    o_hat_mean: float
    o_hat_std: float
    o_hat_count: int

    # Public opinion bias: |ō_private - ō_public|
    public_bias: float

    # Emotion & fatigue
    h_mean: float
    h_std: float
    f_mean: float
    f_std: float

    # Cross-community flow κ
    kappa: Dict[str, float]  # "a→b" → κ_ab

    # Network metrics
    mean_active_neighbors: float  # avg number of A neighbors per agent


@dataclass
class SimulationMetrics:
    """Complete metrics for one simulation run."""

    config: Dict[str, Any] = field(default_factory=dict)
    snapshots: List[StepSnapshot] = field(default_factory=list)

    # Derived time series (populated at finalize)
    steps: np.ndarray = field(default_factory=lambda: np.array([]))
    n_U_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    n_E_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    n_A_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    n_D_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    o_mean_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    o_std_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    o_polarization_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    o_hat_mean_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    o_hat_std_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    h_mean_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    f_mean_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    public_bias_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    # V1.5.1: per-macro-step transition flow time series
    n_U_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    U_to_E_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    E_to_A_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    E_to_D_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    A_to_D_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    D0_to_A_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    D1_to_A_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    actor_flow_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    new_activation_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    reactivation_ts: np.ndarray = field(default_factory=lambda: np.array([]))

    # Summary statistics
    peak_A: int = 0
    peak_A_step: int = 0
    final_opinion_std: float = 0.0
    max_public_bias: float = 0.0
    total_activations: int = 0
    total_decays: int = 0
    total_reactivations: int = 0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"Simulation Summary:\n"
            f"  Peak active: {self.peak_A} agents at step {self.peak_A_step}\n"
            f"  Final opinion σ: {self.final_opinion_std:.4f}\n"
            f"  Max public bias: {self.max_public_bias:.4f}\n"
            f"  Total activations: {self.total_activations}\n"
            f"  Total decays: {self.total_decays}\n"
            f"  Total reactivations: {self.total_reactivations}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "config": self.config,
            "peak_A": self.peak_A,
            "peak_A_step": self.peak_A_step,
            "final_opinion_std": float(self.final_opinion_std),
            "max_public_bias": float(self.max_public_bias),
            "total_activations": self.total_activations,
            "total_decays": self.total_decays,
            "total_reactivations": self.total_reactivations,
            "time_series": {
                "steps": self.steps.tolist(),
                "n_U": self.n_U_ts.tolist(),
                "n_E": self.n_E_ts.tolist(),
                "n_A": self.n_A_ts.tolist(),
                "n_D": self.n_D_ts.tolist(),
                "o_mean": self.o_mean_ts.tolist(),
                "o_std": self.o_std_ts.tolist(),
                "o_polarization": self.o_polarization_ts.tolist(),
                "h_mean": self.h_mean_ts.tolist(),
                "f_mean": self.f_mean_ts.tolist(),
                "public_bias": self.public_bias_ts.tolist(),
                "o_hat_mean": self.o_hat_mean_ts.tolist(),
                "o_hat_std": self.o_hat_std_ts.tolist(),
                "U_to_E": self.U_to_E_ts.tolist(),
                "E_to_A": self.E_to_A_ts.tolist(),
                "E_to_D": self.E_to_D_ts.tolist(),
                "A_to_D": self.A_to_D_ts.tolist(),
                "D0_to_A": self.D0_to_A_ts.tolist(),
                "D1_to_A": self.D1_to_A_ts.tolist(),
                "actor_flow": self.actor_flow_ts.tolist(),
                "new_activation": self.new_activation_ts.tolist(),
                "reactivation": self.reactivation_ts.tolist(),
            },
        }


class MetricsCollector:
    """Collects and processes simulation metrics."""

    def __init__(self):
        self._snapshots: List[dict] = []
        self._prev_z: Optional[np.ndarray] = None
        # Accumulated transition counts (from TransitionEvents)
        self._total_U_to_E: int = 0
        self._total_E_to_A: int = 0
        self._total_E_to_D: int = 0
        self._total_A_to_D: int = 0
        self._total_D0_to_A: int = 0
        self._total_D1_to_A: int = 0
        # V1.5.1: per-macro-step flow time series (all 6 transition types)
        self._U_to_E_ts: list[int] = []
        self._E_to_A_ts: list[int] = []
        self._E_to_D_ts: list[int] = []
        self._A_to_D_ts: list[int] = []
        self._D0_to_A_ts: list[int] = []
        self._D1_to_A_ts: list[int] = []

    def record(
        self,
        state: AgentState,
        communities: Optional[Dict[int, List[int]]] = None,
        G_s: Optional[np.ndarray] = None,
        G_o: Optional[np.ndarray] = None,
        events: Optional[Any] = None,
        t: Optional[int] = None,
        snapshot: bool = True,
    ) -> None:
        """Record state. Events are accumulated every call. Snapshot is optional.

        Args:
            state: Current agent state.
            communities: Community membership dict.
            G_s: Propagation network (for κ computation).
            G_o: Opinion influence network (for κ computation).
            events: Optional TransitionEvents from this step.
            t: Actual simulation step index.
            snapshot: If True, save a full snapshot. If False, only accumulate events.
        """
        # ── Always accumulate transition events ──
        if events is not None:
            self._total_U_to_E += events.U_to_E
            self._total_E_to_A += events.E_to_A
            self._total_E_to_D += events.E_to_D
            self._total_A_to_D += events.A_to_D
            self._total_D0_to_A += events.D0_to_A
            self._total_D1_to_A += events.D1_to_A

        # V1.5.1: record per-step flow for ALL transition types
        if snapshot and events is not None:
            self._U_to_E_ts.append(events.U_to_E)
            self._E_to_A_ts.append(events.E_to_A)
            self._E_to_D_ts.append(events.E_to_D)
            self._A_to_D_ts.append(events.A_to_D)
            self._D0_to_A_ts.append(events.D0_to_A)
            self._D1_to_A_ts.append(events.D1_to_A)

        if not snapshot:
            self._prev_z = state.z.copy()
            return

        n = state.n
        step = t if t is not None else len(self._snapshots)

        # ── Opinion polarization index ──
        # Bimodality: high when there are two distinct clusters
        # Using variance ratio of k-means (k=2) approximation:
        # If |o| distribution is bimodal, σ² of |o| is low relative to σ² of o
        o_abs = np.abs(state.o)
        if state.o.std() > 1e-8:
            polarization = 1.0 - (o_abs.std() / state.o.std())
        else:
            polarization = 0.0

        # ── Public opinion bias ──
        a_mask = state.z == A
        if a_mask.sum() > 0:
            o_hat_mean = float(np.nanmean(state.o_hat[a_mask]))
            o_hat_std = float(np.nanstd(state.o_hat[a_mask]))
            o_hat_count = int(a_mask.sum())
            public_bias = abs(state.o.mean() - o_hat_mean)
        else:
            o_hat_mean = float('nan')
            o_hat_std = float('nan')
            o_hat_count = 0
            public_bias = 0.0

        # ── Dynamic kappa: computed from actual U->E events this step ──
        kappa = {}
        if communities is not None and G_s is not None and len(communities) > 1 and events is not None:
            kappa = _compute_dynamic_kappa(events, G_s, communities, state, self._prev_z)
            if not kappa:  # fallback to static if no events
                kappa = _compute_kappa(G_s, communities)
        elif communities is not None and G_s is not None and len(communities) > 1:
            kappa = _compute_kappa(G_s, communities)

        # ── Mean active neighbors ──
        if G_s is not None:
            active_vec = (state.z == A).astype(np.float64)
            active_neighbors = G_s.dot(active_vec)
            mean_active_neighbors = float(active_neighbors.mean())
        else:
            mean_active_neighbors = 0.0

        # ── Track state transitions ──
        snap = {
            "step": step,
            "n_U": state.n_U,
            "n_E": state.n_E,
            "n_A": state.n_A,
            "n_D": state.n_D,
            "n_aware": state.n_E + state.n_A + state.n_D,
            "o_mean": float(state.o.mean()),
            "o_std": float(state.o.std()),
            "o_median": float(np.median(state.o)),
            "o_polarization": float(polarization),
            "o_positive_frac": float((state.o > 0).mean()),
            "o_hat_mean": o_hat_mean,
            "o_hat_std": o_hat_std,
            "o_hat_count": o_hat_count,
            "public_bias": float(public_bias),
            "h_mean": float(state.h.mean()),
            "h_std": float(state.h.std()),
            "f_mean": float(state.f.mean()),
            "f_std": float(state.f.std()),
            "kappa": kappa,
            "mean_active_neighbors": mean_active_neighbors,
        }
        self._snapshots.append(snap)

        # ── Track state for next step's comparison ──
        self._prev_z = state.z.copy()

    def finalize(self) -> None:
        """Compute derived metrics after simulation completes."""
        if not self._snapshots:
            return

        # Convert snapshots to numpy arrays for the metrics object
        pass  # Done in as_metrics()

    def as_metrics(self) -> SimulationMetrics:
        """Convert collected snapshots to a SimulationMetrics object."""
        if not self._snapshots:
            return SimulationMetrics()

        n_steps = len(self._snapshots)

        metrics = SimulationMetrics()
        metrics.steps = np.array([s["step"] for s in self._snapshots])
        metrics.n_U_ts = np.array([s["n_U"] for s in self._snapshots])
        metrics.n_E_ts = np.array([s["n_E"] for s in self._snapshots])
        metrics.n_A_ts = np.array([s["n_A"] for s in self._snapshots])
        metrics.n_D_ts = np.array([s["n_D"] for s in self._snapshots])
        metrics.o_mean_ts = np.array([s["o_mean"] for s in self._snapshots])
        metrics.o_std_ts = np.array([s["o_std"] for s in self._snapshots])
        metrics.o_polarization_ts = np.array([s["o_polarization"] for s in self._snapshots])
        metrics.h_mean_ts = np.array([s["h_mean"] for s in self._snapshots])
        metrics.f_mean_ts = np.array([s["f_mean"] for s in self._snapshots])
        metrics.public_bias_ts = np.array([s["public_bias"] for s in self._snapshots])
        # V2.0-C: public expression time series
        o_hat_vals = [s.get("o_hat_mean", 0.0) for s in self._snapshots]
        metrics.o_hat_mean_ts = np.array([v if not (isinstance(v, float) and np.isnan(v)) else 0.0 for v in o_hat_vals])
        o_hat_std_vals = [s.get("o_hat_std", 0.0) for s in self._snapshots]
        metrics.o_hat_std_ts = np.array([v if not (isinstance(v, float) and np.isnan(v)) else 0.0 for v in o_hat_std_vals])

        # ── V1.5.1: transition flow time series ──
        def _pad(arr_list, target_len):
            n = len(arr_list)
            if n == 0: return np.zeros(target_len, dtype=np.int32)
            if n < target_len:
                return np.array([0]*(target_len-n) + arr_list, dtype=np.int32)
            return np.array(arr_list[:target_len], dtype=np.int32)

        n_snaps = len(self._snapshots)
        metrics.n_U_ts = metrics.n_U_ts  # already populated above
        metrics.U_to_E_ts = _pad(self._U_to_E_ts, n_snaps)
        metrics.E_to_A_ts = _pad(self._E_to_A_ts, n_snaps)
        metrics.E_to_D_ts = _pad(self._E_to_D_ts, n_snaps)
        metrics.A_to_D_ts = _pad(self._A_to_D_ts, n_snaps)
        metrics.D0_to_A_ts = _pad(self._D0_to_A_ts, n_snaps)
        metrics.D1_to_A_ts = _pad(self._D1_to_A_ts, n_snaps)
        metrics.actor_flow_ts = metrics.E_to_A_ts + metrics.D1_to_A_ts
        metrics.new_activation_ts = metrics.E_to_A_ts.copy()
        metrics.reactivation_ts = metrics.D1_to_A_ts.copy()

        # ── Summary statistics ──
        metrics.peak_A = int(metrics.n_A_ts.max())
        metrics.peak_A_step = int(metrics.steps[metrics.n_A_ts.argmax()])
        metrics.final_opinion_std = float(metrics.o_std_ts[-1])
        metrics.max_public_bias = float(metrics.public_bias_ts.max())

        # ── Transition counts from accumulated TransitionEvents (exact, not net-delta) ──
        metrics.total_activations = (
            self._total_E_to_A + self._total_D0_to_A + self._total_D1_to_A
        )
        metrics.total_decays = self._total_A_to_D
        metrics.total_reactivations = self._total_D1_to_A
        metrics.total_delayed_activations = self._total_D0_to_A

        # Count total transitions from tracking
        metrics.snapshots = [
            StepSnapshot(**{k: v for k, v in s.items() if k in StepSnapshot.__dataclass_fields__})
            for s in self._snapshots
        ]
        metrics.config = {"n_steps": n_steps}

        return metrics


def _compute_dynamic_kappa(events, G_s, communities, state, prev_z):
    """Compute kappa from actual U->E transition events this step.

    kappa_ab = #{U->E events from community a to community b} /
               #{U->E events originating from community a}
    """
    if events is None or events.U_to_E == 0:
        return {}
    if prev_z is None:
        return {}

    # Identify newly exposed agents (those that were U and are now E, A, or D)
    newly_aware = (prev_z == 0) & (state.z != 0)  # 0 = U
    if not newly_aware.any():
        return {}

    node_to_block = np.zeros(G_s.shape[0], dtype=np.int32)
    for bid, nodes in communities.items():
        for node in nodes:
            node_to_block[node] = bid

    # For simplicity: attribute exposure to active neighbors
    active = state.z == 2  # A
    kappa = {}
    eps = 1e-8
    block_ids = sorted(communities.keys())

    for a in block_ids:
        total_from_a = 0
        for b in block_ids:
            cross = 0
            # Count active agents in a that have edges to newly_aware in b
            a_mask = np.zeros(G_s.shape[0], dtype=bool)
            a_mask[list(communities[a])] = True
            b_newly_aware = newly_aware.copy()
            b_newly_aware[~np.isin(np.arange(G_s.shape[0]), communities[b])] = False
            if a_mask.any() and b_newly_aware.any():
                cross = G_s[np.ix_(np.where(b_newly_aware)[0], np.where(a_mask & active)[0])].sum()
                total_from_a += cross
            kappa[f"{a}->{b}"] = float(cross)
        # Normalize
        for b in block_ids:
            kappa[f"{a}->{b}"] = float(kappa[f"{a}->{b}"] / (total_from_a + eps))

    return kappa


def _compute_kappa(
    G_s: np.ndarray,
    communities: Dict[int, List[int]],
) -> Dict[str, float]:
    """Compute cross-community information flow coefficients.

    κ_ab = |E_{a→b}| / (|E_a^{out}| + ε)

    where E_{a→b} are edges from community a to community b
    in the information_flow_view (column j → row i = information flow from j to i).

    In G_s[i, j], the edge is j→i meaning information flows from j to i.
    So E_{a→b} = sum of G_s[i,j] for j∈C_a, i∈C_b.
    """
    kappa = {}
    eps = 1e-8

    block_ids = sorted(communities.keys())
    n_blocks = len(block_ids)

    # Map node → community
    node_to_block = np.zeros(G_s.shape[0], dtype=np.int32)
    for bid, nodes in communities.items():
        for node in nodes:
            node_to_block[node] = bid

    # Compute total out-edges per block
    total_out = {}
    for a in block_ids:
        a_nodes = communities[a]
        # Out-edges from block a = sum of edges where source is in a
        # G_s[i, j] = edge j→i. Source=j, target=i.
        # Out from a = sum over j∈C_a of all out-edges
        total_out[a] = G_s[:, a_nodes].sum() + eps

    # Compute cross-block edges
    for a in block_ids:
        for b in block_ids:
            a_nodes = communities[a]
            b_nodes = communities[b]
            # Edges from a→b: source j∈C_a, target i∈C_b
            # G_s[i, j] for i∈C_b, j∈C_a
            cross = G_s[np.ix_(b_nodes, a_nodes)].sum()
            kappa[f"{a}→{b}"] = float(cross / total_out[a])

    return kappa
