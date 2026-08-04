"""Replay result: serializable output with full provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ReplayRun:
    """Output of one seed's replay."""

    seed: int
    steps: np.ndarray
    active_count: np.ndarray
    cumulative_users: np.ndarray
    n_A_ts: np.ndarray
    n_E_ts: np.ndarray
    n_D_ts: np.ndarray
    o_mean_ts: np.ndarray
    h_mean_ts: np.ndarray
    # V1.3: flow metrics
    actor_flow_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    new_activation_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    reactivation_ts: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ReplayResult:
    """Complete multi-seed replay output with observed data and aggregation.

    Serializable via to_dict().
    """

    case_id: str
    source_dataset: str
    network_mode: str
    step_hours: float
    seeds: tuple[int, ...]
    node_count: int
    interaction_count: int
    tail_steps: int = 4
    last_data_step: int = 0
    truncation_count: int = 0

    # Observed trajectory (from real data)
    observed: Optional[Any] = None  # ObservedTrajectory

    # Per-seed simulation outputs
    per_seed: list[ReplayRun] = field(default_factory=list)

    # Aggregated trajectories (dict[str, np.ndarray])
    simulated_mean: Optional[dict[str, np.ndarray]] = None
    simulated_std: Optional[dict[str, np.ndarray]] = None
    simulated_p5: Optional[dict[str, np.ndarray]] = None
    simulated_p50: Optional[dict[str, np.ndarray]] = None
    simulated_p95: Optional[dict[str, np.ndarray]] = None

    # Metadata
    model_version: str = "v1.2.1"
    assumption_flags: dict[str, bool] = field(default_factory=dict)
    git_sha: str = "unknown"

    # Parameter version
    params_dict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d: dict[str, Any] = {
            "case_id": self.case_id,
            "source_dataset": self.source_dataset,
            "network_mode": self.network_mode,
            "step_hours": self.step_hours,
            "tail_steps": self.tail_steps,
            "last_data_step": self.last_data_step,
            "seeds": list(self.seeds),
            "node_count": self.node_count,
            "interaction_count": self.interaction_count,
            "truncation_count": self.truncation_count,
            "model_version": self.model_version,
            "git_sha": self.git_sha,
            "assumption_flags": self.assumption_flags,
            "params": self.params_dict,
        }

        if self.observed is not None:
            d["observed"] = _trajectory_to_dict(self.observed)

        if self.simulated_mean is not None:
            d["simulated_mean"] = {
                k: _safe_list(v) for k, v in self.simulated_mean.items()
            }
        if self.simulated_std is not None:
            d["simulated_std"] = {
                k: _safe_list(v) for k, v in self.simulated_std.items()
            }
        if self.simulated_p5 is not None:
            d["simulated_p5"] = {
                k: _safe_list(v) for k, v in self.simulated_p5.items()
            }
        if self.simulated_p50 is not None:
            d["simulated_p50"] = {
                k: _safe_list(v) for k, v in self.simulated_p50.items()
            }
        if self.simulated_p95 is not None:
            d["simulated_p95"] = {
                k: _safe_list(v) for k, v in self.simulated_p95.items()
            }

        return d


def _safe_list(arr: np.ndarray) -> list:
    """Convert numpy array to list, handling NaN."""
    return np.where(np.isnan(arr), None, arr).tolist()


def _trajectory_to_dict(obs) -> dict[str, Any]:
    """Serialize ObservedTrajectory to dict."""
    return {
        "steps": obs.steps.tolist(),
        "active_count": obs.active_count.tolist(),
        "cumulative_users": obs.cumulative_users.tolist(),
        "comment_count": obs.comment_count.tolist(),
        "repost_count": obs.repost_count.tolist(),
        "interaction_count": obs.interaction_count.tolist(),
        # V1.3: observed actor count is the flow proxy
        "observed_actor_flow": obs.active_count.tolist(),
    }
