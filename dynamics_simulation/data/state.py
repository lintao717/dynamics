"""
Real-data initial AgentState builder.

Constructs AgentState from an EventCase + NodeIndex, optionally
enriched with precomputed text signals (stance, arousal). No NLP
model is run here; signals are precomputed inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from numpy.random import Generator

from dynamics_simulation.agents import (
    AgentState, AgentAttributes, initialize_agents,
    U, E, A, D,
)
from dynamics_simulation.config import ModelParams
from dynamics_simulation.data.schema import EventCase
from dynamics_simulation.data.indexing import NodeIndex


@dataclass(frozen=True)
class TextSignals:
    """Precomputed stance and arousal values keyed by user ID.

    Values are validated on construction: stance ∈ [-1, 1],
    arousal ∈ [0, 1].
    """

    stance_by_user: Mapping[str, float]
    arousal_by_user: Mapping[str, float]

    def __post_init__(self):
        for uid, val in self.stance_by_user.items():
            if not (-1.0 <= val <= 1.0):
                raise ValueError(
                    f"stance for {uid!r} = {val} is outside [-1, 1]"
                )
        for uid, val in self.arousal_by_user.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"arousal for {uid!r} = {val} is outside [0, 1]"
                )


@dataclass(frozen=True)
class StatePriorConfig:
    """Configuration for state initialization from real data."""

    initial_opinion_dist: str = "moderate"
    root_arousal_if_missing: float = 0.5
    nonroot_arousal_if_missing: float = 0.0


def build_initial_state(
    case: EventCase,
    index: NodeIndex,
    params: ModelParams,
    rng: Generator,
    signals: Optional[TextSignals] = None,
    priors: Optional[StatePriorConfig] = None,
) -> AgentState:
    """Build an AgentState grounded in an EventCase.

    Rules:
      - Root author: z=A, m=1, o_hat set to supplied stance or sampled
        private opinion.
      - All other users: z=U, m=0, o_hat=NaN.
      - No user starts in E.
      - f=0 for all users.
      - Missing stance/arousal uses NaN with mask=False (never 0).

    Args:
        case: Validated EventCase.
        index: NodeIndex for the case.
        params: Model parameters (used for attribute distributions).
        rng: NumPy random generator.
        signals: Optional precomputed stance/arousal signals.
        priors: Optional state prior configuration.

    Returns:
        An AgentState initialized from the real data.

    Raises:
        ValueError: If signals reference users not in the NodeIndex.
    """
    case.validate()
    if priors is None:
        priors = StatePriorConfig()

    N = len(index)

    # Validate signals against index
    if signals is not None:
        for uid in signals.stance_by_user:
            if uid not in index.user_to_idx:
                raise ValueError(f"Signal user {uid!r} not in NodeIndex")
        for uid in signals.arousal_by_user:
            if uid not in index.user_to_idx:
                raise ValueError(f"Signal user {uid!r} not in NodeIndex")

    # Use existing heterogeneous initialization for attrs, o, h, f
    state = initialize_agents(
        n=N,
        initial_active=0,  # We'll set root manually
        initial_opinion_dist=priors.initial_opinion_dist,
        rng=rng,
        opinion_params=params.opinion,
    )

    # ── Root author ──
    root_idx = index.user_to_idx[case.root.user_id]
    state.z[root_idx] = A
    state.m[root_idx] = 1

    if signals is not None and case.root.user_id in signals.stance_by_user:
        stance = float(signals.stance_by_user[case.root.user_id])
        state.o_hat[root_idx] = float(np.clip(stance, -1.0, 1.0))
    else:
        # Default: express private opinion (no text signal available)
        state.o_hat[root_idx] = float(state.o[root_idx])

    if signals is not None and case.root.user_id in signals.arousal_by_user:
        state.h[root_idx] = float(
            np.clip(signals.arousal_by_user[case.root.user_id], 0.0, 1.0)
        )
    else:
        state.h[root_idx] = float(priors.root_arousal_if_missing)

    # ── Non-root users: all U ──
    for uid in index.idx_to_user:
        if uid == case.root.user_id:
            continue
        idx = index.user_to_idx[uid]
        state.z[idx] = U
        state.m[idx] = 0
        # o_hat stays NaN (already from initialize_agents with initial_active=0)
        state.h[idx] = float(priors.nonroot_arousal_if_missing)

    # ── Fatigue ──
    state.f[:] = 0.0

    return state
