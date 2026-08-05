"""Observed trajectories: time-aligned, masked observations from real data.

Missing stance/arousal is always NaN with mask=False, never 0 with mask=True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from dynamics_simulation.data.schema import EventCase
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid


@dataclass(frozen=True)
class ObservedTrajectory:
    """Aggregate observed quantities per simulation step.

    All arrays have shape (T+1,) where T = grid.final_step.
    active_mask has shape (T+1, N). Missing values are NaN.
    """

    steps: np.ndarray          # int, [0, 1, ..., T]
    active_count: np.ndarray   # number of users active in each step
    cumulative_users: np.ndarray  # monotonic cumulative unique users
    comment_count: np.ndarray  # comments per step
    repost_count: np.ndarray   # reposts per step
    interaction_count: np.ndarray  # total interactions per step
    active_mask: np.ndarray    # (T+1, N) bool
    stance_mean: np.ndarray    # NaN where unavailable
    arousal_mean: np.ndarray   # NaN where unavailable
    observation_masks: Mapping[str, np.ndarray]
    # V1.6.1: actor decomposition
    first_actor_count: np.ndarray = field(default_factory=lambda: np.array([]))
    """Users whose first-ever interaction occurs in this step."""
    repeat_actor_count: np.ndarray = field(default_factory=lambda: np.array([]))
    """Users who already interacted in a previous step and interact again."""


def build_observed_trajectory(
    case: EventCase,
    index: NodeIndex,
    grid: TimeGrid,
) -> ObservedTrajectory:
    """Build an ObservedTrajectory from an EventCase.

    Rules:
      - Root author is active at step 0.
      - A user is observed active in every step with ≥1 interaction.
      - Multiple actions by one user in one step: count once in
        active_count, all in interaction_count/comment_count/repost_count.
      - cumulative_users is monotonic.
      - Missing stance/arousal is NaN with mask=False.
    """
    N = len(index)
    T = grid.final_step
    Tp1 = T + 1

    steps = np.arange(Tp1, dtype=np.int32)
    active_mask = np.zeros((Tp1, N), dtype=bool)
    comment_count = np.zeros(Tp1, dtype=np.int32)
    repost_count = np.zeros(Tp1, dtype=np.int32)
    interaction_count = np.zeros(Tp1, dtype=np.int32)

    # Root author is active at step 0
    root_idx = index.user_to_idx[case.root.user_id]
    active_mask[0, root_idx] = True

    # Bin interactions into steps
    for ix in case.interactions:
        step = grid.step_of(ix.timestamp)
        if step > T:
            continue  # beyond grid
        user_idx = index.user_to_idx[ix.user_id]
        active_mask[step, user_idx] = True

        if ix.kind == "comment":
            comment_count[step] += 1
        elif ix.kind == "repost":
            repost_count[step] += 1
        # Both comment and repost also count as interactions
        interaction_count[step] += 1

    # active_count: users with ≥1 interaction in each step
    active_count = active_mask.sum(axis=1).astype(np.int32)

    # cumulative_users: monotonic cumulative unique active users
    any_active = active_mask.any(axis=0)
    cum = np.zeros(Tp1, dtype=np.int32)
    seen = np.zeros(N, dtype=bool)
    running = 0
    for t in range(Tp1):
        newly_active = active_mask[t] & ~seen
        running += int(newly_active.sum())
        seen = seen | active_mask[t]
        cum[t] = running
    cumulative_users = cum

    # V1.6.1: first_actor and repeat_actor decomposition
    first_actor = np.zeros(Tp1, dtype=np.int32)
    repeat_actor = np.zeros(Tp1, dtype=np.int32)
    ever_seen = np.zeros(N, dtype=bool)
    for t in range(Tp1):
        step_new = active_mask[t] & ~ever_seen
        step_repeat = active_mask[t] & ever_seen
        first_actor[t] = int(step_new.sum())
        repeat_actor[t] = int(step_repeat.sum())
        ever_seen = ever_seen | active_mask[t]
    # Invariant: active_count[t] == first_actor[t] + repeat_actor[t]

    # Stance and arousal: NaN because no precomputed signals
    stance_mean = np.full(Tp1, np.nan, dtype=np.float64)
    arousal_mean = np.full(Tp1, np.nan, dtype=np.float64)

    # Tail steps (beyond last_data_step) are NOT real observations.
    # Mark them as unobserved so they don't contribute to fitting or
    # validation loss — only real data steps can participate in loss.
    is_data_step = np.zeros(Tp1, dtype=bool)
    is_data_step[:grid.last_data_step + 1] = True

    observation_masks: dict[str, np.ndarray] = {
        "active_count": is_data_step.copy(),
        "comment_count": is_data_step.copy(),
        "repost_count": is_data_step.copy(),
        "interaction_count": is_data_step.copy(),
        "cumulative_users": is_data_step.copy(),
        "first_actor_count": is_data_step.copy(),
        "repeat_actor_count": is_data_step.copy(),
        "stance": np.zeros(Tp1, dtype=bool),
        "arousal": np.zeros(Tp1, dtype=bool),
    }

    return ObservedTrajectory(
        steps=steps,
        active_count=active_count,
        cumulative_users=cumulative_users,
        comment_count=comment_count,
        repost_count=repost_count,
        interaction_count=interaction_count,
        active_mask=active_mask,
        stance_mean=stance_mean,
        arousal_mean=arousal_mean,
        observation_masks=observation_masks,
        first_actor_count=first_actor,
        repeat_actor_count=repeat_actor,
    )
