"""[V1.7] Forecast slicing: EventHistory and ForecastTarget.

Splits a full EventCase at a cutoff time into:
  - EventHistory: only data at or before the cutoff (visible to model)
  - ForecastTarget: future data after the cutoff (hidden, evaluation only)

Enforces:
  - No future interaction or user identity leaks into history.
  - Cutoff must leave at least one future window for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from dynamics_simulation.data.schema import EventCase, InteractionRecord


@dataclass(frozen=True)
class EventHistory:
    """Data visible at forecast cutoff (step <= cutoff_step).

    Contains only root post, interactions up to the cutoff, and
    users who have appeared by then. Does NOT contain future users.
    """

    case_id: str
    source_dataset: str
    cutoff_step: int
    cutoff_time: datetime
    root: object  # RootPost (copied from original)
    interactions: tuple[InteractionRecord, ...]  # only <= cutoff
    observed_user_ids: tuple[str, ...]  # users seen by cutoff
    n_observed_users: int
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_case(cls, case: EventCase, cutoff_step: int,
                  step_hours: float) -> "EventHistory":
        """Slice an EventCase at a given step index.

        Args:
            case: Full EventCase (all interactions known).
            cutoff_step: Last step visible to the model (0-indexed).
            step_hours: Hours per step (used to compute cutoff time).

        Returns:
            EventHistory with only pre-cutoff data.

        Raises:
            ValueError: If cutoff_step < 0 or no future data remains.
        """
        case.validate()
        if cutoff_step < 0:
            raise ValueError(f"cutoff_step must be >= 0, got {cutoff_step}")

        from datetime import timedelta
        cutoff_time = case.root.timestamp + timedelta(
            hours=step_hours * (cutoff_step + 1))

        # Filter interactions: only those at or before cutoff
        from dynamics_simulation.data.timegrid import TimeGrid
        grid = TimeGrid.from_case(case, step_hours=step_hours, tail_steps=0)
        future_interactions = tuple(
            ix for ix in case.interactions
            if grid.step_of(ix.timestamp) > cutoff_step
        )
        if len(future_interactions) == 0:
            raise ValueError(
                f"No future interactions remain after cutoff_step={cutoff_step}"
            )

        history_interactions = tuple(
            ix for ix in case.interactions
            if grid.step_of(ix.timestamp) <= cutoff_step
        )

        # Users observed by cutoff
        observed = {case.root.user_id}
        for ix in history_interactions:
            observed.add(ix.user_id)

        return cls(
            case_id=case.case_id,
            source_dataset=case.source_dataset,
            cutoff_step=cutoff_step,
            cutoff_time=cutoff_time,
            root=case.root,
            interactions=history_interactions,
            observed_user_ids=tuple(sorted(observed)),
            n_observed_users=len(observed),
            metadata={
                "full_interaction_count": len(case.interactions),
                "history_interaction_count": len(history_interactions),
                "total_users": len(case.user_ids),
                "step_hours": step_hours,
            },
        )


@dataclass(frozen=True)
class ForecastTarget:
    """Future observations after the cutoff (hidden from model).

    Contains the observed active counts, first-actor counts, and
    repeat-actor counts for windows after cutoff_step.
    """

    case_id: str
    cutoff_step: int
    horizons: tuple[int, ...]  # steps after cutoff (e.g., 1, 2, 3)
    active_count: np.ndarray   # shape (len(horizons),)
    first_actor_count: np.ndarray
    repeat_actor_count: np.ndarray
    cumulative_users: np.ndarray
    n_future_users: int  # users first seen after cutoff

    @classmethod
    def from_case(cls, case: EventCase, cutoff_step: int,
                  horizons: tuple[int, ...],
                  step_hours: float) -> "ForecastTarget":
        """Extract future observations from a full EventCase.

        Args:
            case: Full EventCase.
            cutoff_step: Last step visible to model.
            horizons: Future steps to predict (1 = next step, etc.).
            step_hours: Hours per step.

        Returns:
            ForecastTarget with future observations.

        Raises:
            ValueError: If any horizon extends beyond available data.
        """
        case.validate()
        from dynamics_simulation.data.timegrid import TimeGrid
        from dynamics_simulation.data.observations import build_observed_trajectory
        from dynamics_simulation.data.indexing import NodeIndex

        grid = TimeGrid.from_case(case, step_hours=step_hours, tail_steps=0)
        max_step = grid.last_data_step
        if cutoff_step + max(horizons) > max_step:
            raise ValueError(
                f"Horizon {max(horizons)} exceeds available data "
                f"(cutoff={cutoff_step}, max_step={max_step})"
            )

        index = NodeIndex.from_case(case)
        traj = build_observed_trajectory(case, index, grid)

        # Extract future windows
        future_indices = [cutoff_step + h for h in horizons]
        active = np.array([traj.active_count[i] for i in future_indices],
                          dtype=np.int32)
        first = np.array([traj.first_actor_count[i] for i in future_indices],
                         dtype=np.int32)
        repeat = np.array([traj.repeat_actor_count[i] for i in future_indices],
                          dtype=np.int32)
        cum = np.array([traj.cumulative_users[i] for i in future_indices],
                       dtype=np.int32)

        # Users first seen after cutoff
        pre_cutoff_users = set()
        seen = {case.root.user_id}
        for ix in case.interactions:
            if grid.step_of(ix.timestamp) <= cutoff_step:
                pre_cutoff_users.add(ix.user_id)
        all_users = set(case.user_ids)
        n_future = len(all_users - pre_cutoff_users)

        return cls(
            case_id=case.case_id,
            cutoff_step=cutoff_step,
            horizons=horizons,
            active_count=active,
            first_actor_count=first,
            repeat_actor_count=repeat,
            cumulative_users=cum,
            n_future_users=n_future,
        )
