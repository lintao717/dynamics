"""Time grid: maps timestamps to integer simulation steps.

Step 0 = root post time. Interaction at t0 < t < t0+Δt → step 1.
Interaction exactly at t0+Δt → step 1.

``final_step`` is the LAST simulation step index (not a count).
``n_trajectory_points = final_step + 1`` is the number of trajectory
points (including step 0).

``SimulationConfig(T=grid.final_step)`` runs ``final_step`` state
updates, producing ``final_step + 1`` trajectory points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from dynamics_simulation.data.schema import EventCase


@dataclass(frozen=True)
class TimeGrid:
    """Maps wall-clock timestamps to integer simulation steps.

    Step 0 is the root post time. Subsequent steps are contiguous
    windows of length *step_hours*.
    """

    start_time: datetime
    step_hours: float
    final_step: int
    """Last simulation step index. SimulationConfig(T=final_step)
    runs final_step updates, producing final_step + 1 trajectory points."""

    @classmethod
    def from_case(
        cls,
        case: EventCase,
        step_hours: float = 1.0,
        tail_steps: int = 1,
    ) -> "TimeGrid":
        """Build a TimeGrid covering the full event duration plus tail.

        Args:
            case: Validated EventCase.
            step_hours: Duration of each simulation step in hours.
            tail_steps: Extra steps appended after the last data step.
        """
        case.validate()
        start = case.root.timestamp

        if case.interactions:
            last_ts = case.interactions[-1].timestamp
        else:
            last_ts = start

        delta = last_ts - start
        hours = delta.total_seconds() / 3600.0
        # Step of last data interaction: ceil(hours / step_hours)
        #   hours=0   → step 0 (root only)
        #   0 < hours ≤ Δt → step 1
        last_data_step = math.ceil(hours / step_hours)

        # final_step = last_data_step + tail_steps
        # (No spurious +1 — final_step is an index, not a count.)
        final = last_data_step + int(tail_steps)

        return cls(start_time=start, step_hours=float(step_hours),
                   final_step=int(final))

    def step_of(self, timestamp: datetime) -> int:
        """Return the integer step index for *timestamp*.

        Raises:
            ValueError: If timestamp is before start_time.
        """
        if timestamp < self.start_time:
            raise ValueError(
                f"Timestamp {timestamp.isoformat()} is before "
                f"start {self.start_time.isoformat()}"
            )
        delta = timestamp - self.start_time
        hours = delta.total_seconds() / 3600.0
        # ceil: t=t0 → 0; t0 < t ≤ t0+Δt → 1; t0+Δt < t ≤ t0+2Δt → 2
        return math.ceil(hours / self.step_hours)

    @property
    def end_time(self) -> datetime:
        """End of the last simulation step."""
        return self.start_time + timedelta(
            hours=self.step_hours * self.final_step
        )

    @property
    def n_trajectory_points(self) -> int:
        """Number of trajectory points = final_step + 1.

        This is the length of observation and simulation arrays
        (steps 0 through final_step inclusive).
        """
        return self.final_step + 1

    @property
    def n_data_steps(self) -> int:
        """Number of steps that contain real-world observations.

        This is final_step - tail_steps + 1 (since step 0 is the root
        post, which is real data). Conservatively returns
        n_trajectory_points — the tail steps are a config-time
        parameter not stored in the grid itself.

        For precise computation use TimeGrid.from_case() tail_steps.
        """
        return self.n_trajectory_points
