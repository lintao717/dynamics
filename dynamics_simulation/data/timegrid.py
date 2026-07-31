"""Time grid: maps timestamps to integer simulation steps.

Step 0 = root post time. Interaction at t0 < t < t0+Δt → step 1.
Interaction exactly at t0+Δt → step 1.
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
    total_steps: int

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
            tail_steps: Extra steps appended after the last interaction.
        """
        case.validate()
        start = case.root.timestamp

        if case.interactions:
            last_ts = case.interactions[-1].timestamp
        else:
            last_ts = start

        delta = last_ts - start
        hours = delta.total_seconds() / 3600.0
        # Step of last interaction: ceil(hours / step_hours)
        #   hours=0   → step 0 (root only)
        #   0 < hours ≤ Δt → step 1
        last_step = math.ceil(hours / step_hours)
        # Total steps: cover step 0 through last_step, plus tail
        total = last_step + 1 + int(tail_steps)

        return cls(start_time=start, step_hours=float(step_hours),
                   total_steps=int(total))

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
        return self.start_time + timedelta(
            hours=self.step_hours * self.total_steps
        )

    @property
    def n_data_steps(self) -> int:
        """Number of steps that contain observations (total - tail)."""
        return self.total_steps
