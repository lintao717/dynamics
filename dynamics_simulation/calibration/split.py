"""Chronological train/validation split for time-series replay."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological split: steps 0..train_end_step for fitting,
    steps train_end_step+1..total_steps for validation.
    """

    train_end_step: int
    total_steps: int

    @classmethod
    def by_fraction(
        cls,
        total_steps: int,
        train_fraction: float = 0.7,
    ) -> "TemporalSplit":
        """Create a split by fraction of total steps.

        At least 3 training steps and 1 validation step are required.
        """
        if total_steps < 4:
            raise ValueError(
                f"total_steps must be at least 4, got {total_steps}"
            )
        if not (0.0 < train_fraction < 1.0):
            raise ValueError(
                f"train_fraction must be in (0, 1), got {train_fraction}"
            )

        train_end = int(total_steps * train_fraction)
        train_end = min(max(train_end, 3), total_steps - 1)
        return cls(train_end_step=train_end, total_steps=total_steps)

    @property
    def train_steps(self) -> int:
        return self.train_end_step + 1

    @property
    def val_steps(self) -> int:
        return self.total_steps - self.train_end_step
