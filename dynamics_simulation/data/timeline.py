"""
Event input timeline: converts EventCase → per-step ExternalInputs.

Broadcast exposure follows exponential decay: exposure(step) = A * 0.5^(step/h)
Root author receives zero exposure. Non-root users receive equal exposure.

IMPORTANT: All time-dependent signals use FIXED time constants, NOT the event's
total duration. This guarantees no future-data dependency — step t input depends
only on t and the preset BroadcastExposureConfig, never on when the event ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics_simulation.data.schema import EventCase
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.transitions import ExternalInputs


@dataclass(frozen=True)
class BroadcastExposureConfig:
    """Parameters for the broadcast media exposure signal.

    All time constants are in simulation steps and are independent of
    event duration — step-t input depends only on step index and these
    preset constants, never on when the event ends.

    exposure(step) = amplitude * 0.5^(step / exposure_half_life_steps)
    novelty(step)  = novelty_at_root * 0.5^(step / novelty_half_life_steps)
    staleness(step) = 1 - exp(-step / staleness_tau_steps)
    """

    amplitude: float = 1.0
    exposure_half_life_steps: float = 4.0
    novelty_at_root: float = 1.0
    novelty_half_life_steps: float = 6.0
    staleness_tau_steps: float = 12.0

    # ── Root-post shock (V1.3) ──
    root_shock: bool = False
    """Apply an initial fast-decaying exposure pulse at macro-step 0."""
    shock_amplitude: float = 1.0
    """Peak shock intensity (normalised [0,1], scaled by amplitude)."""
    shock_half_life_micro: float = 2.0
    """Micro-steps until shock halves."""

    def __post_init__(self):
        if not (0.0 <= self.amplitude <= 1.0):
            raise ValueError(
                f"amplitude={self.amplitude} must be in [0, 1]"
            )
        if not (0.0 <= self.novelty_at_root <= 1.0):
            raise ValueError(
                f"novelty_at_root={self.novelty_at_root} must be in [0, 1]"
            )
        if not (0.0 <= self.shock_amplitude <= 1.0):
            raise ValueError(
                f"shock_amplitude={self.shock_amplitude} must be in [0, 1]"
            )
        if self.exposure_half_life_steps <= 0:
            raise ValueError(
                f"exposure_half_life_steps={self.exposure_half_life_steps} must be > 0"
            )
        if self.novelty_half_life_steps <= 0:
            raise ValueError(
                f"novelty_half_life_steps={self.novelty_half_life_steps} must be > 0"
            )
        if self.staleness_tau_steps <= 0:
            raise ValueError(
                f"staleness_tau_steps={self.staleness_tau_steps} must be > 0"
            )
        if self.shock_half_life_micro <= 0:
            raise ValueError(
                f"shock_half_life_micro={self.shock_half_life_micro} must be > 0"
            )

    def shock_at(self, micro_step: int) -> float:
        """Root-shock intensity at micro-step index within macro-step 0."""
        if not self.root_shock:
            return 0.0
        return self.amplitude * self.shock_amplitude * (
            0.5 ** (micro_step / self.shock_half_life_micro)
        )

    def exposure_at(self, step: int) -> float:
        """Compute exposure intensity at *step*."""
        return self.amplitude * (0.5 ** (step / self.exposure_half_life_steps))

    def novelty_at(self, step: int) -> float:
        """Compute novelty at *step* using fixed half-life."""
        return float(np.clip(
            self.novelty_at_root * (0.5 ** (step / self.novelty_half_life_steps)),
            0.0, 1.0,
        ))

    def staleness_at(self, step: int) -> float:
        """Compute staleness at *step* using fixed tau (no total_steps)."""
        return float(np.clip(
            1.0 - np.exp(-step / self.staleness_tau_steps),
            0.0, 1.0,
        ))


class EventInputTimeline:
    """Produces ExternalInputs for each simulation step.

    Deterministic; depends only on case, index, grid, and config.
    No future-data dependency: staleness and novelty use FIXED time
    constants, not the event's total duration.
    """

    def __init__(
        self,
        case: EventCase,
        index: NodeIndex,
        grid: TimeGrid,
        broadcast_cfg: BroadcastExposureConfig | None = None,
    ):
        case.validate()
        self._case = case
        self._index = index
        self._grid = grid
        self._bcast = broadcast_cfg or BroadcastExposureConfig()
        self._root_idx = index.user_to_idx[case.root.user_id]

    def inputs_at(self, n: int, step: int,
                  micro_step: int = 0, micro_total: int = 1) -> ExternalInputs:
        """Build ExternalInputs for *step* (macro) and *micro_step* (sub-step).

        All time-dependent signals use FIXED time constants — no
        future-data dependency on when the event ends.

        Root shock (if enabled) applies on macro-step 0 only,
        decaying across micro-steps within that day.

        Args:
            n: Number of agents.
            step: Current macro simulation step (0-indexed).
            micro_step: Sub-step index within the macro-step (0-indexed).
            micro_total: Total sub-steps per macro-step.

        Returns:
            ExternalInputs with media_exposure, staleness, and novelty.
        """
        # Broadcast media exposure: root gets 0, all others get decay
        media = np.zeros(n, dtype=np.float64)
        exposure_val = self._bcast.exposure_at(step)

        # Root shock: bounded combination with base exposure
        # M_combined = 1 - (1-M_base)(1-M_shock), then clamped to [0,1]
        shock_val = self._bcast.shock_at(micro_step) if step == 0 else 0.0

        for i in range(n):
            if i != self._root_idx:
                combined = 1.0 - (1.0 - exposure_val) * (1.0 - shock_val)
                media[i] = float(np.clip(combined, 0.0, 1.0))

        # Global macro-step index for staleness/novelty
        global_step = step * micro_total + micro_step
        global_total = self._grid.final_step * micro_total if self._grid else micro_total

        # Staleness: saturating exponential with fixed tau
        staleness = self._bcast.staleness_at(step)

        # Novelty: exponential decay with fixed half-life
        novelty = self._bcast.novelty_at(step)

        return ExternalInputs(
            media_exposure=media,
            staleness=staleness,
            novelty=novelty,
            shock=0.0,
            V=0.0,
        )
