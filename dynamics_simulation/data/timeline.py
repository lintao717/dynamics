"""
Event input timeline: converts EventCase → per-step ExternalInputs.

Broadcast exposure follows exponential decay: exposure(step) = A * 0.5^(step/h)
Root author receives zero exposure. Non-root users receive equal exposure.
No future-data dependency: exposure depends only on step index and root time.
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

    exposure(step) = amplitude * 0.5^(step / half_life_steps)
    """

    amplitude: float = 1.0
    half_life_steps: float = 4.0
    novelty_at_root: float = 1.0

    def exposure_at(self, step: int) -> float:
        """Compute exposure intensity at *step*."""
        return self.amplitude * (0.5 ** (step / self.half_life_steps))


class EventInputTimeline:
    """Produces ExternalInputs for each simulation step.

    Deterministic; depends only on case, index, grid, and config.
    No future-data dependency: interactions do not affect exposure.
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

    def inputs_at(self, n: int, step: int, total_steps: int) -> ExternalInputs:
        """Build ExternalInputs for *step*.

        Args:
            n: Number of agents.
            step: Current simulation step (0-indexed).
            total_steps: Total simulation steps.

        Returns:
            ExternalInputs with media_exposure and staleness set.
        """
        # Broadcast media exposure: root gets 0, all others get decay
        media = np.zeros(n, dtype=np.float64)
        exposure_val = self._bcast.exposure_at(step)
        for i in range(n):
            if i != self._root_idx:
                media[i] = exposure_val

        # Staleness increases linearly with time
        staleness = float(np.clip(step / max(total_steps, 1), 0.0, 1.0))

        # Novelty: high at root, decays
        novelty = float(np.clip(
            self._bcast.novelty_at_root * (0.5 ** (step / max(total_steps, 1))),
            0.0, 1.0,
        ))

        return ExternalInputs(
            media_exposure=media,
            staleness=staleness,
            novelty=novelty,
            shock=0.0,
            V=0.0,
        )
