"""Node index: hashed user ID ↔ contiguous integer mapping.

Index 0 is always the root author. Deterministic given the same EventCase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from dynamics_simulation.data.schema import EventCase


@dataclass(frozen=True)
class NodeIndex:
    """Bidirectional mapping between string user IDs and contiguous int indices.

    Immutable. Index 0 is always the root author.
    """

    user_to_idx: Mapping[str, int]
    idx_to_user: tuple[str, ...]

    @classmethod
    def from_case(cls, case: EventCase) -> "NodeIndex":
        """Build a NodeIndex from a validated EventCase.

        The ordering is deterministic: root author first, then
        interaction users in (timestamp, interaction_id) order,
        deduplicated.
        """
        case.validate()
        idx_to_user = case.user_ids
        user_to_idx = {
            user_id: idx for idx, user_id in enumerate(idx_to_user)
        }
        return cls(user_to_idx=user_to_idx, idx_to_user=idx_to_user)

    def __len__(self) -> int:
        return len(self.idx_to_user)

    @property
    def n(self) -> int:
        return len(self.idx_to_user)
