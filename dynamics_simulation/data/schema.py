"""
Canonical immutable event data records.

Defines the universal data contract for all dataset adapters.
Every adapter (CHECKED, CED, future sources) must produce EventCase instances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Mapping, Any

InteractionKind = Literal["comment", "repost", "interaction"]


@dataclass(frozen=True)
class RootPost:
    """Root microblog/post that initiates a cascade.

    All timestamps must be timezone-aware (UTC preferred).
    """

    post_id: str
    user_id: str
    timestamp: datetime
    text: str
    label: str | None
    expert_analysis: str | None


@dataclass(frozen=True)
class InteractionRecord:
    """A single comment, repost, or generic interaction linked to a root post."""

    interaction_id: str
    root_post_id: str
    user_id: str
    timestamp: datetime
    kind: InteractionKind
    text: str


@dataclass(frozen=True)
class EventCase:
    """Complete microblog cascade: root post + all observed interactions.

    Immutable once constructed. Call ``validate()`` to enforce invariants.
    """

    case_id: str
    source_dataset: str
    root: RootPost
    interactions: tuple[InteractionRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate all invariants. Raises ValueError on first violation."""

        def require_id(value: str, field_name: str) -> None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        def require_aware(value: datetime, field_name: str) -> None:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name} must be timezone-aware")

        require_id(self.case_id, "case_id")
        require_id(self.source_dataset, "source_dataset")
        require_id(self.root.post_id, "root.post_id")
        require_id(self.root.user_id, "root.user_id")
        require_aware(self.root.timestamp, "root.timestamp")
        if self.case_id != self.root.post_id:
            raise ValueError("case_id must equal root.post_id")

        for key, value in self.metadata.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metadata[{key!r}] must be finite")

        for interaction in self.interactions:
            require_id(interaction.interaction_id, "interaction_id")
            require_id(interaction.root_post_id, "interaction.root_post_id")
            require_id(interaction.user_id, "interaction.user_id")
            require_aware(interaction.timestamp, "interaction.timestamp")
            if interaction.root_post_id != self.root.post_id:
                raise ValueError(
                    "interaction root_post_id does not match root"
                )
            if interaction.timestamp < self.root.timestamp:
                raise ValueError(
                    "interaction timestamp is before root timestamp"
                )

    @property
    def user_ids(self) -> tuple[str, ...]:
        """Return user IDs in deterministic order.

        Root author is always first. Remaining users appear in
        (timestamp, interaction_id) order of their first interaction,
        deduplicated.
        """
        ordered = [self.root.user_id]
        seen = {self.root.user_id}
        for item in sorted(
            self.interactions,
            key=lambda x: (x.timestamp, x.interaction_id),
        ):
            if item.user_id not in seen:
                ordered.append(item.user_id)
                seen.add(item.user_id)
        return tuple(ordered)
