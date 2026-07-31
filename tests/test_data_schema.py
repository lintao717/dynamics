"""Tests for canonical event data schema."""

from datetime import datetime, timezone
import pytest
from dynamics_simulation.data.schema import (
    EventCase, RootPost, InteractionRecord, InteractionKind,
)


def test_event_case_requires_root_author_and_monotonic_timestamps():
    """Valid case with root and one late interaction should pass validation."""
    root = RootPost(
        post_id="root-1",
        user_id="user-root",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text="root",
        label="fake",
        expert_analysis=None,
    )
    late = InteractionRecord(
        interaction_id="c1",
        root_post_id="root-1",
        user_id="u1",
        timestamp=datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
        kind="comment",
        text="comment",
    )
    case = EventCase(
        case_id="root-1",
        source_dataset="CHECKED",
        root=root,
        interactions=(late,),
        metadata={},
    )
    case.validate()


def test_event_case_rejects_interaction_before_root():
    """Interaction before root timestamp must raise ValueError."""
    root = RootPost(
        post_id="root-1",
        user_id="user-root",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text="root",
        label="fake",
        expert_analysis=None,
    )
    early = InteractionRecord(
        interaction_id="c0",
        root_post_id="root-1",
        user_id="u0",
        timestamp=datetime(2019, 12, 31, 23, tzinfo=timezone.utc),
        kind="comment",
        text="early",
    )
    case = EventCase(
        case_id="root-1",
        source_dataset="CHECKED",
        root=root,
        interactions=(early,),
        metadata={},
    )
    with pytest.raises(ValueError, match="before root"):
        case.validate()


def test_event_case_rejects_empty_case_id():
    root = RootPost(
        post_id="root-1", user_id="u0",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text="t", label="fake", expert_analysis=None,
    )
    case = EventCase(
        case_id="", source_dataset="CHECKED",
        root=root, interactions=(), metadata={},
    )
    with pytest.raises(ValueError):
        case.validate()


def test_event_case_rejects_naive_datetime():
    root = RootPost(
        post_id="root-1", user_id="u0",
        timestamp=datetime(2020, 1, 1),  # naive!
        text="t", label="fake", expert_analysis=None,
    )
    case = EventCase(
        case_id="root-1", source_dataset="CHECKED",
        root=root, interactions=(), metadata={},
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        case.validate()


def test_event_case_rejects_root_id_mismatch():
    root = RootPost(
        post_id="root-1", user_id="u0",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text="t", label="fake", expert_analysis=None,
    )
    case = EventCase(
        case_id="different-id", source_dataset="CHECKED",
        root=root, interactions=(), metadata={},
    )
    with pytest.raises(ValueError, match="case_id must equal root.post_id"):
        case.validate()


def test_event_case_rejects_interaction_wrong_root():
    root = RootPost(
        post_id="root-1", user_id="u0",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text="t", label="fake", expert_analysis=None,
    )
    wrong = InteractionRecord(
        interaction_id="i1", root_post_id="other-root",
        user_id="u1",
        timestamp=datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
        kind="comment", text="c",
    )
    case = EventCase(
        case_id="root-1", source_dataset="CHECKED",
        root=root, interactions=(wrong,), metadata={},
    )
    with pytest.raises(ValueError, match="root_post_id does not match"):
        case.validate()


def test_user_ids_preserves_deterministic_order():
    """Root author first, then interaction users in (timestamp, id) order."""
    root = RootPost(
        post_id="ev1", user_id="root-user",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text="root", label="real", expert_analysis=None,
    )
    i1 = InteractionRecord(
        interaction_id="c2", root_post_id="ev1", user_id="u2",
        timestamp=datetime(2020, 1, 1, 2, tzinfo=timezone.utc),
        kind="comment", text="c2",
    )
    i2 = InteractionRecord(
        interaction_id="c1", root_post_id="ev1", user_id="u1",
        timestamp=datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
        kind="repost", text="c1",
    )
    # u1 appears again later
    i3 = InteractionRecord(
        interaction_id="c3", root_post_id="ev1", user_id="u1",
        timestamp=datetime(2020, 1, 1, 3, tzinfo=timezone.utc),
        kind="comment", text="c3",
    )
    case = EventCase(
        case_id="ev1", source_dataset="CHECKED",
        root=root, interactions=(i1, i2, i3), metadata={},
    )
    # Expected: root-user first, then u1 (earliest timestamp), then u2
    assert case.user_ids == ("root-user", "u1", "u2")


def test_immutability():
    """EventCase, RootPost, InteractionRecord should be frozen."""
    root = RootPost(
        post_id="r1", user_id="u0",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text="t", label="fake", expert_analysis=None,
    )
    with pytest.raises(Exception):
        root.post_id = "other"  # type: ignore[misc]


def test_interaction_kind_literal():
    """InteractionKind must be a valid literal type."""
    from typing import get_args
    kinds = set(get_args(InteractionKind))
    assert kinds == {"comment", "repost", "interaction"}
