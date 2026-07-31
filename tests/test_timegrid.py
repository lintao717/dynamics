"""Tests for TimeGrid — timestamp to integer step mapping."""

from datetime import datetime, timezone, timedelta
import pytest
from dynamics_simulation.data.timegrid import TimeGrid


def test_step_zero_is_root_time():
    """step 0 must correspond to the exact root timestamp."""
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, total_steps=10)
    assert grid.step_of(root_ts) == 0


def test_interaction_after_start_is_step_one():
    """Interaction 30 min after root must be step 1 (not step 0)."""
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, total_steps=10)
    ts = root_ts + timedelta(minutes=30)
    assert grid.step_of(ts) == 1


def test_interaction_exactly_at_boundary():
    """Interaction exactly at start + Δt belongs to step 1."""
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, total_steps=10)
    ts = root_ts + timedelta(hours=1)
    assert grid.step_of(ts) == 1


def test_interaction_before_root_raises():
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, total_steps=10)
    early = root_ts - timedelta(seconds=1)
    with pytest.raises(ValueError, match="before start"):
        grid.step_of(early)


def test_step_of_late_interaction():
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, total_steps=10)
    ts = root_ts + timedelta(hours=5, minutes=30)
    # ceil(5.5/1.0) = 6: (5.0, 6.0] → step 6
    assert grid.step_of(ts) == 6


def test_total_steps():
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, total_steps=10)
    assert grid.total_steps == 10


def test_end_time():
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=2.0, total_steps=5)
    assert grid.end_time == root_ts + timedelta(hours=10)


def test_from_case():
    """TimeGrid.from_case must use root timestamp as start."""
    from dynamics_simulation.data.schema import EventCase, RootPost

    root = RootPost(
        post_id="ev1", user_id="u0",
        timestamp=datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc),
        text="t", label="fake", expert_analysis=None,
    )
    case = EventCase(
        case_id="ev1", source_dataset="CHECKED",
        root=root, interactions=(),
    )
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=3)
    assert grid.start_time == root.timestamp
    # No interactions → total_steps = 1 (step 0) + tail_steps
    assert grid.total_steps == 1 + 3


def test_from_case_with_interactions():
    """Total steps must cover the latest interaction plus tail."""
    from dynamics_simulation.data.schema import (
        EventCase, RootPost, InteractionRecord,
    )
    root = RootPost(
        post_id="ev1", user_id="u0",
        timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
        text="t", label="fake", expert_analysis=None,
    )
    # Last interaction at t+5h → step 5
    late = InteractionRecord(
        interaction_id="i1", root_post_id="ev1", user_id="u1",
        timestamp=datetime(2020, 1, 1, 13, 0, tzinfo=timezone.utc),
        kind="comment", text="late",
    )
    case = EventCase(
        case_id="ev1", source_dataset="CHECKED",
        root=root, interactions=(late,),
    )
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=2)
    # step_of(late) = 5, so 6 data steps (0-5) + 2 tail = 8
    assert grid.total_steps == 8
