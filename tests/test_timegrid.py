"""Tests for TimeGrid — timestamp to integer step mapping."""

from datetime import datetime, timezone, timedelta
import pytest
from dynamics_simulation.data.timegrid import TimeGrid


def test_step_zero_is_root_time():
    """step 0 must correspond to the exact root timestamp."""
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, last_data_step=10, final_step=10)
    assert grid.step_of(root_ts) == 0


def test_interaction_after_start_is_step_one():
    """Interaction 30 min after root must be step 1 (not step 0)."""
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, last_data_step=10, final_step=10)
    ts = root_ts + timedelta(minutes=30)
    assert grid.step_of(ts) == 1


def test_interaction_exactly_at_boundary():
    """Interaction exactly at start + Δt belongs to step 1."""
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, last_data_step=10, final_step=10)
    ts = root_ts + timedelta(hours=1)
    assert grid.step_of(ts) == 1


def test_interaction_before_root_raises():
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, last_data_step=10, final_step=10)
    early = root_ts - timedelta(seconds=1)
    with pytest.raises(ValueError, match="before start"):
        grid.step_of(early)


def test_step_of_late_interaction():
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, last_data_step=10, final_step=10)
    ts = root_ts + timedelta(hours=5, minutes=30)
    # ceil(5.5/1.0) = 6: (5.0, 6.0] → step 6
    assert grid.step_of(ts) == 6


def test_final_step():
    """final_step is the last simulation step index, NOT a count."""
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=1.0, last_data_step=10, final_step=10)
    assert grid.final_step == 10
    # n_trajectory_points = final_step + 1
    assert grid.n_trajectory_points == 11


def test_end_time():
    root_ts = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    grid = TimeGrid(start_time=root_ts, step_hours=2.0, last_data_step=5, final_step=5)
    # end_time = start + step_hours * final_step
    assert grid.end_time == root_ts + timedelta(hours=10)


def test_from_case_no_interactions():
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
    # No interactions → last_step = 0, final_step = 0 + 3 = 3
    # (previously was 1 + 3 = 4, which gave an extra simulation step)
    assert grid.final_step == 3
    # Trajectory: step 0 (root), step 1, 2, 3 (tail) = 4 points
    assert grid.n_trajectory_points == 4


def test_from_case_with_interactions():
    """final_step must cover the latest interaction plus tail, no extra +1."""
    from dynamics_simulation.data.schema import (
        EventCase, RootPost, InteractionRecord,
    )
    root = RootPost(
        post_id="ev1", user_id="u0",
        timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
        text="t", label="fake", expert_analysis=None,
    )
    # Last interaction at t+5h → step_of = ceil(5/1) = 5
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
    # last_step = 5, final_step = 5 + 2 = 7
    assert grid.final_step == 7
    # n_trajectory_points = 8 (steps 0-7)
    assert grid.n_trajectory_points == 8


def test_n_data_steps():
    """n_data_steps conservatively returns n_trajectory_points
    since TimeGrid does not internally track tail_steps."""
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
    # last_data_step=0 (root only), final_step=3, n_trajectory_points=4
    assert grid.final_step == 3
    assert grid.n_trajectory_points == 4
    assert grid.n_data_steps == 1  # last_data_step + 1 = 1 (real data only)


def test_last_data_step_preserved():
    """last_data_step must be stored and accessible."""
    from dynamics_simulation.data.schema import EventCase, RootPost, InteractionRecord
    root = RootPost(
        post_id="ev1", user_id="u0",
        timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
        text="t", label="fake", expert_analysis=None,
    )
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
    assert grid.last_data_step == 5  # ceil(5h/1h) = 5
    assert grid.final_step == 7      # 5 + 2 = 7
    assert grid.n_data_steps == 6    # last_data_step + 1 = 6


def test_timegrid_maps_simulation_config_correctly():
    """SimulationConfig(T=grid.final_step) must produce
    grid.final_step updates = grid.n_trajectory_points trajectory points."""
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
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    # last_step=0, final_step=1, n_trajectory_points=2
    # SimulationConfig(T=final_step=1) → 1 update, 2 trajectory points
    assert grid.final_step == 1
    assert grid.n_trajectory_points == 2
