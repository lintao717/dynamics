"""Tests for ObservedTrajectory — masked time-series observations."""

from datetime import datetime, timezone
import numpy as np
import pytest
from dynamics_simulation.data.schema import (
    EventCase, RootPost, InteractionRecord,
)
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.observations import (
    ObservedTrajectory,
    build_observed_trajectory,
)


def _make_case():
    """Minimal 3-user case: root + two comments."""
    root = RootPost(
        post_id="ev1", user_id="root",
        timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
        text="root", label="fake", expert_analysis=None,
    )
    c1 = InteractionRecord(
        interaction_id="c1", root_post_id="ev1", user_id="u1",
        timestamp=datetime(2020, 1, 1, 8, 30, tzinfo=timezone.utc),
        kind="comment", text="c1",
    )
    c2 = InteractionRecord(
        interaction_id="c2", root_post_id="ev1", user_id="u2",
        timestamp=datetime(2020, 1, 1, 9, 30, tzinfo=timezone.utc),
        kind="repost", text="c2",
    )
    return EventCase(
        case_id="ev1", source_dataset="CHECKED",
        root=root, interactions=(c1, c2),
    )


def test_root_active_at_step_zero():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    traj = build_observed_trajectory(case, index, grid)

    root_idx = index.user_to_idx["root"]
    assert traj.active_mask[0, root_idx] == True
    assert traj.active_count[0] == 1


def test_user_active_only_in_interaction_step():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    traj = build_observed_trajectory(case, index, grid)

    u1_idx = index.user_to_idx["u1"]
    # step 0 (t=8:00): root only, u1 not yet active
    assert traj.active_mask[0, u1_idx] == False
    # step 1 (t=8:00-9:00): u1 commented at 8:30
    assert traj.active_mask[1, u1_idx] == True


def test_cumulative_users_monotonic():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    traj = build_observed_trajectory(case, index, grid)

    diff = np.diff(traj.cumulative_users)
    assert np.all(diff >= 0), f"cumulative_users not monotonic: {traj.cumulative_users}"


def test_comment_and_repost_counts():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    traj = build_observed_trajectory(case, index, grid)

    # step 1 has 1 comment (u1)
    assert traj.comment_count[1] == 1
    assert traj.repost_count[1] == 0
    # step 2 has 1 repost (u2)
    assert traj.comment_count[2] == 0
    assert traj.repost_count[2] == 1


def test_interaction_count_aggregates_all_types():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    traj = build_observed_trajectory(case, index, grid)

    assert traj.interaction_count.sum() == 2  # total interactions


def test_stance_and_arousal_are_nan():
    """Without precomputed signals, stance/arousal must be NaN with mask=False."""
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    traj = build_observed_trajectory(case, index, grid)

    assert np.all(np.isnan(traj.stance_mean))
    assert np.all(np.isnan(traj.arousal_mean))
    assert not np.any(traj.observation_masks.get("stance", np.zeros(1)))


def test_missing_values_not_zero():
    """NaN stance must never become zero."""
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    traj = build_observed_trajectory(case, index, grid)

    assert not np.any(traj.stance_mean == 0.0)


def test_shape_consistency():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    traj = build_observed_trajectory(case, index, grid)

    T_plus_1 = grid.final_step + 1  # steps 0..total_steps
    N = len(index.idx_to_user)
    assert traj.active_count.shape == (T_plus_1,)
    assert traj.active_mask.shape == (T_plus_1, N)
    assert traj.cumulative_users.shape == (T_plus_1,)
    assert len(traj.steps) == T_plus_1


def test_node_index_root_is_zero():
    case = _make_case()
    index = NodeIndex.from_case(case)
    assert index.user_to_idx["root"] == 0
    assert index.idx_to_user[0] == "root"


def test_node_index_deterministic():
    """Same case must produce identical index every time."""
    case = _make_case()
    i1 = NodeIndex.from_case(case)
    i2 = NodeIndex.from_case(case)
    assert i1.user_to_idx == i2.user_to_idx
    assert i1.idx_to_user == i2.idx_to_user


def test_tail_steps_are_unobserved():
    """Steps beyond last_data_step must be masked as False."""
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=3)
    traj = build_observed_trajectory(case, index, grid)

    # last interaction is at step 2 (9:30), so last_data_step=2
    assert grid.last_data_step == 2
    assert grid.final_step == 5  # 2 + 3tail
    assert grid.n_trajectory_points == 6

    # Steps 3, 4, 5 are tail — must be unobserved
    for key in ["active_count", "comment_count", "repost_count",
                "interaction_count"]:
        mask = traj.observation_masks[key]
        assert np.all(mask[:grid.last_data_step + 1])  # data steps: True
        assert not np.any(mask[grid.last_data_step + 1:])  # tail: False


def _make_large_case():
    """Case with enough interactions for temporal split (>=4 data steps)."""
    root = RootPost(
        post_id="ev2", user_id="root",
        timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
        text="root", label="fake", expert_analysis=None,
    )
    interactions = tuple(
        InteractionRecord(
            interaction_id=f"c{i}", root_post_id="ev2",
            user_id=f"u{i}",
            timestamp=datetime(2020, 1, 1, 8 + i, 0, tzinfo=timezone.utc),
            kind="comment", text=f"c{i}",
        )
        for i in range(1, 6)  # 5 users, 5 data steps
    )
    return EventCase(
        case_id="ev2", source_dataset="CHECKED",
        root=root, interactions=interactions,
    )


def test_tail_steps_excluded_from_loss():
    """Tail-zero observations must not influence calibration loss."""
    from dynamics_simulation.calibration.objective import (
        compute_replay_loss, LossWeights,
    )
    from dynamics_simulation.calibration.split import TemporalSplit

    case = _make_large_case()
    index = NodeIndex.from_case(case)
    # 5 interactions at t+1h through t+5h, step_hours=1 → last_data_step=5
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=3)
    traj = build_observed_trajectory(case, index, grid)

    # last_data_step=5, final_step=8 (5+3), n_trajectory_points=9
    assert grid.last_data_step == 5  # enough for split (>=4)

    # Build a dummy simulated dict that matches observed exactly
    sim = {
        "active_count": traj.active_count.copy().astype(np.float64),
    }
    # Split on data-only steps
    split = TemporalSplit.by_fraction(
        total_steps=grid.last_data_step,
        train_fraction=0.7,
    )
    weights = LossWeights(active_count=1.0)
    masks = {"active_count": traj.observation_masks["active_count"]}

    loss = compute_replay_loss(
        {"active_count": traj.active_count.astype(np.float64)},
        sim, split, weights, masks,
    )
    # Perfect match → zero loss
    assert loss.train_total == 0.0
    assert loss.val_total == 0.0

    # Now perturb the tail (steps beyond last_data_step)
    sim["active_count"] = sim["active_count"].copy()
    sim["active_count"][grid.last_data_step + 1:] = 999.0
    loss2 = compute_replay_loss(
        {"active_count": traj.active_count.astype(np.float64)},
        sim, split, weights, masks,
    )
    # Tail perturbation must not change loss (masked out)
    assert loss2.train_total == 0.0
    assert loss2.val_total == 0.0
