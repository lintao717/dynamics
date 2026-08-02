"""Tests for chronological split and masked replay loss."""

from datetime import datetime, timezone

import numpy as np
import pytest
from dynamics_simulation.calibration.split import TemporalSplit
from dynamics_simulation.calibration.objective import (
    LossWeights, ReplayLoss, compute_replay_loss,
)


def test_temporal_split_by_fraction():
    split = TemporalSplit.by_fraction(total_steps=20, train_fraction=0.7)
    assert split.train_end_step == 14  # 20 * 0.7 = 14
    assert split.total_steps == 20


def test_temporal_split_minimum_steps():
    """Must have at least 4 total steps."""
    with pytest.raises(ValueError):
        TemporalSplit.by_fraction(total_steps=3, train_fraction=0.7)


def test_temporal_split_train_fraction_bounds():
    with pytest.raises(ValueError):
        TemporalSplit.by_fraction(total_steps=10, train_fraction=0.0)
    with pytest.raises(ValueError):
        TemporalSplit.by_fraction(total_steps=10, train_fraction=1.0)


def test_temporal_split_at_least_three_train():
    split = TemporalSplit.by_fraction(total_steps=8, train_fraction=0.2)
    assert split.train_end_step >= 3  # floor(max(1.6, 3)) = 3
    assert split.train_end_step <= 7  # at most total-1


def test_split_train_val_separation():
    """Train and validation steps must not overlap."""
    split = TemporalSplit.by_fraction(total_steps=10, train_fraction=0.7)
    # Steps: 0,1,2,...,train_end_step are train
    # Steps: train_end_step+1,...,total_steps are val
    assert split.train_end_step < split.total_steps


def test_loss_weights_defaults():
    """Default LossWeights: only active_count is non-zero.
    cumulative_users, interaction_count, stance, arousal are not
    produced by the current simulator and default to 0.
    peak_time and final_size are per-segment scalars; default to 0.
    """
    w = LossWeights()
    assert w.active_count == 1.0
    assert w.cumulative_users == 0.0
    assert w.interaction_count == 0.0
    assert w.peak_time == 0.0
    assert w.final_size == 0.0
    assert w.stance == 0.0
    assert w.arousal == 0.0


def test_compute_replay_loss_train_val_separate():
    """Perfect training fit + bad validation = train loss 0, val loss > 0."""
    T = 20
    split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)

    # Observed: simple linear ramp
    obs_active = np.arange(T + 1, dtype=np.float64)
    obs_mask = np.ones(T + 1, dtype=bool)

    # Simulated: perfect on train, wrong on val
    sim_active = obs_active.copy()
    sim_active[split.train_end_step + 1:] = 0.0  # diverge after train

    observed = {"active_count": obs_active}
    simulated = {"active_count": sim_active}
    masks = {"active_count": obs_mask}

    weights = LossWeights(
        active_count=1.0, cumulative_users=0.0,
        interaction_count=0.0, peak_time=0.0, final_size=0.0,
        stance=0.0, arousal=0.0,
    )
    loss = compute_replay_loss(observed, simulated, split, weights, masks)

    # Train active_count NRMSE = 0 (exact match on train segment)
    assert loss.train_active_count == pytest.approx(0.0, abs=1e-6)
    # Validation active_count NRMSE > 0
    assert loss.val_active_count > 0.0
    # Total train loss = 0 (only active_count has non-zero weight)
    assert loss.train_total == pytest.approx(0.0, abs=1e-6)
    assert loss.val_total > 0.0


def test_compute_replay_loss_masked_target():
    """Targets with zero valid observations raise ValueError."""
    T = 10
    split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)
    obs_active = np.zeros(T + 1)
    sim_active = np.zeros(T + 1)
    # All masks False → no valid observations
    masks = {"active_count": np.zeros(T + 1, dtype=bool)}

    weights = LossWeights(active_count=1.0, cumulative_users=0.0,
                          interaction_count=0.0, peak_time=0.0, final_size=0.0,
                          stance=0.0, arousal=0.0)
    with pytest.raises(ValueError, match="No valid observations"):
        compute_replay_loss(
            {"active_count": obs_active},
            {"active_count": sim_active},
            split, weights, masks,
        )


def test_replay_loss_components():
    T = 10
    split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)
    obs = {"active_count": np.ones(T + 1), "cumulative_users": np.ones(T + 1)}
    sim = {"active_count": np.zeros(T + 1), "cumulative_users": np.zeros(T + 1)}
    masks = {
        "active_count": np.ones(T + 1, dtype=bool),
        "cumulative_users": np.ones(T + 1, dtype=bool),
    }
    weights = LossWeights(
        active_count=1.0, cumulative_users=0.0,
        interaction_count=0.0, peak_time=0.0, final_size=0.0,
        stance=0.0, arousal=0.0,
    )
    loss = compute_replay_loss(obs, sim, split, weights, masks)
    # active_count NRMSE should be 1.0 (zero predictions vs ones)
    assert loss.train_active_count > 0.0
    # cumulative_users has weight 0 → no contribution
    assert loss.train_total == pytest.approx(loss.train_active_count)


def test_peak_time_uses_train_segment_only():
    """Peak time must be computed from the training segment only,
    not from the full array (which would leak validation data)."""
    T = 20
    split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.5)
    train_end = split.train_end_step  # = 10

    # Observed: peak at step 8 (in train) on train, step 18 on val
    obs_active = np.zeros(T + 1)
    obs_active[8] = 10.0   # train peak
    obs_active[18] = 100.0  # val peak (much bigger but NOT visible in train)

    # Simulated: peak at step 5 (in train)
    sim_active = np.zeros(T + 1)
    sim_active[5] = 10.0

    weights = LossWeights(
        active_count=0.0, cumulative_users=0.0,
        interaction_count=0.0, peak_time=1.0, final_size=0.0,
        stance=0.0, arousal=0.0,
    )
    loss = compute_replay_loss(
        {"active_count": obs_active},
        {"active_count": sim_active},
        split, weights,
        {"active_count": np.ones(T + 1, dtype=bool)},
    )

    # train peak: obs=8, sim=5 → error |8-5|/20 = 0.15
    assert loss.train_peak_time == pytest.approx(0.15, abs=0.01)
    # val peak: still uses per-segment logic (val segment has peak at 18)
    # but since sim is all flat in val, this is a big error
    assert loss.val_peak_time > 0.0


def test_final_size_uses_train_segment_only():
    """Final size must be computed per segment, not from the last step
    of the full trajectory."""
    T = 20
    split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.5)
    train_end = split.train_end_step  # = 10

    obs_active = np.arange(T + 1, dtype=np.float64)
    sim_active = obs_active.copy()
    # Perfect match on train, diverge on val
    sim_active[train_end + 1:] = sim_active[train_end]  # flat after train

    weights = LossWeights(
        active_count=0.0, cumulative_users=0.0,
        interaction_count=0.0, peak_time=0.0, final_size=1.0,
        stance=0.0, arousal=0.0,
    )
    loss = compute_replay_loss(
        {"active_count": obs_active},
        {"active_count": sim_active},
        split, weights,
        {"active_count": np.ones(T + 1, dtype=bool)},
    )

    # train final = obs[train_end] vs sim[train_end] — identical
    assert loss.train_final_size == pytest.approx(0.0, abs=0.01)
    # val final = obs[-1] vs sim[-1] — different (obs continues, sim flat)
    assert loss.val_final_size > 0.0


def test_explicit_split_rejects_tail_steps():
    """fit_stage1 must reject an explicit split that includes tail steps."""
    from dynamics_simulation.data.schema import (
        EventCase, RootPost, InteractionRecord,
    )
    from dynamics_simulation.data.networks import ReplayNetworkMode
    from dynamics_simulation.replay.config import ReplayConfig
    from dynamics_simulation.calibration.estimator import fit_stage1
    from dynamics_simulation.calibration.split import TemporalSplit
    from dynamics_simulation.config import default_params

    root = RootPost(
        post_id="s1", user_id="root",
        timestamp=datetime.now(timezone.utc),
        text="test", label="fake", expert_analysis=None,
    )
    interactions = tuple(
        InteractionRecord(
            interaction_id=f"c{i}", root_post_id="s1",
            user_id=f"u{i}",
            timestamp=datetime.now(timezone.utc),
            kind="comment", text=f"c{i}",
        )
        for i in range(1, 8)  # enough steps for split
    )
    case = EventCase(
        case_id="s1", source_dataset="CHECKED",
        root=root, interactions=interactions,
    )

    # Explicit split using final_step (which includes tail) must be rejected
    from dynamics_simulation.data.timegrid import TimeGrid
    grid = TimeGrid.from_case(case, step_hours=0.01, tail_steps=4)
    # last_data_step covers the interactions, final_step adds 4 tail
    bad_split = TemporalSplit.by_fraction(
        total_steps=grid.final_step, train_fraction=0.7,
    )
    config = ReplayConfig(
        step_hours=0.01, tail_steps=4,
        network_mode=ReplayNetworkMode.BROADCAST,
        seeds=(42,),
    )
    with pytest.raises(ValueError, match="tail"):
        fit_stage1(case, default_params(), config, split=bad_split)
