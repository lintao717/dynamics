"""Tests for chronological split and masked replay loss."""

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
    w = LossWeights()
    assert w.active_count == 1.0
    assert w.cumulative_users == 0.5
    assert w.interaction_count == 0.25
    assert w.peak_time == 0.25
    assert w.final_size == 0.25
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

    observed = {
        "active_count": obs_active,
        "cumulative_users": obs_active,
        "interaction_count": obs_active,
    }
    simulated = {
        "active_count": sim_active,
        "cumulative_users": sim_active,
        "interaction_count": sim_active,
    }
    masks = {
        "active_count": obs_mask,
        "cumulative_users": obs_mask,
        "interaction_count": obs_mask,
    }

    weights = LossWeights(stance=0.0, arousal=0.0)
    loss = compute_replay_loss(observed, simulated, split, weights, masks)

    # Train-on-curve loss: active_count/ cumulative_users/interaction are NRMSE-zero
    # (exact match). peak_time and final_size are scalar — they look at full array,
    # so train_total > 0 even with perfect curve fit.
    assert loss.train_active_count == pytest.approx(0.0, abs=1e-6)
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
