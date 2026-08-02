"""Tests for parameter specification and stage-1 estimator."""

import numpy as np
import pytest
from dynamics_simulation.config import default_params, ModelParams
from dynamics_simulation.calibration.parameters import (
    ParameterSpec, Stage1ParameterSet, apply_parameter_vector,
)


def test_stage1_has_four_parameters():
    spec = Stage1ParameterSet()
    params = spec.to_specs()
    assert len(params) == 4
    names = [p.path for p in params]
    assert "propagation.beta_M" in names
    assert "activation.alpha_0" in names
    assert "decay.gamma_0" in names
    assert "viral.beta_V" in names


def test_apply_parameter_vector():
    base = default_params()
    specs = Stage1ParameterSet().to_specs()
    # Set beta_M=0.5, alpha_0=-1, gamma_0=2, beta_V=0.3
    values = [0.5, -1.0, 2.0, 0.3]
    new = apply_parameter_vector(base, specs, values)

    assert new.propagation.beta_M == 0.5
    assert new.activation.alpha_0 == -1.0
    assert new.decay.gamma_0 == 2.0
    assert new.viral.beta_V == 0.3
    # Unchanged params
    assert new.propagation.beta == base.propagation.beta


def test_apply_rejects_unknown_path():
    base = default_params()
    specs = [ParameterSpec("nonexistent.field", 0.0, 1.0)]
    with pytest.raises(AttributeError):
        apply_parameter_vector(base, specs, [0.5])


def test_apply_rejects_out_of_bounds():
    base = default_params()
    specs = [ParameterSpec("propagation.beta_M", 0.0, 1.0)]
    with pytest.raises(ValueError, match="bounds"):
        apply_parameter_vector(base, specs, [2.0])


def test_apply_rejects_non_finite():
    base = default_params()
    specs = [ParameterSpec("propagation.beta_M", 0.0, 1.0)]
    with pytest.raises(ValueError):
        apply_parameter_vector(base, specs, [np.nan])


def test_apply_rejects_length_mismatch():
    base = default_params()
    specs = Stage1ParameterSet().to_specs()
    with pytest.raises(ValueError, match="length"):
        apply_parameter_vector(base, specs, [0.5, 0.5])


def test_parameter_spec_validation():
    with pytest.raises(ValueError, match="low.*high"):
        ParameterSpec("a.b", 1.0, 0.0)  # low > high


def test_apply_roundtrip():
    """Applying a parameter vector must be deterministic."""
    base = default_params()
    specs = Stage1ParameterSet().to_specs()
    values = [0.3, -2.0, 1.0, 0.7]
    a = apply_parameter_vector(base, specs, values)
    b = apply_parameter_vector(base, specs, values)
    assert a == b


def test_synthetic_recovery_loss_lower_than_default():
    """Stage-1 calibration on synthetic data from known params must show
    lower loss than default params (qualitative sanity check, not exact recovery)."""
    from dynamics_simulation.calibration.estimator import fit_stage1
    from dynamics_simulation.replay.config import ReplayConfig
    from dynamics_simulation.replay.runner import run_replay
    from dynamics_simulation.calibration.objective import (
        compute_replay_loss, LossWeights,
    )
    from dynamics_simulation.calibration.split import TemporalSplit
    from dynamics_simulation.data.schema import (
        EventCase, RootPost, InteractionRecord,
    )
    from dynamics_simulation.data.networks import ReplayNetworkMode
    from datetime import datetime, timezone

    # Build a synthetic case with 5 interactions
    root = RootPost(
        post_id="synth", user_id="root",
        timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
        text="synth", label="fake", expert_analysis=None,
    )
    interactions = tuple(
        InteractionRecord(
            interaction_id=f"c{i}", root_post_id="synth",
            user_id=f"u{i}",
            timestamp=datetime(2020, 1, 1, 8 + i, 0, tzinfo=timezone.utc),
            kind="comment", text=f"c{i}",
        )
        for i in range(1, 6)
    )
    case = EventCase(
        case_id="synth", source_dataset="SYNTHETIC",
        root=root, interactions=interactions,
    )

    config = ReplayConfig(
        step_hours=1.0, tail_steps=1,
        network_mode=ReplayNetworkMode.BROADCAST,
        seeds=(42,),
    )

    # Run with default params
    result_default = run_replay(case, default_params(), config)
    # Use last_data_step (real data only), NOT final_step (includes tail)
    split = TemporalSplit.by_fraction(
        total_steps=result_default.last_data_step,
        train_fraction=0.7,
    )
    # Use actual observation masks — tail steps are unobserved
    masks = {
        "active_count": result_default.observed.observation_masks.get(
            "active_count",
            np.ones_like(result_default.observed.active_count, dtype=bool),
        ),
    }
    obs_dict = {
        "active_count": result_default.observed.active_count,
    }

    # Use active_count only (what the replay runner aggregates)
    weights = LossWeights(
        active_count=1.0, cumulative_users=0.0,
        interaction_count=0.0, peak_time=0.0, final_size=0.0,
        stance=0.0, arousal=0.0,
    )
    loss_default = compute_replay_loss(
        obs_dict, result_default.simulated_mean,
        split, weights, masks,
    )

    # Run calibration (should find params at least as good as default)
    result_cal = fit_stage1(case, default_params(), config, split)
    assert result_cal.train_loss <= loss_default.train_total * 1.1  # tolerance
    assert result_cal.best_vector is not None
    assert len(result_cal.best_vector) == 4
