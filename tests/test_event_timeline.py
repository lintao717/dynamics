"""Tests for event input timeline and ExternalInputs validation."""

from datetime import datetime, timezone
import numpy as np
import pytest
from dynamics_simulation.transitions import ExternalInputs
from dynamics_simulation.data.schema import (
    EventCase, RootPost, InteractionRecord,
)
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.timeline import (
    BroadcastExposureConfig,
    EventInputTimeline,
)


# ── ExternalInputs validation ──

def test_external_inputs_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ExternalInputs(media_exposure=np.zeros(4)).resolve(3)


def test_external_inputs_rejects_info_evidence_out_of_range():
    with pytest.raises(ValueError, match="info_evidence"):
        ExternalInputs(info_evidence=np.array([0.0, 2.0, 0.0])).resolve(3)


def test_external_inputs_rejects_nan():
    with pytest.raises(ValueError):
        ExternalInputs(media_exposure=np.array([np.nan, 0.0, 0.0])).resolve(3)


def test_external_inputs_rejects_inf():
    with pytest.raises(ValueError):
        ExternalInputs(media_exposure=np.array([np.inf, 0.0, 0.0])).resolve(3)


def test_external_inputs_valid_input_passes():
    ext = ExternalInputs(
        media_exposure=np.array([0.1, 0.2, 0.3]),
        info_evidence=np.array([-0.5, 0.0, 0.5]),
        official_info=np.array([-1.0, 0.0, 1.0]),
        shock=0.5, novelty=0.3, staleness=0.7, V=0.1,
    )
    result = ext.resolve(3)
    assert "media_exposure" in result
    assert result["media_exposure"].shape == (3,)


def test_external_inputs_scalar_range_validation():
    with pytest.raises(ValueError, match="shock"):
        ExternalInputs(shock=1.5).resolve(1)
    with pytest.raises(ValueError, match="novelty"):
        ExternalInputs(novelty=-0.1).resolve(1)


# ── Broadcast exposure ──

def test_broadcast_exposure_decays():
    cfg = BroadcastExposureConfig(
        amplitude=1.0, exposure_half_life_steps=4.0,
    )
    v0 = cfg.exposure_at(0)  # step 0: amplitude
    v4 = cfg.exposure_at(4)  # should be halved
    v8 = cfg.exposure_at(8)  # should be quartered
    assert v0 == pytest.approx(1.0)
    assert v4 == pytest.approx(0.5)
    assert v8 == pytest.approx(0.25)


def test_broadcast_novelty_decays():
    cfg = BroadcastExposureConfig(
        novelty_at_root=1.0, novelty_half_life_steps=6.0,
    )
    v0 = cfg.novelty_at(0)
    v6 = cfg.novelty_at(6)
    v12 = cfg.novelty_at(12)
    assert v0 == pytest.approx(1.0)
    assert v6 == pytest.approx(0.5)
    assert v12 == pytest.approx(0.25)


def test_broadcast_staleness_saturates():
    cfg = BroadcastExposureConfig(staleness_tau_steps=12.0)
    s0 = cfg.staleness_at(0)
    s12 = cfg.staleness_at(12)
    s48 = cfg.staleness_at(48)
    assert s0 == pytest.approx(0.0)  # no staleness at t=0
    assert s12 == pytest.approx(1.0 - np.exp(-1.0), rel=1e-3)  # at tau
    assert s48 > 0.98  # nearly saturated


def test_broadcast_staleness_independent_of_total_steps():
    """Staleness depends only on step and fixed tau, NOT on event duration."""
    cfg = BroadcastExposureConfig(staleness_tau_steps=10.0)
    # Same step, different hypothetical durations — must give same staleness
    s1 = cfg.staleness_at(5)
    s2 = cfg.staleness_at(5)  # no total_steps argument
    assert s1 == s2


# ── EventInputTimeline ──

def _make_case():
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
    return EventCase(
        case_id="ev1", source_dataset="CHECKED",
        root=root, interactions=(c1,),
    )


def test_timeline_root_exposure_is_zero():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    timeline = EventInputTimeline(case, index, grid)
    root_idx = index.user_to_idx["root"]

    inputs = timeline.inputs_at(index.n, 0)
    resolved = inputs.resolve(index.n)
    assert resolved["media_exposure"][root_idx] == 0.0


def test_timeline_non_root_gets_exposure():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    timeline = EventInputTimeline(case, index, grid)
    u1_idx = index.user_to_idx["u1"]

    inputs = timeline.inputs_at(index.n, 0)
    resolved = inputs.resolve(index.n)
    assert resolved["media_exposure"][u1_idx] > 0.0


def test_timeline_is_deterministic():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)

    t1 = EventInputTimeline(case, index, grid)
    t2 = EventInputTimeline(case, index, grid)

    for step in range(3):
        i1 = t1.inputs_at(index.n, step)
        i2 = t2.inputs_at(index.n, step)
        assert np.allclose(
            i1.resolve(index.n)["media_exposure"],
            i2.resolve(index.n)["media_exposure"],
        )


def test_timeline_no_interaction_dependency():
    """Timeline should depend only on root time, not on interactions."""
    case = _make_case()
    index = NodeIndex.from_case(case)

    # Case without interactions but same root
    root = RootPost(
        post_id="ev1", user_id="root",
        timestamp=case.root.timestamp,
        text="root", label="fake", expert_analysis=None,
    )
    case_empty = EventCase(
        case_id="ev1", source_dataset="CHECKED",
        root=root, interactions=(),
    )

    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    grid_empty = TimeGrid.from_case(case_empty, step_hours=1.0, tail_steps=1)

    tl = EventInputTimeline(case, index, grid)
    tl_empty = EventInputTimeline(case_empty, index, grid_empty)

    for step in range(3):
        i1 = tl.inputs_at(index.n, step)
        i2 = tl_empty.inputs_at(index.n, step)
        assert np.allclose(
            i1.resolve(index.n)["media_exposure"],
            i2.resolve(index.n)["media_exposure"],
        )
