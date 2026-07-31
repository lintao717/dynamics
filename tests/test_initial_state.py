"""Tests for real-data initial AgentState construction."""

from datetime import datetime, timezone
import numpy as np
import pytest
from dynamics_simulation.data.schema import (
    EventCase, RootPost, InteractionRecord,
)
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.state import (
    TextSignals, StatePriorConfig, build_initial_state,
)
from dynamics_simulation.config import default_params
from dynamics_simulation.agents import U, E, A, D


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


def test_root_author_is_active():
    case = _make_case()
    index = NodeIndex.from_case(case)
    params = default_params()
    rng = np.random.default_rng(42)
    state = build_initial_state(case, index, params, rng)

    root_idx = index.user_to_idx["root"]
    assert state.z[root_idx] == A
    assert state.m[root_idx] == 1
    assert np.isfinite(state.o_hat[root_idx])


def test_non_root_users_are_uncertain():
    case = _make_case()
    index = NodeIndex.from_case(case)
    params = default_params()
    rng = np.random.default_rng(42)
    state = build_initial_state(case, index, params, rng)

    for uid in case.user_ids:
        if uid == "root":
            continue
        idx = index.user_to_idx[uid]
        assert state.z[idx] == U, f"user {uid} should be U, got z={state.z[idx]}"
        assert state.m[idx] == 0


def test_no_user_starts_in_exposed():
    case = _make_case()
    index = NodeIndex.from_case(case)
    params = default_params()
    rng = np.random.default_rng(42)
    state = build_initial_state(case, index, params, rng)

    assert state.n_E == 0


def test_fatigue_is_zero_for_all():
    case = _make_case()
    index = NodeIndex.from_case(case)
    params = default_params()
    rng = np.random.default_rng(42)
    state = build_initial_state(case, index, params, rng)

    assert np.all(state.f == 0.0)


def test_non_root_public_expression_is_nan():
    case = _make_case()
    index = NodeIndex.from_case(case)
    params = default_params()
    rng = np.random.default_rng(42)
    state = build_initial_state(case, index, params, rng)

    for uid in case.user_ids:
        if uid == "root":
            continue
        idx = index.user_to_idx[uid]
        assert np.isnan(state.o_hat[idx])


def test_state_n_matches_index():
    case = _make_case()
    index = NodeIndex.from_case(case)
    params = default_params()
    rng = np.random.default_rng(42)
    state = build_initial_state(case, index, params, rng)

    assert state.n == len(index)


def test_text_signals_reject_unknown_user():
    case = _make_case()
    index = NodeIndex.from_case(case)
    params = default_params()
    rng = np.random.default_rng(42)
    signals = TextSignals(
        stance_by_user={"nonexistent": 0.5},
        arousal_by_user={},
    )
    with pytest.raises(ValueError, match="not in NodeIndex"):
        build_initial_state(case, index, params, rng, signals=signals)


def test_text_signals_reject_stance_out_of_range():
    with pytest.raises(ValueError):
        TextSignals(stance_by_user={"u": 1.5}, arousal_by_user={})


def test_text_signals_reject_arousal_out_of_range():
    with pytest.raises(ValueError):
        TextSignals(stance_by_user={}, arousal_by_user={"u": -0.1})


def test_empty_case_rejected():
    """Case with no root and no interactions is still valid, just tiny."""
    # Build with root-only case (still valid)
    root = RootPost(
        post_id="ev1", user_id="root",
        timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
        text="root", label="fake", expert_analysis=None,
    )
    case = EventCase(
        case_id="ev1", source_dataset="CHECKED",
        root=root, interactions=(),
    )
    index = NodeIndex.from_case(case)
    params = default_params()
    rng = np.random.default_rng(42)
    state = build_initial_state(case, index, params, rng)
    assert state.n == 1  # root only
    assert state.z[0] == A


def test_state_prior_config_defaults():
    cfg = StatePriorConfig()
    assert cfg.initial_opinion_dist == "moderate"
    assert cfg.root_arousal_if_missing == 0.5
    assert cfg.nonroot_arousal_if_missing == 0.0
