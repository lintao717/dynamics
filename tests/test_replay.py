"""Tests for historical replay: runner, config, result."""

from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
from dynamics_simulation.data.schema import (
    EventCase, RootPost, InteractionRecord,
)
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay


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
    c2 = InteractionRecord(
        interaction_id="c2", root_post_id="ev1", user_id="u2",
        timestamp=datetime(2020, 1, 1, 9, 30, tzinfo=timezone.utc),
        kind="repost", text="c2",
    )
    return EventCase(
        case_id="ev1", source_dataset="CHECKED",
        root=root, interactions=(c1, c2),
    )


def test_run_replay_deterministic():
    """Same case + config + seed must produce identical results."""
    from dynamics_simulation.config import default_params
    case = _make_case()
    params = default_params()
    config = ReplayConfig(
        step_hours=1.0, tail_steps=2,
        network_mode=ReplayNetworkMode.BROADCAST,
        seeds=(11, 23),
    )
    r1 = run_replay(case, params, config)
    r2 = run_replay(case, params, config)
    assert r1.to_dict() == r2.to_dict()


def test_run_replay_observed_trajectory():
    """Observed trajectory must match expected values."""
    from dynamics_simulation.config import default_params
    case = _make_case()
    params = default_params()
    config = ReplayConfig(
        step_hours=1.0, tail_steps=1,
        network_mode=ReplayNetworkMode.BROADCAST,
        seeds=(42,),
    )
    result = run_replay(case, params, config)
    obs = result.observed
    # Root active at step 0
    assert obs.active_count[0] == 1
    # Step 1: u1 commented
    assert obs.comment_count[1] == 1
    # Step 2: u2 reposted
    assert obs.repost_count[2] == 1


def test_run_replay_json_roundtrip():
    """ReplayResult serialization must preserve metadata and numeric arrays."""
    from dynamics_simulation.config import default_params
    case = _make_case()
    params = default_params()
    config = ReplayConfig(
        step_hours=1.0, tail_steps=1,
        network_mode=ReplayNetworkMode.BROADCAST,
        seeds=(42,),
    )
    result = run_replay(case, params, config)
    d = result.to_dict()
    # Round-trip through JSON
    s = json.dumps(d, default=str)
    d2 = json.loads(s)
    assert d2["case_id"] == "ev1"
    assert d2["source_dataset"] == "CHECKED"
    assert "observed" in d2
    assert "simulated_mean" in d2


def test_run_replay_multi_seed():
    """Multiple seeds must produce per-seed and aggregated results."""
    from dynamics_simulation.config import default_params
    case = _make_case()
    params = default_params()
    config = ReplayConfig(
        step_hours=1.0, tail_steps=1,
        network_mode=ReplayNetworkMode.BROADCAST,
        seeds=(11, 23, 37),
    )
    result = run_replay(case, params, config)
    assert len(result.per_seed) == 3
    # Mean and percentiles should be computed
    assert result.simulated_mean is not None
    assert result.simulated_p5 is not None
    assert result.simulated_p50 is not None
    assert result.simulated_p95 is not None


def test_replay_config_defaults():
    cfg = ReplayConfig()
    assert cfg.step_hours == 1.0
    assert cfg.tail_steps == 4
    assert cfg.network_mode == ReplayNetworkMode.BROADCAST
    assert cfg.seeds == (11, 23, 37, 53, 71)
    assert cfg.max_nodes == 1000
