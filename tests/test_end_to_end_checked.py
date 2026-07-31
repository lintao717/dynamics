"""End-to-end CHECKED fixture validation: load → replay → split → score."""

from pathlib import Path
import numpy as np
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.config import default_params
from dynamics_simulation.calibration.split import TemporalSplit
from dynamics_simulation.calibration.objective import (
    compute_replay_loss, LossWeights,
)


FIXTURE = Path(__file__).parent / "fixtures" / "checked_case.json"


def test_end_to_end_checked_fixture():
    """Complete pipeline: fixture → replay → split → loss → JSON."""
    case = load_checked_case(FIXTURE)
    config = ReplayConfig(
        step_hours=1.0,
        tail_steps=2,
        network_mode=ReplayNetworkMode.BROADCAST,
        seeds=(11, 23),
        max_nodes=100,
    )
    result = run_replay(case, default_params(), config)

    # Validate observed trajectory
    assert result.observed.active_count[0] == 1  # root

    # Shape consistency
    assert result.simulated_mean["active_count"].shape == \
        result.observed.active_count.shape

    # Chronological split
    T = len(result.observed.steps) - 1
    split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)

    # Score
    masks = {
        "active_count": np.ones_like(result.observed.active_count, dtype=bool),
    }
    weights = LossWeights(
        active_count=1.0, cumulative_users=0.0,
        interaction_count=0.0, peak_time=0.0, final_size=0.0,
        stance=0.0, arousal=0.0,
    )
    loss = compute_replay_loss(
        {"active_count": result.observed.active_count},
        result.simulated_mean, split, weights, masks,
    )
    assert np.isfinite(loss.train_total)

    # JSON round-trip
    d = result.to_dict()
    assert d["case_id"] == "root-hash"
    assert d["source_dataset"] == "CHECKED"
    assert len(d["seeds"]) == 2
