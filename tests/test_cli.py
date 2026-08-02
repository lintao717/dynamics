"""Tests for CLI workflow entry points."""

import json
import sys
from pathlib import Path
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "checked_case.json"


def test_inspect_checked_dataset(tmp_path, capsys):
    from dynamics_simulation.cli.inspect_dataset import main
    ret = main(["--dataset", "checked", "--root", str(FIXTURE.parent),
               "--top", "5"])
    assert ret == 0  # invalid dir for iter, but should not crash
    # Actually the fixture dir doesn't contain CHECKED files directly


def test_inspect_missing_root(capsys):
    from dynamics_simulation.cli.inspect_dataset import main
    ret = main(["--dataset", "checked", "--root", "nonexistent_dir_xyz"])
    assert ret == 1


def test_replay_event_missing_file(capsys):
    from dynamics_simulation.cli.replay_event import main
    ret = main(["--case", "nonexistent.json"])
    assert ret == 1


def test_replay_event_broadcast_output(tmp_path):
    from dynamics_simulation.cli.replay_event import main
    out = tmp_path / "replay.json"
    ret = main([
        "--case", str(FIXTURE),
        "--mode", "broadcast",
        "--seeds", "42,99",
        "--step-hours", "1.0",
        "--output", str(out),
    ])
    assert ret == 0
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["case_id"] == "root-hash"
    assert data["network_mode"] == "broadcast"
    assert len(data["seeds"]) == 2


def test_replay_event_unknown_mode(capsys):
    from dynamics_simulation.cli.replay_event import main
    with pytest.raises(SystemExit):
        main(["--case", str(FIXTURE), "--mode", "nonexistent_mode"])


def test_calibrate_event_missing_file(capsys):
    from dynamics_simulation.cli.calibrate_event import main
    ret = main(["--case", "nonexistent.json"])
    assert ret == 1


def test_calibrate_event_invalid_train_fraction():
    """CLI must exit non-zero for invalid --train-fraction (via ValueError)."""
    from dynamics_simulation.cli.calibrate_event import main
    fixture = Path(__file__).parent / "fixtures" / "checked_case.json"
    if not fixture.exists():
        pytest.skip("checked_case.json fixture not available")
    # 0.5 is valid range but may fail due to too few steps in test fixture.
    # Either way, CLI should not crash — it should return non-zero or 0
    # depending on whether the fixture has enough steps.
    ret = main([
        "--case", str(fixture),
        "--train-fraction", "1.5",  # out of range (0, 1)
    ])
    assert ret != 0  # must fail for invalid fraction
