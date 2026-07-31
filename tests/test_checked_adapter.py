"""Tests for CHECKED dataset adapter."""

from pathlib import Path
import json
import pytest
from dynamics_simulation.data.checked import load_checked_case, iter_checked_cases


FIXTURE = Path(__file__).parent / "fixtures" / "checked_case.json"


def test_load_checked_case_preserves_comment_repost_types():
    """CHECKED adapter must distinguish comments from reposts and sort by time."""
    case = load_checked_case(FIXTURE)
    assert case.case_id == "root-hash"
    assert case.root.user_id == "root-user-hash"
    assert [x.kind for x in case.interactions] == [
        "comment", "repost", "comment", "repost"
    ]
    # Root first, then by earliest interaction timestamp:
    # u1 first at 09:10, u3 at 09:30, u2 at 10:10
    assert case.user_ids == ("root-user-hash", "u1", "u3", "u2")


def test_load_checked_case_root_fields():
    case = load_checked_case(FIXTURE)
    assert case.root.label == "fake"
    assert case.root.text == "root text"
    assert case.root.expert_analysis == "expert analysis"
    assert case.source_dataset == "CHECKED"


def test_load_checked_case_is_timezone_aware():
    case = load_checked_case(FIXTURE)
    assert case.root.timestamp.tzinfo is not None
    for ix in case.interactions:
        assert ix.timestamp.tzinfo is not None


def test_load_checked_case_rejects_missing_root_user():
    """Adapter must raise ValueError for missing root user_id."""
    import tempfile
    import os

    bad = {
        "label": "fake", "id": "bad",
        "date": "2020-04-02 08:52",
        "text": "no user id",
        "comments": [], "reposts": [],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(bad, f)
        tmp = f.name
    try:
        with pytest.raises(ValueError, match="user_id"):
            load_checked_case(Path(tmp))
    finally:
        os.unlink(tmp)


def test_load_checked_case_rejects_malformed_timestamp():
    """Adapter must raise ValueError for unparseable timestamps."""
    import tempfile
    import os

    bad = {
        "label": "fake", "id": "bad-time",
        "date": "2020-04-02 08:52",
        "user_id": "u0",
        "text": "root",
        "comments": [
            {"id": "c1", "date": "NOT_A_DATE", "user_id": "u1", "text": "c1"}
        ],
        "reposts": [],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(bad, f)
        tmp = f.name
    try:
        with pytest.raises(ValueError):
            load_checked_case(Path(tmp))
    finally:
        os.unlink(tmp)


def test_iter_checked_cases_yields_all(tmp_path):
    """iter_checked_cases must yield EventCase for each JSON file."""
    # Create two minimal cases
    for i in range(2):
        case = {
            "label": "fake",
            "id": f"case-{i}",
            "date": "2020-04-02 08:52",
            "user_id": f"root-{i}",
            "text": f"text {i}",
            "comments": [],
            "reposts": [],
        }
        (tmp_path / f"case_{i}.json").write_text(json.dumps(case))

    cases = list(iter_checked_cases(tmp_path))
    assert len(cases) == 2
    assert {c.case_id for c in cases} == {"case-0", "case-1"}


def test_iter_checked_cases_label_filter(tmp_path):
    """Label filter must only return matching cases."""
    (tmp_path / "fake.json").write_text(json.dumps({
        "label": "fake", "id": "f1", "date": "2020-04-02 08:52",
        "user_id": "u0", "text": "t", "comments": [], "reposts": [],
    }))
    (tmp_path / "real.json").write_text(json.dumps({
        "label": "real", "id": "r1", "date": "2020-04-02 08:52",
        "user_id": "u0", "text": "t", "comments": [], "reposts": [],
    }))

    fakes = list(iter_checked_cases(tmp_path, label="fake"))
    assert len(fakes) == 1
    assert fakes[0].case_id == "f1"

    reals = list(iter_checked_cases(tmp_path, label="real"))
    assert len(reals) == 1
    assert reals[0].case_id == "r1"
