"""Tests for CHECKED dataset adapter."""

import os
import tempfile
from pathlib import Path
import json
import pytest
from dynamics_simulation.data.checked import (
    load_checked_case, iter_checked_cases,
    DatasetLoadReport, DatasetLoadFailure,
)


FIXTURE = Path(__file__).parent / "fixtures" / "checked_case.json"
EMPTY_TEXT_FIXTURE = Path(__file__).parent / "fixtures" / "checked_empty_text.json"


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


# ── P0-1: empty text handling ──

def test_empty_text_comment_does_not_drop_entire_case():
    """A comment with text="" must not cause the whole event to be skipped."""
    case = load_checked_case(EMPTY_TEXT_FIXTURE)
    assert case.case_id == "empty-text-hash"
    # All 3 comments + 1 repost should be loaded
    assert len(case.interactions) == 4
    # Empty-text interaction should have text=""
    empty_text_interactions = [
        ix for ix in case.interactions if ix.text == ""
    ]
    assert len(empty_text_interactions) >= 1


def test_empty_text_interaction_kind_preserved():
    """Interactions with empty text must still preserve their kind."""
    case = load_checked_case(EMPTY_TEXT_FIXTURE)
    kinds = [ix.kind for ix in case.interactions]
    assert "comment" in kinds
    assert "repost" in kinds


def test_empty_root_text_raises_value_error():
    """Root post with missing text field should still raise."""
    import tempfile
    import os

    bad = {
        "label": "fake", "id": "no-root-text",
        "date": "2020-04-02 08:52",
        "user_id": "u0",
        "comments": [], "reposts": [],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(bad, f)
        tmp = f.name
    try:
        with pytest.raises(ValueError, match="root.text"):
            load_checked_case(Path(tmp))
    finally:
        os.unlink(tmp)


def test_iter_checked_cases_skips_malformed_but_reports(tmp_path):
    """Valid cases are yielded; malformed ones are reported in DatasetLoadReport."""
    # Valid case
    (tmp_path / "valid.json").write_text(json.dumps({
        "label": "fake", "id": "v1", "date": "2020-04-02 08:52",
        "user_id": "u0", "text": "t", "comments": [], "reposts": [],
    }))
    # Malformed: missing root.user_id
    (tmp_path / "bad.json").write_text(json.dumps({
        "label": "fake", "id": "b1", "date": "2020-04-02 08:52",
        "text": "t", "comments": [], "reposts": [],
    }))

    cases, report = iter_checked_cases(tmp_path, report=True)
    case_list = list(cases)

    assert len(case_list) == 1
    assert case_list[0].case_id == "v1"
    assert report.scanned_files == 2
    assert report.loaded_cases == 1
    assert report.failed_files == 1
    assert len(report.failures) == 1
    assert report.failures[0].file_name == "bad.json"


def test_dataset_load_report_counts_empty_text(tmp_path):
    """Cases with empty-text interactions should load successfully."""
    (tmp_path / "case.json").write_text(json.dumps({
        "label": "fake", "id": "with-empty",
        "date": "2020-04-02 08:52",
        "user_id": "u0", "text": "ok",
        "comments": [
            {"id": "c1", "date": "2020-04-02 09:10", "user_id": "u1", "text": ""}
        ],
        "reposts": [],
    }))

    cases, report = iter_checked_cases(tmp_path, report=True)
    case_list = list(cases)

    assert len(case_list) == 1
    assert report.scanned_files == 1
    assert report.loaded_cases == 1
    assert report.failed_files == 0
    assert report.empty_text_comments == 1
    assert report.empty_text_reposts == 0

    # Verify empty text comment has text=""
    case = case_list[0]
    assert case.interactions[0].text == ""


def test_dataset_load_report_tracks_failure_reasons(tmp_path):
    """Each failure in the report must identify the failing file and cause."""
    # Create one valid and two invalid files
    (tmp_path / "ok.json").write_text(json.dumps({
        "label": "fake", "id": "ok", "date": "2020-04-02 08:52",
        "user_id": "u0", "text": "t", "comments": [], "reposts": [],
    }))
    (tmp_path / "bad_date.json").write_text(json.dumps({
        "label": "fake", "id": "bd", "date": "NOT_A_DATE",
        "user_id": "u0", "text": "t", "comments": [], "reposts": [],
    }))
    (tmp_path / "bad_user.json").write_text(json.dumps({
        "label": "fake", "id": "bu", "date": "2020-04-02 08:52",
        "text": "t", "comments": [], "reposts": [],
    }))

    cases, report = iter_checked_cases(tmp_path, report=True)
    list(cases)  # consume

    assert report.scanned_files == 3
    assert report.loaded_cases == 1
    assert report.failed_files == 2
    assert len(report.failures) == 2
    failure_files = {f.file_name for f in report.failures}
    assert failure_files == {"bad_date.json", "bad_user.json"}
    for f in report.failures:
        assert f.reason  # non-empty reason string


def test_iter_checked_cases_default_no_report(tmp_path):
    """Without report=True, iter_checked_cases must still work as before."""
    (tmp_path / "c.json").write_text(json.dumps({
        "label": "fake", "id": "c1", "date": "2020-04-02 08:52",
        "user_id": "u0", "text": "t", "comments": [], "reposts": [],
    }))

    # Default behavior: no report
    cases = list(iter_checked_cases(tmp_path))
    assert len(cases) == 1
