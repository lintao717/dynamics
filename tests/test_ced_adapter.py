"""Tests for CED (Chinese Rumor Dataset) compatibility adapter."""

from pathlib import Path
import pytest
from dynamics_simulation.data.ced import load_ced_case


ORIG = Path(__file__).parent / "fixtures" / "ced_original.json"
INTER = Path(__file__).parent / "fixtures" / "ced_interactions.json"


def test_load_ced_case_kind_is_interaction():
    """CED cannot distinguish comment vs repost — all must be 'interaction'."""
    case = load_ced_case(ORIG, INTER, label="fake")
    for ix in case.interactions:
        assert ix.kind == "interaction"


def test_load_ced_case_root_first_user_order():
    """Root author must be first in user_ids."""
    case = load_ced_case(ORIG, INTER, label="fake")
    assert case.user_ids[0] == "root-user"
    assert len(case.user_ids) == 3  # root + 2 unique interaction users


def test_load_ced_case_deterministic_case_id():
    """Case ID must be derived from the original file stem."""
    case = load_ced_case(ORIG, INTER, label="fake")
    assert case.case_id == "ced_original"


def test_load_ced_case_source_dataset():
    case = load_ced_case(ORIG, INTER, label="fake")
    assert case.source_dataset == "CED"


def test_load_ced_case_root_label():
    case = load_ced_case(ORIG, INTER, label="rumor")
    assert case.root.label == "rumor"


def test_load_ced_case_timezone_aware():
    case = load_ced_case(ORIG, INTER, label="fake")
    assert case.root.timestamp.tzinfo is not None
    for ix in case.interactions:
        assert ix.timestamp.tzinfo is not None


def test_load_ced_case_both_date_keys():
    """CED interactions may use 'data' or 'date' as timestamp key.
    Both records should be loaded successfully."""
    case = load_ced_case(ORIG, INTER, label="fake")
    assert len(case.interactions) == 2


def test_load_ced_case_unix_timestamp_root():
    """Root timestamp may be a Unix epoch integer."""
    case = load_ced_case(ORIG, INTER, label="fake")
    # 1400000000 = 2014-05-13 16:26:40 UTC
    assert case.root.timestamp.year == 2014


def test_load_ced_case_missing_original_file():
    with pytest.raises(FileNotFoundError):
        load_ced_case(Path("nonexistent.json"), INTER, label="fake")


def test_load_ced_case_missing_interactions_file():
    with pytest.raises(FileNotFoundError):
        load_ced_case(ORIG, Path("nonexistent.json"), label="fake")
