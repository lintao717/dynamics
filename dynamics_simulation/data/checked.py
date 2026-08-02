"""
CHECKED dataset adapter.

Converts CHECKED root JSON files into canonical EventCase records.

CHECKED format per file:
{
  "label": "fake" | "real",
  "id": "<root-hash>",
  "date": "YYYY-MM-DD HH:MM",
  "user_id": "<user-hash>",
  "text": "<root text>",
  "analysis": "<expert analysis>",
  "comments": [{...}, ...],
  "reposts":  [{...}, ...]
}

Nested records may use variant keys: id/comment_id/repost_id,
user_id/uid, date/data/time, text/content. All datetimes are
parsed as Asia/Shanghai and converted to UTC.

Comments and reposts may have empty text ("") — this is valid
CHECKED data and must not cause the entire event to be discarded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Tuple

from zoneinfo import ZoneInfo

from dynamics_simulation.data.schema import (
    EventCase,
    InteractionRecord,
    InteractionKind,
    RootPost,
)

# ── Field alias tables ──

ID_KEYS = ("id", "comment_id", "repost_id")
USER_KEYS = ("user_id", "uid")
DATE_KEYS = ("date", "data", "time")
TEXT_KEYS = ("text", "content")

# Default timezone for CHECKED naive timestamps
CHECKED_TZ = ZoneInfo("Asia/Shanghai")


# ── Audit / reporting types ──

@dataclass
class DatasetLoadFailure:
    """Record of a single file that could not be loaded."""
    file_name: str
    reason: str
    error_type: str = ""


@dataclass
class DatasetLoadReport:
    """Aggregated statistics from a dataset scan."""

    scanned_files: int = 0
    loaded_cases: int = 0
    failed_files: int = 0
    empty_text_comments: int = 0
    empty_text_reposts: int = 0
    timestamp_errors: int = 0
    field_missing_errors: int = 0
    failures: list[DatasetLoadFailure] = field(default_factory=list)


# ── Field extraction ──

def _pick_required_string(
    d: dict,
    keys: tuple[str, ...],
    field_name: str,
    path: Path,
) -> str:
    """Pick the first matching key from *keys* in *d*, raising on missing/empty.

    Use this for ID fields, user IDs, and date strings that MUST be
    non-empty strings.
    """
    for k in keys:
        if k in d:
            val = d[k]
            if isinstance(val, str) and val.strip():
                return val.strip()
    raise ValueError(
        f"Missing or empty field '{field_name}' (tried {keys}) in {path}"
    )


def _pick_optional_text(
    d: dict,
    keys: tuple[str, ...],
) -> str:
    """Pick the first matching key from *keys* in *d*, returning "" on missing/empty.

    Use this for comment and repost text fields — CHECKED contains
    real cases with ``text: ""``.
    """
    for k in keys:
        if k in d:
            val = d[k]
            if val is None:
                return ""
            if isinstance(val, str):
                return val  # preserve empty string
            return str(val)
    return ""


def _parse_datetime(raw: str, path: Path, field_name: str) -> datetime:
    """Parse a naive datetime string in CHECKED timezone → UTC."""
    raw = raw.strip()
    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]
    for fmt in formats:
        try:
            dt_naive = datetime.strptime(raw, fmt)
            dt_local = dt_naive.replace(tzinfo=CHECKED_TZ)
            return dt_local.astimezone(timezone.utc)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse datetime '{raw}' ({field_name}) in {path}"
    )


# ── Main loading functions ──

def load_checked_case(path: Path) -> EventCase:
    """Load a single CHECKED JSON file as an EventCase.

    Args:
        path: Path to a CHECKED root JSON file.

    Returns:
        A validated EventCase.

    Raises:
        ValueError: If required fields are missing or timestamps unparseable.
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # ── Root post ──
    # ID, user_id, date are REQUIRED non-empty strings
    root_id = _pick_required_string(raw, ID_KEYS, "root.id", path)
    root_user = _pick_required_string(raw, USER_KEYS, "root.user_id", path)
    root_date = _pick_required_string(raw, DATE_KEYS, "root.date", path)
    # Text is required for the root post
    root_text = _pick_required_string(raw, TEXT_KEYS, "root.text", path)
    label = raw.get("label")
    expert_analysis = raw.get("analysis")

    root = RootPost(
        post_id=root_id,
        user_id=root_user,
        timestamp=_parse_datetime(root_date, path, "root.date"),
        text=root_text,
        label=label if isinstance(label, str) else None,
        expert_analysis=(
            expert_analysis if isinstance(expert_analysis, str)
            else None
        ),
    )

    # ── Interactions ──
    interactions: list[InteractionRecord] = []

    for kind, key in [("comment", "comments"), ("repost", "reposts")]:
        for item in raw.get(key, []):
            ix_id = _pick_required_string(item, ID_KEYS, f"{key}.id", path)
            ix_user = _pick_required_string(item, USER_KEYS, f"{key}.user_id", path)
            ix_date = _pick_required_string(item, DATE_KEYS, f"{key}.date", path)
            # Text may be empty — this is valid CHECKED data
            ix_text = _pick_optional_text(item, TEXT_KEYS)

            interactions.append(InteractionRecord(
                interaction_id=ix_id,
                root_post_id=root_id,
                user_id=ix_user,
                timestamp=_parse_datetime(ix_date, path, f"{key}.date"),
                kind=kind,  # type: ignore[arg-type]
                text=ix_text,
            ))

    # Sort by (timestamp, interaction_id) for deterministic ordering
    interactions.sort(key=lambda x: (x.timestamp, x.interaction_id))

    case = EventCase(
        case_id=root_id,
        source_dataset="CHECKED",
        root=root,
        interactions=tuple(interactions),
        metadata={"label": label} if label else {},
    )
    case.validate()
    return case


def _count_empty_text(interactions: tuple) -> Tuple[int, int]:
    """Count empty-text comments and reposts."""
    empty_comments = sum(
        1 for ix in interactions
        if ix.kind == "comment" and ix.text == ""
    )
    empty_reposts = sum(
        1 for ix in interactions
        if ix.kind == "repost" and ix.text == ""
    )
    return empty_comments, empty_reposts


def iter_checked_cases(
    dataset_root: Path,
    label: str | None = None,
    report: bool = False,
) -> Iterator[EventCase] | Tuple[Iterator[EventCase], DatasetLoadReport]:
    """Iterate over all CHECKED JSON files in *dataset_root*.

    Args:
        dataset_root: Directory containing CHECKED JSON files (may be
            nested in label subdirectories).
        label: If provided, only yield cases matching this label
            ("fake" or "real").
        report: If True, returns (iterator, DatasetLoadReport) tuple.
            The iterator must be consumed before the report is complete.

    Yields:
        Validated EventCase instances (when report=False).

    Returns:
        When report=True: (iterator, DatasetLoadReport) tuple.
        Otherwise: a plain iterator of EventCase.
    """
    dataset_root = Path(dataset_root)
    load_report = DatasetLoadReport()

    def _generate():
        if not dataset_root.is_dir():
            return

        for path in sorted(dataset_root.rglob("*.json")):
            load_report.scanned_files += 1
            try:
                case = load_checked_case(path)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                load_report.failed_files += 1
                failure = DatasetLoadFailure(
                    file_name=path.name,
                    reason=str(exc),
                    error_type=type(exc).__name__,
                )
                # Classify error type for aggregated stats
                msg = str(exc).lower()
                if "timestamp" in msg or "datetime" in msg or "parse" in msg:
                    load_report.timestamp_errors += 1
                elif "missing" in msg or "empty field" in msg.lower():
                    load_report.field_missing_errors += 1
                load_report.failures.append(failure)
                continue

            # Track empty-text interactions
            ec, er = _count_empty_text(case.interactions)
            load_report.empty_text_comments += ec
            load_report.empty_text_reposts += er

            if label is not None and case.root.label != label:
                continue

            load_report.loaded_cases += 1
            yield case

    gen = _generate()

    if report:
        return gen, load_report
    return gen
