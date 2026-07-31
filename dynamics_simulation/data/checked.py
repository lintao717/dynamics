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
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
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


def _pick(d: dict, keys: tuple[str, ...], field_name: str, path: Path) -> str:
    """Pick the first matching key from *keys* in *d*, raising on missing."""
    for k in keys:
        if k in d:
            val = d[k]
            if isinstance(val, str) and val.strip():
                return val.strip()
    raise ValueError(
        f"Missing or empty field '{field_name}' (tried {keys}) in {path}"
    )


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
    root_id = _pick(raw, ID_KEYS, "root.id", path)
    root_user = _pick(raw, USER_KEYS, "root.user_id", path)
    root_date = _pick(raw, DATE_KEYS, "root.date", path)
    root_text = _pick(raw, TEXT_KEYS, "root.text", path)
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
            ix_id = _pick(item, ID_KEYS, f"{key}.id", path)
            ix_user = _pick(item, USER_KEYS, f"{key}.user_id", path)
            ix_date = _pick(item, DATE_KEYS, f"{key}.date", path)
            ix_text = _pick(item, TEXT_KEYS, f"{key}.text", path)

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


def iter_checked_cases(
    dataset_root: Path,
    label: str | None = None,
) -> Iterator[EventCase]:
    """Iterate over all CHECKED JSON files in *dataset_root*.

    Args:
        dataset_root: Directory containing CHECKED JSON files (may be
            nested in label subdirectories).
        label: If provided, only yield cases matching this label
            ("fake" or "real").

    Yields:
        Validated EventCase instances.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        return

    for path in sorted(dataset_root.rglob("*.json")):
        try:
            case = load_checked_case(path)
        except (ValueError, KeyError, json.JSONDecodeError):
            # Skip malformed files in bulk iteration; individual
            # errors are caught by load_checked_case tests.
            continue
        if label is not None and case.root.label != label:
            continue
        yield case
