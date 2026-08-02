"""
CED (Chinese Rumor Dataset) compatibility adapter.

CED format:
  - Original file: {text, user: {id}, time: <unix_ts>}
  - Interactions file: [{uid, text, date|data}, ...]

CED does not reliably distinguish comment from repost, so all
interaction records use kind="interaction". Interaction IDs are
deterministic SHA-256 hashes of (case_id, uid, timestamp, text).

All raw user IDs are namespace-hashed with a "CED:" prefix before
use, per the data-governance policy. CHECKED IDs are already hashed
by the dataset authors; CED IDs are not.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dynamics_simulation.data.schema import (
    EventCase,
    InteractionRecord,
    RootPost,
)


# CED timestamps: Unix epochs are interpreted in UTC; naive strings are
# assumed to be Asia/Shanghai (Weibo platform local time), matching the
# CHECKED adapter convention.
CED_TZ = ZoneInfo("Asia/Shanghai")


def _parse_timestamp(raw) -> datetime:
    """Parse a CED timestamp: Unix epoch → UTC, date string → Asia/Shanghai → UTC."""
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    raw_str = str(raw).strip()
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]
    for fmt in formats:
        try:
            dt_naive = datetime.strptime(raw_str, fmt)
            dt_local = dt_naive.replace(tzinfo=CED_TZ)
            return dt_local.astimezone(timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse CED timestamp: {raw!r}")


def _make_interaction_id(case_id: str, uid: str, ts: datetime, text: str) -> str:
    """Generate a deterministic interaction ID from content hash."""
    payload = f"{case_id}|{uid}|{ts.isoformat()}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _hash_user_id(raw_id: str) -> str:
    """Hash a raw CED user ID with namespace prefix.

    CED user IDs are not guaranteed to be pre-hashed (unlike CHECKED).
    Per data-governance policy, all external user IDs must be hashed
    at the adapter boundary.
    """
    return hashlib.sha256(
        f"CED:{raw_id}".encode("utf-8")
    ).hexdigest()[:16]


def load_ced_case(original_path: Path, interactions_path: Path, label: str) -> EventCase:
    """Load a CED case from original + interactions JSON files.

    Args:
        original_path: Path to the original post JSON.
        interactions_path: Path to the interactions list JSON.
        label: Case label (e.g., "fake", "rumor", "real").

    Returns:
        A validated EventCase with kind="interaction" for all records.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If required fields are missing.
    """
    original_path = Path(original_path)
    interactions_path = Path(interactions_path)

    case_id = original_path.stem

    with open(original_path, "r", encoding="utf-8") as fh:
        orig = json.load(fh)

    root_text = orig.get("text", "")
    root_user_raw = orig.get("user", {}).get("id", "")
    if not root_user_raw:
        raise ValueError(
            f"Missing user.id in {original_path}"
        )
    root_user = _hash_user_id(root_user_raw)

    root_time_raw = orig.get("time")
    if root_time_raw is None:
        raise ValueError(f"Missing 'time' field in {original_path}")

    root = RootPost(
        post_id=case_id,
        user_id=root_user,
        timestamp=_parse_timestamp(root_time_raw),
        text=root_text if isinstance(root_text, str) else "",
        label=label,
        expert_analysis=None,
    )

    # Load interactions
    with open(interactions_path, "r", encoding="utf-8") as fh:
        interactions_raw = json.load(fh)

    interactions: list[InteractionRecord] = []
    for item in interactions_raw:
        uid = item.get("uid", "")
        if not uid:
            continue

        # Support both 'data' and 'date' keys
        date_raw = item.get("data") or item.get("date")
        if date_raw is None:
            continue

        text = item.get("text", "")
        if not isinstance(text, str):
            text = ""

        ts = _parse_timestamp(date_raw)
        ix_id = _make_interaction_id(case_id, uid, ts, text)

        interactions.append(InteractionRecord(
            interaction_id=ix_id,
            root_post_id=case_id,
            user_id=_hash_user_id(uid),
            timestamp=ts,
            kind="interaction",
            text=text,
        ))

    # Sort by (timestamp, interaction_id)
    interactions.sort(key=lambda x: (x.timestamp, x.interaction_id))

    case = EventCase(
        case_id=case_id,
        source_dataset="CED",
        root=root,
        interactions=tuple(interactions),
        metadata={"label": label},
    )
    case.validate()
    return case
