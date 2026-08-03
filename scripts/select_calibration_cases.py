"""
CHECKED matched-pair case selection for V1.2.1 calibration.

Pipeline:
  1. Load all CHECKED cases with audit report
  2. Compute per-case statistics (D95, peak, share ratios, etc.)
  3. Apply hard filters
  4. Quality-score and rank candidates
  5. Match fake-real pairs via log-scaled Euclidean distance
  6. Output YAML config, CSV summary, and audit report
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# ── Add project root ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics_simulation.data.checked import iter_checked_cases

# ── Constants ──
SEED = 20260803
STEP_HOURS = 24.0
OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "checked_audit"
CFG_DIR = Path(__file__).resolve().parent.parent / "configs" / "experiments"
DATA_ROOT = Path("data/raw/CHECKED/dataset")

# Hard filters
MIN_INTERACTIONS = 100
MIN_USERS = 100
MAX_USERS = 1000
MIN_DURATION_H = 96.0  # 4 days for 24h steps
MIN_COMMENTS = 20
MIN_REPOSTS = 20
MAX_USER_SHARE = 0.5  # no single user >50% of interactions
MIN_DATA_STEPS = 4     # 24h window → >=4 data steps


@dataclass
class CaseStats:
    """Per-case statistics for selection and matching."""
    case_id: str
    file_path: str
    file_sha256: str
    label: str
    n_users: int
    n_interactions: int
    n_comments: int
    n_reposts: int
    duration_h: float
    d95_h: float              # time to 95% of interactions
    peak_24h: int             # max interactions in any 24h window
    first_24h_share: float    # fraction of interactions in first 24h
    comment_share: float      # comments / total interactions
    max_user_share: float     # max share from a single user
    max_gap_h: float          # longest gap between consecutive interactions
    n_active_windows: int     # number of 24h windows with ≥1 interaction
    truncated: bool           # would be truncated at 1000 nodes
    quality_score: float = 0.0
    rank: int = 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compute_d95(case) -> float:
    """Time (hours) until 95% of interactions have occurred."""
    if not case.interactions:
        return 0.0
    t0 = case.root.timestamp
    total = len(case.interactions)
    target = int(total * 0.95)
    t95 = case.interactions[min(target, total - 1)].timestamp
    return (t95 - t0).total_seconds() / 3600.0


def _compute_peak_24h(case, step_h: float = 24.0) -> int:
    """Max interactions in any 24-hour window."""
    if not case.interactions:
        return 0
    t0 = case.root.timestamp
    bins = {}
    for ix in case.interactions:
        step = int((ix.timestamp - t0).total_seconds() / 3600 / step_h)
        bins[step] = bins.get(step, 0) + 1
    return max(bins.values()) if bins else 0


def _compute_first_24h_share(case) -> float:
    """Fraction of interactions in first 24 hours."""
    if not case.interactions:
        return 0.0
    t0 = case.root.timestamp
    total = len(case.interactions)
    in_24h = sum(
        1 for ix in case.interactions
        if (ix.timestamp - t0).total_seconds() <= 86400
    )
    return in_24h / total


def _compute_max_user_share(case) -> float:
    """Maximum fraction of interactions from a single user."""
    if not case.interactions:
        return 0.0
    from collections import Counter
    counts = Counter(ix.user_id for ix in case.interactions)
    return max(counts.values()) / len(case.interactions)


def _compute_max_gap_h(case) -> float:
    """Longest gap (hours) between consecutive interactions."""
    if len(case.interactions) < 2:
        return 0.0
    gaps = []
    timestamps = [case.root.timestamp] + [
        ix.timestamp for ix in case.interactions
    ]
    for i in range(1, len(timestamps)):
        g = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
        gaps.append(g)
    return max(gaps)


def _compute_n_active_windows(case, step_h: float = 24.0) -> int:
    """Number of distinct 24h windows with at least 1 interaction."""
    if not case.interactions:
        return 0
    t0 = case.root.timestamp
    active = set()
    for ix in case.interactions:
        step = int((ix.timestamp - t0).total_seconds() / 3600 / step_h)
        active.add(step)
    return len(active)


def build_case_stats(case, file_path: Path) -> CaseStats:
    """Compute all per-case statistics."""
    n_comments = sum(1 for ix in case.interactions if ix.kind == "comment")
    n_reposts = sum(1 for ix in case.interactions if ix.kind == "repost")
    n_ix = len(case.interactions)
    n_users = len(case.user_ids)

    if case.interactions:
        dur = (case.interactions[-1].timestamp -
               case.root.timestamp).total_seconds() / 3600
    else:
        dur = 0.0

    d95 = _compute_d95(case)
    peak = _compute_peak_24h(case, STEP_HOURS)
    first24 = _compute_first_24h_share(case)
    cshare = n_comments / max(n_ix, 1)
    user_share = _compute_max_user_share(case)
    max_gap = _compute_max_gap_h(case)
    n_windows = _compute_n_active_windows(case, STEP_HOURS)
    truncated = n_users > MAX_USERS

    return CaseStats(
        case_id=case.case_id,
        file_path=str(file_path),
        file_sha256=_sha256(file_path),
        label=case.root.label or "unknown",
        n_users=n_users,
        n_interactions=n_ix,
        n_comments=n_comments,
        n_reposts=n_reposts,
        duration_h=dur,
        d95_h=d95,
        peak_24h=peak,
        first_24h_share=first24,
        comment_share=cshare,
        max_user_share=user_share,
        max_gap_h=max_gap,
        n_active_windows=n_windows,
        truncated=truncated,
    )


def passes_hard_filters(s: CaseStats) -> bool:
    """Check if a case passes all hard selection criteria."""
    if s.n_interactions < MIN_INTERACTIONS:
        return False
    if s.n_users < MIN_USERS:
        return False
    if s.n_users > MAX_USERS:
        return False
    if s.duration_h < MIN_DURATION_H:
        return False
    if s.n_comments < MIN_COMMENTS:
        return False
    if s.n_reposts < MIN_REPOSTS:
        return False
    if s.max_user_share > MAX_USER_SHARE:
        return False
    # 24h window needs at least 4 data steps for 70/30 split
    if int(math.ceil(s.duration_h / STEP_HOURS)) < MIN_DATA_STEPS:
        return False
    return True


def _quality_score(s: CaseStats) -> float:
    """Compute a quality score (0-1, higher = better for calibration).

    Penalises: extreme user concentration, very short/long D95 ratio,
    very few active windows, single-user dominance.
    """
    score = 1.0

    # Penalise extreme user concentration
    if s.max_user_share > 0.3:
        score -= (s.max_user_share - 0.3) * 2.0

    # Penalise too few active windows relative to duration
    expected_windows = max(1, s.duration_h / STEP_HOURS)
    window_ratio = s.n_active_windows / max(expected_windows, 1)
    if window_ratio < 0.3:
        score -= (0.3 - window_ratio) * 1.5

    # Penalise D95 too different from total duration (long tail)
    d95_ratio = s.d95_h / max(s.duration_h, 1)
    if d95_ratio < 0.3:
        score -= (0.3 - d95_ratio) * 1.0

    # Very low comment share is suspicious
    if s.comment_share < 0.05:
        score -= 0.3

    # Very low repost share is also suspicious
    if s.n_reposts / max(s.n_interactions, 1) < 0.05:
        score -= 0.3

    return max(0.0, score)


def _match_features(stats: list[CaseStats]) -> np.ndarray:
    """Compute normalised log-feature matrix for matching."""
    features = np.array([
        [
            math.log(1 + s.n_users),
            math.log(1 + s.n_interactions),
            math.log(1 + s.duration_h),
            s.comment_share,
        ]
        for s in stats
    ])
    # Standardise
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    return (features - mean) / std


def compute_matched_pairs(
    fake_candidates: list[CaseStats],
    real_candidates: list[CaseStats],
    n_pairs: int,
    n_backup: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[CaseStats, CaseStats, float]],
           list[tuple[CaseStats, CaseStats, float]]]:
    """Greedy matching: for each fake, find the closest unused real."""
    fake_feat = _match_features(fake_candidates)
    real_feat = _match_features(real_candidates)

    used_real = set()
    pairs = []

    # Sort fake by quality score (best first)
    fake_order = sorted(
        range(len(fake_candidates)),
        key=lambda i: fake_candidates[i].quality_score,
        reverse=True,
    )

    for fi in fake_order:
        if len(pairs) >= n_pairs + n_backup:
            break

        best_dist = float("inf")
        best_ri = -1
        for ri in range(len(real_candidates)):
            if ri in used_real:
                continue
            dist = float(np.linalg.norm(fake_feat[fi] - real_feat[ri]))
            if dist < best_dist:
                best_dist = dist
                best_ri = ri

        if best_ri >= 0:
            used_real.add(best_ri)
            pairs.append((
                fake_candidates[fi],
                real_candidates[best_ri],
                best_dist,
            ))

    # Select primary from best matches, rest as backup
    primary = pairs[:n_pairs]
    backup = pairs[n_pairs:n_pairs + n_backup]
    return primary, backup


def export_yaml(
    primary: list, backup: list,
    source_revision: str,
    output_path: Path,
):
    """Export matched pairs as YAML experiment config."""
    def _pair_entry(fake, real, dist, pair_id):
        return {
            "pair_id": pair_id,
            "fake": {
                "case_id": fake.case_id,
                "file_sha256": fake.file_sha256,
                "users": fake.n_users,
                "interactions": fake.n_interactions,
                "duration_hours": round(fake.duration_h, 1),
                "d95_hours": round(fake.d95_h, 1),
                "comment_share": round(fake.comment_share, 3),
                "peak_24h": fake.peak_24h,
                "quality_score": round(fake.quality_score, 3),
            },
            "real": {
                "case_id": real.case_id,
                "file_sha256": real.file_sha256,
                "users": real.n_users,
                "interactions": real.n_interactions,
                "duration_hours": round(real.duration_h, 1),
                "d95_hours": round(real.d95_h, 1),
                "comment_share": round(real.comment_share, 3),
                "peak_24h": real.peak_24h,
                "quality_score": round(real.quality_score, 3),
            },
            "matching_distance": round(dist, 4),
        }

    config = {
        "experiment_id": "checked-pilot-v1",
        "selection_seed": SEED,
        "step_hours": STEP_HOURS,
        "source_revision": source_revision,
        "selection_method": "matched_pairs_log_euclidean",
        "matching_features": [
            "log(1 + n_users)",
            "log(1 + n_interactions)",
            "log(1 + duration_hours)",
            "comment_share",
        ],
        "hard_filters": {
            "min_interactions": MIN_INTERACTIONS,
            "min_users": MIN_USERS,
            "max_users": MAX_USERS,
            "min_duration_hours": MIN_DURATION_H,
            "min_comments": MIN_COMMENTS,
            "min_reposts": MIN_REPOSTS,
            "max_user_share": MAX_USER_SHARE,
            "min_data_steps": MIN_DATA_STEPS,
            "step_hours": STEP_HOURS,
        },
        "primary_pairs": [
            _pair_entry(f, r, d, f"pair_{i+1:02d}")
            for i, (f, r, d) in enumerate(primary)
        ],
        "backup_pairs": [
            _pair_entry(f, r, d, f"backup_{i+1:02d}")
            for i, (f, r, d) in enumerate(backup)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.dump(config, fh, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=120)
    print(f"  Config: {output_path}")


def export_csv(all_stats: list[CaseStats], output_path: Path):
    """Export full case summary CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        headers = [
            "case_id", "label", "n_users", "n_interactions",
            "n_comments", "n_reposts", "duration_h", "d95_h",
            "peak_24h", "first_24h_share", "comment_share",
            "max_user_share", "max_gap_h", "n_active_windows",
            "truncated", "passes_filter", "quality_score",
        ]
        fh.write(",".join(headers) + "\n")
        for s in sorted(all_stats, key=lambda x: x.quality_score, reverse=True):
            passed = "yes" if passes_hard_filters(s) else "no"
            fh.write(
                f"{s.case_id},{s.label},{s.n_users},{s.n_interactions},"
                f"{s.n_comments},{s.n_reposts},{s.duration_h:.1f},{s.d95_h:.1f},"
                f"{s.peak_24h},{s.first_24h_share:.3f},{s.comment_share:.3f},"
                f"{s.max_user_share:.3f},{s.max_gap_h:.1f},{s.n_active_windows},"
                f"{'yes' if s.truncated else 'no'},{passed},{s.quality_score:.3f}\n"
            )
    print(f"  CSV: {output_path}")


def main():
    rng = np.random.default_rng(SEED)
    data_root = DATA_ROOT

    print("=" * 60)
    print("  CHECKED Calibration Case Selection")
    print(f"  Seed: {SEED}  |  Step: {STEP_HOURS}h  |  Users: {MIN_USERS}-{MAX_USERS}")
    print(f"  Min interactions: {MIN_INTERACTIONS}  |  Min duration: {MIN_DURATION_H}h")
    print("=" * 60)

    # ── Step 1: Load all cases and compute stats ──
    print("\n[1/4] Loading cases and computing statistics...")
    all_stats: list[CaseStats] = []

    for label in ["fake", "real"]:
        label_dir = data_root / f"{label}_news"
        cases_iter, report = iter_checked_cases(label_dir, report=True)

        for case in cases_iter:
            # Resolve file path from case_id
            file_path = label_dir / f"{case.case_id}.json"
            if not file_path.exists():
                # Try without extension
                candidates = list(label_dir.glob(f"{case.case_id}*"))
                if candidates:
                    file_path = candidates[0]
                else:
                    continue

            stats = build_case_stats(case, file_path)
            all_stats.append(stats)

        print(f"  [{label}] {report.loaded_cases} loaded, "
              f"{report.failed_files} failed, "
              f"{report.empty_text_comments} empty-cmt, "
              f"{report.empty_text_reposts} empty-rpt")

    print(f"  Total: {len(all_stats)} cases")

    # ── Step 2: Apply hard filters ──
    print(f"\n[2/4] Applying hard filters...")
    candidates = [s for s in all_stats if passes_hard_filters(s)]
    fake_candidates = [s for s in candidates if s.label == "fake"]
    real_candidates = [s for s in candidates if s.label == "real"]

    print(f"  Filtered: {len(candidates)} pass ({len(fake_candidates)} fake, "
          f"{len(real_candidates)} real)")
    print(f"  Excluded: {len(all_stats) - len(candidates)} cases")

    # Breakdown of exclusions
    excl_reasons = {
        "too_few_interactions": sum(
            1 for s in all_stats if s.n_interactions < MIN_INTERACTIONS),
        "too_few_users": sum(
            1 for s in all_stats if s.n_users < MIN_USERS),
        "too_many_users": sum(
            1 for s in all_stats if s.n_users > MAX_USERS),
        "too_short": sum(
            1 for s in all_stats if s.duration_h < MIN_DURATION_H),
        "too_few_comments": sum(
            1 for s in all_stats
            if s.n_interactions >= MIN_INTERACTIONS and s.n_comments < MIN_COMMENTS),
        "too_few_reposts": sum(
            1 for s in all_stats
            if s.n_interactions >= MIN_INTERACTIONS and s.n_reposts < MIN_REPOSTS),
        "user_concentration": sum(
            1 for s in all_stats if s.max_user_share > MAX_USER_SHARE),
        "insufficient_data_steps": sum(
            1 for s in all_stats
            if int(math.ceil(s.duration_h / STEP_HOURS)) < MIN_DATA_STEPS),
    }
    for reason, count in excl_reasons.items():
        if count > 0:
            print(f"    - {reason}: {count}")

    # ── Step 3: Quality score ──
    print(f"\n[3/4] Computing quality scores...")
    for s in fake_candidates + real_candidates:
        s.quality_score = _quality_score(s)

    fake_candidates.sort(key=lambda x: x.quality_score, reverse=True)
    real_candidates.sort(key=lambda x: x.quality_score, reverse=True)

    print(f"  Fake top-5 quality: {[round(s.quality_score, 3) for s in fake_candidates[:5]]}")
    print(f"  Real top-5 quality: {[round(s.quality_score, 3) for s in real_candidates[:5]]}")

    # ── Step 4: Matched pairs ──
    print(f"\n[4/4] Matching pairs...")
    primary, backup = compute_matched_pairs(
        fake_candidates, real_candidates,
        n_pairs=10, n_backup=5, rng=rng,
    )

    print(f"\n  Primary pairs ({len(primary)}):")
    print(f"  {'Pair':<8} {'Fake ID':<35} {'Real ID':<35} {'Dist':<8} {'F-Users':<8} {'R-Users':<8}")
    print(f"  {'-'*8} {'-'*35} {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for i, (f, r, d) in enumerate(primary):
        print(f"  pair_{i+1:02d}  {f.case_id:<35} {r.case_id:<35} "
              f"{d:<8.4f} {f.n_users:<8} {r.n_users:<8}")

    print(f"\n  Backup pairs ({len(backup)}):")
    for i, (f, r, d) in enumerate(backup):
        print(f"  backup_{i+1:02d} {f.case_id:<35} {r.case_id:<35} "
              f"{d:<8.4f} {f.n_users:<8} {r.n_users:<8}")

    # ── Export artifacts ──
    print(f"\n  Exporting artifacts...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    export_csv(all_stats, OUT_DIR / "checked_event_summary.csv")

    source_rev = "CHECKED (unknown commit — dataset release)"
    export_yaml(primary, backup, source_rev,
                CFG_DIR / "checked_pilot_matched_20.yaml")
    export_yaml(backup, [], source_rev,
                CFG_DIR / "checked_pilot_backup_10.yaml")

    # ── Quality report ──
    report_path = OUT_DIR / "checked_quality_report.json"
    report = {
        "selection_seed": SEED,
        "step_hours": STEP_HOURS,
        "total_scanned": len(all_stats),
        "total_passed_filter": len(candidates),
        "fake_passed": len(fake_candidates),
        "real_passed": len(real_candidates),
        "primary_pairs": len(primary),
        "backup_pairs": len(backup),
        "hard_filters": {
            "min_interactions": MIN_INTERACTIONS,
            "min_users": MIN_USERS,
            "max_users": MAX_USERS,
            "min_duration_hours": MIN_DURATION_H,
            "min_comments": MIN_COMMENTS,
            "min_reposts": MIN_REPOSTS,
            "max_user_share": MAX_USER_SHARE,
            "min_data_steps": MIN_DATA_STEPS,
        },
        "exclusion_counts": excl_reasons,
        "supplementary_stats": {
            "d95_available": True,
            "peak_24h_available": True,
            "first_24h_share_available": True,
            "max_user_share_available": True,
        },
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Report: {report_path}")

    print("\n" + "=" * 60)
    print("  Selection complete.")
    print(f"  10 primary pairs → {CFG_DIR / 'checked_pilot_matched_20.yaml'}")
    print(f"  5 backup pairs  → {CFG_DIR / 'checked_pilot_backup_10.yaml'}")
    print(f"  Full CSV        → {OUT_DIR / 'checked_event_summary.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
