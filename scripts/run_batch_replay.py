"""
Batch default-parameter replay for CHECKED matched pairs.

Runs broadcast and cumulative_interaction modes on all 20
selected cases, 5 seeds each, 24-hour steps, default parameters.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics_simulation.config import default_params
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay

# ── Configuration ──
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "experiments" / "checked_pilot_matched_20.yaml"
DATA_ROOT = Path("data/raw/CHECKED/dataset")
OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "replay"
SEEDS = (11, 23, 37, 53, 71)
STEP_HOURS = 24.0
TAIL_STEPS = 4

MODES = [
    ("broadcast", ReplayNetworkMode.BROADCAST),
    ("cumulative_interaction", ReplayNetworkMode.CUMULATIVE_INTERACTION),
]


def case_path(label: str, case_id: str) -> Path:
    return DATA_ROOT / f"{label}_news" / f"{case_id}.json"


def run_one(case_id: str, label: str, mode_name: str, mode: ReplayNetworkMode):
    """Run replay for one case and return result dict."""
    path = case_path(label, case_id)
    case = load_checked_case(path)

    config = ReplayConfig(
        step_hours=STEP_HOURS,
        tail_steps=TAIL_STEPS,
        network_mode=mode,
        seeds=SEEDS,
    )
    params = default_params()

    t0 = time.perf_counter()
    result = run_replay(case, params, config)
    elapsed = time.perf_counter() - t0

    out = result.to_dict()
    out["elapsed_seconds"] = round(elapsed, 1)
    out["label"] = label
    out["mode"] = mode_name

    return out


def main():
    print(f"Loading config: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    pairs = cfg["primary_pairs"]
    print(f"Found {len(pairs)} pairs, {len(MODES)} modes each = "
          f"{len(pairs) * 2 * len(MODES)} total runs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for mode_name, mode in MODES:
        print(f"\n{'='*60}")
        print(f"  MODE: {mode_name}")
        print(f"{'='*60}")

        for pair in pairs:
            pid = pair["pair_id"]
            for label_key in ["fake", "real"]:
                entry = pair[label_key]
                case_id = entry["case_id"]
                label = "fake" if label_key == "fake" else "real"

                out_name = f"{pid}_{mode_name}_{label}_{case_id[:12]}.json"
                out_path = OUT_DIR / out_name

                if out_path.exists():
                    print(f"  [{pid}] {label} {case_id[:16]}... SKIP (exists)")
                    with open(out_path, "r") as fh:
                        summary = json.load(fh)
                    all_summaries.append({
                        "pair_id": pid,
                        "label": label,
                        "mode": mode_name,
                        "case_id": case_id,
                        "users": entry["users"],
                        "interactions": entry["interactions"],
                        "elapsed_s": summary.get("elapsed_seconds", 0),
                        "status": "cached",
                    })
                    continue

                try:
                    print(f"  [{pid}] {label} {case_id[:16]}...", end=" ", flush=True)
                    result = run_one(case_id, label, mode_name, mode)

                    with open(out_path, "w", encoding="utf-8") as fh:
                        json.dump(result, fh, indent=2, default=str)

                    print(f"DONE ({result['elapsed_seconds']:.1f}s, "
                          f"N={result['node_count']}, "
                          f"steps={len(result.get('observed',{}).get('steps',[]))})")

                    all_summaries.append({
                        "pair_id": pid,
                        "label": label,
                        "mode": mode_name,
                        "case_id": case_id,
                        "users": entry["users"],
                        "interactions": entry["interactions"],
                        "elapsed_s": result["elapsed_seconds"],
                        "status": "ok",
                    })

                except Exception as exc:
                    print(f"FAILED: {exc}")
                    all_summaries.append({
                        "pair_id": pid,
                        "label": label,
                        "mode": mode_name,
                        "case_id": case_id,
                        "users": entry["users"],
                        "interactions": entry["interactions"],
                        "elapsed_s": 0,
                        "status": f"error: {exc}",
                    })

    # ── Write summary ──
    summary_path = OUT_DIR / "replay_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(all_summaries, fh, indent=2)
    print(f"\nSummary: {summary_path}")

    ok = sum(1 for s in all_summaries if s["status"] in ("ok", "cached"))
    err = sum(1 for s in all_summaries if "error" in s["status"])
    print(f"  Completed: {ok}  |  Failed: {err}  |  Total: {len(all_summaries)}")


if __name__ == "__main__":
    main()
