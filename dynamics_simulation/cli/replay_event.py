"""CLI: replay a single CHECKED/CED cascade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dynamics_simulation.config import default_params
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replay a single event cascade.")
    p.add_argument("--dataset", choices=["checked", "ced"], default="checked")
    p.add_argument("--case", type=Path, required=True,
                   help="Path to the case JSON file.")
    p.add_argument("--step-hours", type=float, default=1.0)
    p.add_argument("--mode", default="broadcast",
                   choices=["broadcast", "cumulative_interaction", "oracle_static"])
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path.")
    p.add_argument("--seeds", type=str, default="11,23,37,53,71",
                   help="Comma-separated seed list.")
    args = p.parse_args(argv)

    case_path = Path(args.case)
    if not case_path.is_file():
        print(f"ERROR: {case_path} not found", file=sys.stderr)
        return 1

    seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    mode = ReplayNetworkMode(args.mode)

    # Load case
    if args.dataset == "checked":
        case = load_checked_case(case_path)
    else:
        print("CED replay not yet implemented", file=sys.stderr)
        return 1

    config = ReplayConfig(
        step_hours=args.step_hours,
        tail_steps=4,
        network_mode=mode,
        seeds=seeds,
    )
    params = default_params()

    result = run_replay(case, params, config)
    d = result.to_dict()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(d, indent=2, default=str),
                               encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(json.dumps(d, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
