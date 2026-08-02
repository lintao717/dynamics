"""CLI: calibrate stage-1 parameters on a single cascade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dynamics_simulation.config import default_params
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.calibration.estimator import fit_stage1
from dynamics_simulation.calibration.split import TemporalSplit


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Calibrate stage-1 parameters on one cascade."
    )
    p.add_argument("--dataset", choices=["checked"], default="checked")
    p.add_argument("--case", type=Path, required=True,
                   help="Path to the case JSON file.")
    p.add_argument("--mode", default="broadcast",
                   choices=["broadcast", "cumulative_interaction"])
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path.")
    args = p.parse_args(argv)

    case_path = Path(args.case)
    if not case_path.is_file():
        print(f"ERROR: {case_path} not found", file=sys.stderr)
        return 1

    mode = ReplayNetworkMode(args.mode)
    case = load_checked_case(case_path)

    replay_cfg = ReplayConfig(
        step_hours=1.0, tail_steps=4,
        network_mode=mode,
        seeds=(11, 23, 37, 53, 71),
    )

    result = fit_stage1(
        case, default_params(), replay_cfg,
        train_fraction=args.train_fraction,
    )

    d = {
        "case_id": result.case_id,
        "best_vector": result.best_vector,
        "best_loss": result.best_loss,
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
        "success": result.success,
        "message": result.message,
        "n_iterations": result.n_iterations,
        "optimizer_settings": result.optimizer_settings,
        "parameter_specs": result.parameter_specs,
        "seed_tuple": list(result.seed_tuple),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(json.dumps(d, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
