"""CLI: inspect CHECKED/CED dataset cascades."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dynamics_simulation.data.checked import iter_checked_cases


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Inspect CHECKED/CED dataset cascades."
    )
    p.add_argument("--dataset", choices=["checked", "ced"], default="checked")
    p.add_argument("--root", type=Path, required=True,
                   help="Dataset root directory.")
    p.add_argument("--top", type=int, default=20,
                   help="Maximum cases to list.")
    p.add_argument("--label", type=str, default=None,
                   help="Filter by label (e.g. 'fake', 'real').")
    args = p.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    if args.dataset == "checked":
        cases = list(iter_checked_cases(root, label=args.label))
    else:
        print("CED inspection not yet implemented", file=sys.stderr)
        return 1

    print(f"{'case_id':30s} {'label':6s} {'root_time':20s} "
          f"{'users':>6s} {'comments':>8s} {'reposts':>7s} "
          f"{'interactions':>12s} {'dur_h':>6s}")
    print("-" * 100)

    for case in cases[:args.top]:
        n_comments = sum(1 for ix in case.interactions if ix.kind == "comment")
        n_reposts = sum(1 for ix in case.interactions if ix.kind == "repost")
        n_users = len(case.user_ids)
        root_time = case.root.timestamp.isoformat()

        if case.interactions:
            dur = (case.interactions[-1].timestamp - case.root.timestamp)
            dur_h = dur.total_seconds() / 3600.0
            first_24h = sum(
                1 for ix in case.interactions
                if (ix.timestamp - case.root.timestamp).total_seconds() <= 86400
            )
        else:
            dur_h = 0.0
            first_24h = 0

        print(
            f"{case.case_id:30s} "
            f"{case.root.label or '?':6s} "
            f"{root_time:20s} "
            f"{n_users:>6d} "
            f"{n_comments:>8d} "
            f"{n_reposts:>7d} "
            f"{len(case.interactions):>12d} "
            f"{dur_h:>6.1f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
