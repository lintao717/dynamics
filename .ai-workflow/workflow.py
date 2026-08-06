#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

STATES = [
    "DRAFT",
    "SCOUTED",
    "SPEC_FROZEN",
    "IMPLEMENTED",
    "UNDER_REVIEW",
    "CHANGES_REQUESTED",
    "FIXED",
    "ACCEPTED",
    "CLOSED",
    "BLOCKED",
]

TRANSITIONS = {
    "DRAFT": {"SCOUTED", "BLOCKED"},
    "SCOUTED": {"SPEC_FROZEN", "BLOCKED"},
    "SPEC_FROZEN": {"IMPLEMENTED", "BLOCKED"},
    "IMPLEMENTED": {"UNDER_REVIEW", "BLOCKED"},
    "UNDER_REVIEW": {"CHANGES_REQUESTED", "ACCEPTED", "BLOCKED"},
    "CHANGES_REQUESTED": {"FIXED", "BLOCKED"},
    "FIXED": {"UNDER_REVIEW", "ACCEPTED", "BLOCKED"},
    "ACCEPTED": {"CLOSED"},
    "BLOCKED": {"SCOUTED", "SPEC_FROZEN", "IMPLEMENTED", "CHANGES_REQUESTED", "FIXED"},
    "CLOSED": set(),
}

REQUIRED_BY_GATE = {
    "scout-start": ["00_TASK_BRIEF.md"],
    "scout-complete": ["00_TASK_BRIEF.md", "01_REPO_FACTS.md"],
    "implement-start": ["00_TASK_BRIEF.md", "01_REPO_FACTS.md", "02_IMPLEMENTATION_SPEC.md"],
    "implement-complete": ["03_IMPLEMENTATION_MAP.md", "04_IMPLEMENTATION_REPORT.md"],
    "review": ["02_IMPLEMENTATION_SPEC.md", "03_IMPLEMENTATION_MAP.md", "04_IMPLEMENTATION_REPORT.md"],
    "fix": ["05_REVIEW_REPORT.md"],
    "close": ["07_FINAL_ACCEPTANCE.md"],
}


def run(cmd: list[str], cwd: Path, check: bool = True) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def repo_root() -> Path:
    try:
        return Path(run(["git", "rev-parse", "--show-toplevel"], Path.cwd())).resolve()
    except Exception:
        return Path.cwd().resolve()


def load_config(root: Path) -> dict:
    path = root / ".ai-workflow" / "config.json"
    if not path.exists():
        raise SystemExit(f"Workflow config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def work_dir(root: Path, config: dict, item_id: str) -> Path:
    base = root / config["work_items_dir"]
    exact = base / item_id
    if exact.exists():
        return exact
    matches = list(base.glob(f"{item_id}*")) if base.exists() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"Work item not found: {item_id}")
    raise SystemExit(f"Work item ID is ambiguous: {item_id}\n" + "\n".join(str(p) for p in matches))


def git_value(root: Path, *args: str, default: str = "unknown") -> str:
    try:
        value = run(["git", *args], root)
        return value or default
    except Exception:
        return default


def safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", value)
    return value.strip("-") or "task"


def next_id(base: Path, slug: str) -> str:
    day = dt.datetime.now().strftime("%Y%m%d")
    existing = list(base.glob(f"WI-{day}-*")) if base.exists() else []
    nums = []
    for p in existing:
        m = re.match(rf"WI-{day}-(\d{{3}})", p.name)
        if m:
            nums.append(int(m.group(1)))
    return f"WI-{day}-{max(nums, default=0)+1:03d}-{safe_slug(slug)}"


def render_template(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def templates_dir(root: Path) -> Path:
    return root / ".ai-workflow" / "templates"


def read_status(wd: Path) -> dict:
    path = wd / "status.json"
    if not path.exists():
        raise SystemExit(f"status.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(wd: Path, data: dict) -> None:
    data["updated_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    (wd / "status.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command_new(args: argparse.Namespace, root: Path, config: dict) -> None:
    base = root / config["work_items_dir"]
    base.mkdir(parents=True, exist_ok=True)
    item_id = next_id(base, args.slug or args.title)

    if args.create_branch:
        if config.get("require_clean_tree_for_new_branch", True):
            dirty = git_value(root, "status", "--porcelain", default="")
            if dirty:
                raise SystemExit("Branch was not created because the tree is dirty. Commit or stash existing changes first.")
        branch_name = f"feature/{item_id}"
        run(["git", "switch", "-c", branch_name], root)

    wd = base / item_id
    wd.mkdir(parents=True)
    (wd / "_handoff").mkdir()

    base_commit = git_value(root, "rev-parse", "HEAD")
    branch = git_value(root, "branch", "--show-current")
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    values = {
        "ID": item_id,
        "TITLE": args.title,
        "BASE_COMMIT": base_commit,
        "BRANCH": branch,
        "CREATED_AT": now,
    }

    for template in templates_dir(root).glob("*.md"):
        content = render_template(template.read_text(encoding="utf-8"), values)
        (wd / template.name).write_text(content, encoding="utf-8")

    status = {
        "id": item_id,
        "title": args.title,
        "state": "DRAFT",
        "base_commit": base_commit,
        "created_branch": branch,
        "current_head": base_commit,
        "review_round": 0,
        "max_review_rounds": config.get("max_review_rounds", 2),
        "blockers": [],
        "created_at": now,
        "updated_at": now,
    }
    write_status(wd, status)

    print(item_id)
    print(f"Path: {wd}")
    print(f"Next: edit {wd / '00_TASK_BRIEF.md'} and remove WORKFLOW_PLACEHOLDER")
    print(f"Then run in Claude Code: /wf-scout {item_id}")


def check_files(wd: Path, names: Iterable[str]) -> list[str]:
    missing = []
    for name in names:
        path = wd / name
        if not path.exists():
            missing.append(name)
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content or "WORKFLOW_PLACEHOLDER" in content:
            missing.append(name)
    return missing


def command_check(args: argparse.Namespace, root: Path, config: dict) -> None:
    wd = work_dir(root, config, args.id)
    required = REQUIRED_BY_GATE.get(args.gate)
    if required is None:
        raise SystemExit(f"Unknown gate: {args.gate}")
    missing = check_files(wd, required)
    if missing:
        print("GATE FAIL")
        for name in missing:
            print(f"- missing or empty: {name}")
        raise SystemExit(2)
    print(f"GATE PASS: {args.gate}")


def command_mark(args: argparse.Namespace, root: Path, config: dict) -> None:
    if args.state not in STATES:
        raise SystemExit(f"Invalid state: {args.state}")
    wd = work_dir(root, config, args.id)
    status = read_status(wd)
    current = status["state"]
    if args.state != current and args.state not in TRANSITIONS.get(current, set()) and not args.force:
        raise SystemExit(f"Invalid transition: {current} -> {args.state}. Use --force only for manual recovery.")
    status["state"] = args.state
    status["current_head"] = git_value(root, "rev-parse", "HEAD")
    if args.state == "UNDER_REVIEW":
        status["review_round"] = int(status.get("review_round", 0)) + 1
    write_status(wd, status)
    print(f"{args.id}: {current} -> {args.state}")


def section_file(path: Path) -> str:
    if not path.exists():
        return f"> Missing file: {path.name}\n"
    content = path.read_text(encoding="utf-8", errors="replace")
    return f"\n---\n\n## FILE: {path.name}\n\n{content.rstrip()}\n"


def git_section(root: Path, title: str, args: list[str]) -> str:
    try:
        content = run(["git", *args], root, check=False)
    except Exception as exc:
        content = f"Unable to run git command: {exc}"
    return f"\n---\n\n## {title}\n\n```text\n{content}\n```\n"


def write_handoff(wd: Path, name: str, content: str) -> Path:
    out = wd / "_handoff" / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out


def command_export_plan(args: argparse.Namespace, root: Path, config: dict) -> None:
    wd = work_dir(root, config, args.id)
    missing = check_files(wd, REQUIRED_BY_GATE["scout-complete"])
    if missing:
        raise SystemExit("Cannot export plan; missing: " + ", ".join(missing))
    status = read_status(wd)
    content = f"# GPT PLANNING INPUT — {status['id']}\n\n"
    content += "> ChatGPT command: `规划 " + status["id"] + "`\n"
    content += section_file(wd / "status.json")
    content += section_file(wd / "00_TASK_BRIEF.md")
    content += section_file(wd / "01_REPO_FACTS.md")
    out = write_handoff(wd, "PLAN_INPUT.md", content)
    print(out)


def all_diffs(root: Path, base: str) -> tuple[str, str]:
    committed = run(["git", "diff", "--find-renames", f"{base}..HEAD"], root, check=False)
    staged = run(["git", "diff", "--cached", "--find-renames"], root, check=False)
    unstaged = run(["git", "diff", "--find-renames"], root, check=False)
    full = "\n\n# COMMITTED DIFF\n" + committed + "\n\n# STAGED DIFF\n" + staged + "\n\n# UNSTAGED DIFF\n" + unstaged
    stat = run(["git", "diff", "--stat", f"{base}"], root, check=False)
    return full, stat


def command_export_review(args: argparse.Namespace, root: Path, config: dict) -> None:
    wd = work_dir(root, config, args.id)
    missing = check_files(wd, REQUIRED_BY_GATE["review"])
    if missing:
        raise SystemExit("Cannot export review; missing: " + ", ".join(missing))
    status = read_status(wd)
    full_diff, stat = all_diffs(root, status["base_commit"])
    max_chars = int(config.get("max_inline_diff_chars", 500000))
    content = f"# GPT REVIEW INPUT — {status['id']}\n\n"
    content += "> ChatGPT command: `验收 " + status["id"] + "`\n"
    for name in ["status.json", "00_TASK_BRIEF.md", "01_REPO_FACTS.md", "02_IMPLEMENTATION_SPEC.md", "03_IMPLEMENTATION_MAP.md", "04_IMPLEMENTATION_REPORT.md"]:
        content += section_file(wd / name)
    content += f"\n---\n\n## GIT DIFF STAT\n\n```text\n{stat}\n```\n"
    content += git_section(root, "GIT STATUS", ["status", "--short"])
    content += git_section(root, "RECENT COMMITS", ["log", "--oneline", "--decorate", "-10"])
    patch = wd / "_handoff" / "FULL_DIFF.patch"
    patch.write_text(full_diff, encoding="utf-8")
    if len(full_diff) <= max_chars:
        content += f"\n---\n\n## FULL DIFF\n\n```diff\n{full_diff}\n```\n"
    else:
        content += f"\n---\n\n## FULL DIFF\n\nDiff is {len(full_diff)} characters and is stored separately as `FULL_DIFF.patch`. Upload it with this file.\n"
    out = write_handoff(wd, "REVIEW_INPUT.md", content)
    print(out)
    print(patch)


def command_export_recheck(args: argparse.Namespace, root: Path, config: dict) -> None:
    wd = work_dir(root, config, args.id)
    missing = check_files(wd, ["05_REVIEW_REPORT.md", "06_FIX_REPORT.md"])
    if missing:
        raise SystemExit("Cannot export recheck; missing: " + ", ".join(missing))
    status = read_status(wd)
    full_diff, stat = all_diffs(root, status["base_commit"])
    max_chars = int(config.get("max_inline_diff_chars", 500000))
    content = f"# GPT RECHECK INPUT — {status['id']}\n\n"
    content += "> ChatGPT command: `复验 " + status["id"] + "`\n"
    for name in ["status.json", "02_IMPLEMENTATION_SPEC.md", "04_IMPLEMENTATION_REPORT.md", "05_REVIEW_REPORT.md", "06_FIX_REPORT.md"]:
        content += section_file(wd / name)
    content += f"\n---\n\n## CURRENT DIFF STAT\n\n```text\n{stat}\n```\n"
    content += git_section(root, "GIT STATUS", ["status", "--short"])
    patch = wd / "_handoff" / "FULL_DIFF.patch"
    patch.write_text(full_diff, encoding="utf-8")
    if len(full_diff) <= max_chars:
        content += f"\n---\n\n## CURRENT FULL DIFF\n\n```diff\n{full_diff}\n```\n"
    else:
        content += f"\n---\n\n## CURRENT FULL DIFF\n\nDiff is {len(full_diff)} characters and is stored separately as `FULL_DIFF.patch`. Upload it with this file.\n"
    out = write_handoff(wd, "RECHECK_INPUT.md", content)
    print(out)
    print(patch)


def command_status(args: argparse.Namespace, root: Path, config: dict) -> None:
    wd = work_dir(root, config, args.id)
    status = read_status(wd)
    status["current_head"] = git_value(root, "rev-parse", "HEAD")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print("\nFiles:")
    for name in [
        "00_TASK_BRIEF.md", "01_REPO_FACTS.md", "02_IMPLEMENTATION_SPEC.md",
        "03_IMPLEMENTATION_MAP.md", "04_IMPLEMENTATION_REPORT.md", "05_REVIEW_REPORT.md",
        "06_FIX_REPORT.md", "07_FINAL_ACCEPTANCE.md", "BACKLOG.md"
    ]:
        p = wd / name
        if not p.exists():
            state = "MISSING"
        else:
            content = p.read_text(encoding="utf-8").strip()
            state = "INCOMPLETE" if (not content or "WORKFLOW_PLACEHOLDER" in content) else "OK"
        print(f"- {name}: {state}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT-Claude Code workflow utility")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("new")
    p.add_argument("--title", required=True)
    p.add_argument("--slug")
    p.add_argument("--create-branch", action="store_true")

    p = sub.add_parser("check")
    p.add_argument("--id", required=True)
    p.add_argument("--gate", required=True, choices=sorted(REQUIRED_BY_GATE))

    p = sub.add_parser("mark")
    p.add_argument("--id", required=True)
    p.add_argument("--state", required=True, choices=STATES)
    p.add_argument("--force", action="store_true")

    for name in ["export-plan", "export-review", "export-recheck", "status"]:
        p = sub.add_parser(name)
        p.add_argument("--id", required=True)

    return parser


def main() -> int:
    root = repo_root()
    config = load_config(root)
    args = build_parser().parse_args()
    commands = {
        "new": command_new,
        "check": command_check,
        "mark": command_mark,
        "export-plan": command_export_plan,
        "export-review": command_export_review,
        "export-recheck": command_export_recheck,
        "status": command_status,
    }
    commands[args.command](args, root, config)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
