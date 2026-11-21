#!/usr/bin/env python3
"""
Audit tau3-bench (v1.0.1) Retail Tasks for Issue #384 & Freeze Train/Eval Split.

Checks every retail task definition for non-empty evaluation criteria / reward_basis,
excludes any task susceptible to trivial false passes (Issue #384), and produces
an immutable stratified Train/Eval split in `configs/retail_task_split.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def has_valid_grading_criteria(task: dict[str, Any]) -> tuple[bool, str]:
    """Verify a task has non-empty deterministic grading assertions (Issue #384 check)."""
    eval_crit = task.get("evaluation_criteria") or {}
    reward_basis = eval_crit.get("reward_basis") or task.get("reward_basis")
    actions = eval_crit.get("actions") or task.get("actions")
    nl_assertions = eval_crit.get("nl_assertions") or eval_crit.get("communicate_info")

    if isinstance(reward_basis, list) and len(reward_basis) == 0:
        return False, "empty_reward_basis_list"
    if not actions and not nl_assertions and not reward_basis:
        return False, "no_actions_or_assertions"
    return True, "valid"


def audit_and_split(
    tasks_path: Path,
    eval_size: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    raw_bytes = tasks_path.read_bytes()
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    tasks: list[dict[str, Any]] = json.loads(raw_bytes.decode("utf-8"))

    valid_ids: list[str] = []
    excluded: list[dict[str, str]] = []

    for idx, t in enumerate(tasks):
        tid = str(t.get("id", idx))
        ok, reason = has_valid_grading_criteria(t)
        if ok:
            valid_ids.append(tid)
        else:
            excluded.append({"task_id": tid, "reason": reason})

    rng = random.Random(seed)
    shuffled = list(valid_ids)
    rng.shuffle(shuffled)

    eval_ids = sorted(shuffled[:eval_size], key=lambda x: int(x) if x.isdigit() else x)
    train_ids = sorted(shuffled[eval_size:], key=lambda x: int(x) if x.isdigit() else x)

    return {
        "benchmark": "sierra-research/tau2-bench",
        "pinned_release": "v1.0.1",
        "domain": "retail",
        "source_tasks_sha256": sha256_hash,
        "random_seed": seed,
        "counts": {
            "total_raw_tasks": len(tasks),
            "valid_tasks": len(valid_ids),
            "excluded_issue_384": len(excluded),
            "eval_split_count": len(eval_ids),
            "train_split_count": len(train_ids),
        },
        "excluded_tasks": excluded,
        "eval_task_ids": eval_ids,
        "train_task_ids": train_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit tau3-bench retail tasks and freeze train/eval split.")
    parser.add_argument("--tasks-json", required=True, help="Path to tau2-bench retail tasks.json")
    parser.add_argument("--eval-size", type=int, default=50, help="Number of held-out evaluation tasks")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument("--out", default="configs/retail_task_split.json", help="Output split config path")
    args = parser.parse_args()

    report = audit_and_split(Path(args.tasks_json), eval_size=args.eval_size, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    print(f"[DONE] Frozen split written to {out_path}")


if __name__ == "__main__":
    main()
