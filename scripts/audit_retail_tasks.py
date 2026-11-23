#!/usr/bin/env python3
"""
[CBA-8] Audit tau3-bench (v1.0.1) Retail Tasks & Freeze Leak-Free Persona-Grouped Split.

Audits all 114 retail tasks for:
  1. Missing evaluation_criteria or empty reward_basis
  2. True v1.0.1 vacuous-pass defect: zero expected WRITE tool calls AND zero graded
     nl_assertions (so an empty do-nothing trajectory scores 1.0 under EvaluationType.ALL)

Constructs a deterministic persona-grouped stratified split over the 108 non-vacuous
tasks (zero shared customer `known_info` personas between Train and Eval), balanced
across `has_nl_assertions` and `H_oracle` quartiles.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
)


def _h_oracle_bucket(h_tokens: int | None) -> str:
    if h_tokens is None:
        return "unknown"
    if h_tokens < 1000:
        return "<1k"
    if h_tokens < 2000:
        return "1k-2k"
    if h_tokens < 4000:
        return "2k-4k"
    return ">=4k"


def audit_task(task: dict[str, Any], write_tools: set[str] | frozenset[str]) -> tuple[bool, str, dict[str, Any]]:
    """Audit a single task for empty grading criteria or vacuous-pass behavior."""
    eval_crit = task.get("evaluation_criteria")
    if eval_crit is None:
        return False, "missing_evaluation_criteria", {}

    reward_basis = eval_crit.get("reward_basis") or []
    if len(reward_basis) == 0:
        return False, "empty_reward_basis", {}

    actions = eval_crit.get("actions") or []
    nl_assertions = eval_crit.get("nl_assertions") or []
    communicate_info = eval_crit.get("communicate_info") or []

    has_write = any(a.get("name") in write_tools for a in actions)
    has_graded_nl = ("NL_ASSERTION" in reward_basis) and bool(nl_assertions)
    has_comm = bool(communicate_info)
    is_ci_only = has_comm and not has_graded_nl

    flags = {
        "has_write": has_write,
        "has_nl_assertions": has_graded_nl,
        "has_communicate_info": has_comm,
        "is_ci_only": is_ci_only,
        "reward_basis": list(reward_basis),
        "num_expected_actions": len(actions),
        "uses_transfer": any(a.get("name") == "transfer_to_human_agents" for a in actions),
    }

    if not has_write and not has_graded_nl:
        reason = "vacuous_pass_transfer_only" if flags["uses_transfer"] else "vacuous_pass_no_write_no_nl"
        return False, reason, flags

    return True, "valid", flags


def _solve_exact_subset_indices(sizes: list[int], target: int, prefer_mask: list[bool]) -> list[int] | None:
    """
    Find a subset of indices in `sizes` whose sum is exactly `target`.
    Prefers selecting indices marked `True` in `prefer_mask` (from initial stratified pass).
    """
    # DP: map sum -> list of chosen indices; break ties by maximizing count of preferred items
    dp: dict[int, tuple[int, list[int]]] = {0: (0, [])}
    for idx, sz in enumerate(sizes):
        pref_gain = 1 if prefer_mask[idx] else 0
        next_dp = dict(dp)
        for s, (score, chosen) in dp.items():
            ns = s + sz
            if ns > target:
                continue
            cand_score = score + pref_gain
            if ns not in next_dp or cand_score > next_dp[ns][0]:
                next_dp[ns] = (cand_score, chosen + [idx])
        dp = next_dp
    return dp[target][1] if target in dp else None


def audit_and_split(
    tasks_path: Path,
    split_json_path: Path | None = None,
    tool_metadata_path: Path | None = None,
    oracle_profile_path: Path | None = None,
    eval_size: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    raw_bytes = tasks_path.read_bytes()
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    tasks: list[dict[str, Any]] = json.loads(raw_bytes.decode("utf-8"))

    write_tools = set(DEFAULT_WRITE_TOOLS)
    if tool_metadata_path and tool_metadata_path.exists():
        meta = json.loads(tool_metadata_path.read_text(encoding="utf-8"))
        if meta.get("write_tools"):
            write_tools = set(meta["write_tools"])

    official_test_ids: set[str] = set()
    official_train_ids: set[str] = set()
    if split_json_path and split_json_path.exists():
        sp = json.loads(split_json_path.read_text(encoding="utf-8"))
        official_test_ids = set(map(str, sp.get("test", [])))
        official_train_ids = set(map(str, sp.get("train", [])))

    oracle_by_id: dict[str, dict[str, Any]] = {}
    if oracle_profile_path and oracle_profile_path.exists():
        for row in json.loads(oracle_profile_path.read_text(encoding="utf-8")):
            oracle_by_id[str(row["id"])] = row

    valid_tasks: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    task_metadata: dict[str, dict[str, Any]] = {}

    for idx, t in enumerate(tasks):
        tid = str(t.get("id", idx))
        known_info = (
            t.get("user_scenario", {})
            .get("instructions", {})
            .get("known_info", f"unknown_{tid}")
            .strip()
        )
        official_split = (
            "test" if tid in official_test_ids else ("train" if tid in official_train_ids else "unknown")
        )
        h_oracle = oracle_by_id.get(tid, {}).get("H_oracle")
        s_min = oracle_by_id.get(tid, {}).get("S_min")

        ok, reason, flags = audit_task(t, write_tools)
        meta_entry = {
            "task_id": tid,
            "valid": ok,
            "audit_reason": reason,
            "known_info": known_info,
            "official_split": official_split,
            "H_oracle": h_oracle,
            "H_bucket": _h_oracle_bucket(h_oracle),
            "S_min": s_min,
            **flags,
        }
        task_metadata[tid] = meta_entry

        if ok:
            valid_tasks.append(t)
        else:
            excluded.append(
                {
                    "task_id": tid,
                    "reason": reason,
                    "official_split": official_split,
                    "uses_transfer": flags.get("uses_transfer", False),
                }
            )

    # Group valid tasks by exact customer persona (`known_info`)
    persona_groups_map: dict[str, list[str]] = collections.defaultdict(list)
    for t in valid_tasks:
        tid = str(t["id"])
        persona_groups_map[task_metadata[tid]["known_info"]].append(tid)

    # Deterministic persona IDs ordered by smallest numeric task id in the group
    sorted_personas = sorted(
        persona_groups_map.items(),
        key=lambda kv: min(int(x) if x.isdigit() else 0 for x in kv[1]),
    )
    persona_records: list[dict[str, Any]] = []
    for p_idx, (known_info, tids) in enumerate(sorted_personas):
        pid = f"persona_{p_idx:02d}"
        sorted_tids = sorted(tids, key=lambda x: int(x) if x.isdigit() else x)
        for tid in sorted_tids:
            task_metadata[tid]["persona_id"] = pid
        has_nl = any(task_metadata[tid]["has_nl_assertions"] for tid in sorted_tids)
        h_vals = [task_metadata[tid]["H_oracle"] for tid in sorted_tids if task_metadata[tid]["H_oracle"] is not None]
        mean_h = round(sum(h_vals) / len(h_vals)) if h_vals else None
        persona_records.append(
            {
                "persona_id": pid,
                "known_info": known_info,
                "task_ids": sorted_tids,
                "size": len(sorted_tids),
                "has_nl": has_nl,
                "h_bucket": _h_oracle_bucket(mean_h),
            }
        )

    # Stratified seeded allocation across (has_nl, h_bucket)
    rng = random.Random(seed)
    strata: dict[tuple[bool, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in persona_records:
        strata[(rec["has_nl"], rec["h_bucket"])].append(rec)

    ordered_groups: list[dict[str, Any]] = []
    for key in sorted(strata.keys()):
        grp_list = list(strata[key])
        rng.shuffle(grp_list)
        strata[key] = grp_list

    # Proportional quota pass per stratum, then exact subset match preserving stratified preference
    total_valid = len(valid_tasks)
    target_ratio = eval_size / total_valid
    prefer_eval_pids: set[str] = set()
    for key, grp_list in strata.items():
        stratum_tasks = sum(g["size"] for g in grp_list)
        stratum_target = round(stratum_tasks * target_ratio)
        acc = 0
        for g in grp_list:
            if acc + g["size"] <= stratum_target or (acc < stratum_target and abs(acc + g["size"] - stratum_target) < abs(acc - stratum_target)):
                prefer_eval_pids.add(g["persona_id"])
                acc += g["size"]
        ordered_groups.extend(grp_list)

    rng.shuffle(ordered_groups)
    sizes = [g["size"] for g in ordered_groups]
    prefer_mask = [g["persona_id"] in prefer_eval_pids for g in ordered_groups]
    chosen_indices = _solve_exact_subset_indices(sizes, eval_size, prefer_mask)
    if chosen_indices is None:
        raise RuntimeError(f"Could not partition {len(ordered_groups)} persona groups into exact eval_size={eval_size}.")

    eval_group_set = {ordered_groups[i]["persona_id"] for i in chosen_indices}
    eval_ids: list[str] = []
    train_ids: list[str] = []
    eval_personas: set[str] = set()
    train_personas: set[str] = set()

    for g in persona_records:
        if g["persona_id"] in eval_group_set:
            eval_ids.extend(g["task_ids"])
            eval_personas.add(g["persona_id"])
        else:
            train_ids.extend(g["task_ids"])
            train_personas.add(g["persona_id"])

    eval_ids = sorted(eval_ids, key=lambda x: int(x) if x.isdigit() else x)
    train_ids = sorted(train_ids, key=lambda x: int(x) if x.isdigit() else x)
    excluded_ids = {e["task_id"] for e in excluded}

    # Strict leakage & integrity assertions
    assert len(eval_ids) == eval_size, f"Expected {eval_size} eval tasks, got {len(eval_ids)}"
    assert len(eval_ids) + len(train_ids) == len(valid_tasks)
    assert set(eval_ids).isdisjoint(set(train_ids)), "Task overlap between eval and train!"
    assert eval_personas.isdisjoint(train_personas), "Persona overlap between eval and train!"
    assert excluded_ids.isdisjoint(set(eval_ids) | set(train_ids)), "Vacuous task leaked into split!"

    def _split_stats(tids: list[str]) -> dict[str, Any]:
        buckets = collections.Counter(task_metadata[tid]["H_bucket"] for tid in tids)
        return {
            "count": len(tids),
            "has_nl_assertions": sum(1 for tid in tids if task_metadata[tid]["has_nl_assertions"]),
            "has_communicate_info": sum(1 for tid in tids if task_metadata[tid]["has_communicate_info"]),
            "is_ci_only": sum(1 for tid in tids if task_metadata[tid]["is_ci_only"]),
            "multi_write": sum(1 for tid in tids if task_metadata[tid]["num_expected_actions"] > 0 and task_metadata[tid]["has_write"]),
            "official_test_overlap": sum(1 for tid in tids if task_metadata[tid]["official_split"] == "test"),
            "H_oracle_buckets": dict(sorted(buckets.items())),
        }

    return {
        "benchmark": "sierra-research/tau2-bench",
        "pinned_release": "v1.0.1",
        "pinned_commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992",
        "domain": "retail",
        "source_tasks_sha256": sha256_hash,
        "random_seed": seed,
        "split_strategy": "persona_grouped_stratified",
        "counts": {
            "total_raw_tasks": len(tasks),
            "valid_tasks": len(valid_tasks),
            "excluded_vacuous_pass": len(excluded),
            "eval_split_count": len(eval_ids),
            "train_split_count": len(train_ids),
            "total_valid_persona_groups": len(persona_records),
            "eval_persona_groups": len(eval_personas),
            "train_persona_groups": len(train_personas),
            "persona_overlap_count": len(eval_personas & train_personas),
        },
        "excluded_tasks": excluded,
        "eval_task_ids": eval_ids,
        "train_task_ids": train_ids,
        "split_balance_stats": {
            "eval": _split_stats(eval_ids),
            "train": _split_stats(train_ids),
        },
        "task_metadata": task_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit tau3-bench retail tasks and freeze leak-free split.")
    parser.add_argument("--tasks-json", required=True, help="Path to tau2-bench retail tasks.json")
    parser.add_argument("--split-json", default=None, help="Optional path to official split_tasks.json")
    parser.add_argument("--tool-metadata", default=None, help="Optional path to configs/retail_tool_metadata.json")
    parser.add_argument("--oracle-profile", default=None, help="Optional path to oracle_profile.json")
    parser.add_argument("--eval-size", type=int, default=50, help="Number of held-out evaluation tasks")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument("--out", default="configs/retail_task_split.json", help="Output split config path")
    args = parser.parse_args()

    report = audit_and_split(
        tasks_path=Path(args.tasks_json),
        split_json_path=Path(args.split_json) if args.split_json else None,
        tool_metadata_path=Path(args.tool_metadata) if args.tool_metadata else None,
        oracle_profile_path=Path(args.oracle_profile) if args.oracle_profile else None,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "split_balance_stats": report["split_balance_stats"]}, indent=2))
    print(f"[DONE] Frozen split written to {out_path}")


if __name__ == "__main__":
    main()
