#!/usr/bin/env python3
"""
[CBA-6 / CBA-8] Authoritative Token Accounting with Qwen3 Native Chat Template.

Measures exact BPE token consumption across the three competing context components:
  - S (Tool Schemas): Base retail tool definitions + optional distractor tools
  - P (Policy Context): System prompt (`AGENT_INSTRUCTION` + `<policy>` + `policy.md`)
  - H (Interaction History): Multi-turn user/assistant/tool messages

Fixes the `transformers 5.x` BatchEncoding bug (`apply_chat_template(..., tokenize=True)`
returning a dict of length 2) by rendering `tokenize=False` and encoding explicitly.
Also reports both literal `B_total` and allocatable `B_alloc (S + H)` feasibility across
the proposal's `2k (2048)`, `4k (4096)`, and `8k (8192)` budgets.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


BUDGET_LEVELS = {
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
}


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def count_chat_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Count exact serialized tokens using Qwen3's native chat template safely across transformers versions."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools if tools else None,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError(f"Expected rendered template str, got {type(rendered).__name__}")
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    count = len(token_ids)
    if messages and count <= 5:
        raise RuntimeError(
            f"Unexpectedly low token count ({count}) for non-empty messages; "
            "check chat template / tokenizer compatibility."
        )
    return count


def _summarize_distribution(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0}
    ordered = sorted(values)
    q = lambda p: ordered[min(len(ordered) - 1, int(p * len(ordered)))]
    return {
        "min": ordered[0],
        "p50": q(0.5),
        "p90": q(0.9),
        "max": ordered[-1],
        "mean": round(statistics.mean(ordered)),
    }


def compute_component_breakdown(
    tokenizer: Any,
    system_prompt_text: str,
    tools: list[dict[str, Any]],
    history_messages: list[dict[str, Any]],
    raw_policy_text: str | None = None,
    distractor_tools: list[dict[str, Any]] | None = None,
    oracle_rows: list[dict[str, Any]] | None = None,
    output_reserve: int = 512,
) -> dict[str, Any]:
    """
    Decompose prompt tokens into Policy (P), Tool Schemas (S), Interaction History (H),
    and evaluate both literal B_total and allocatable B_alloc (S + H) budget regimes.
    """
    minimal_user = [{"role": "user", "content": "Hi"}]
    base_framing = count_chat_tokens(tokenizer, minimal_user, tools=None)

    # 1. Policy component (P)
    system_msg = [{"role": "system", "content": system_prompt_text}]
    p_raw_tokens = len(
        tokenizer.encode(
            raw_policy_text if raw_policy_text is not None else system_prompt_text,
            add_special_tokens=False,
        )
    )
    with_p = count_chat_tokens(tokenizer, system_msg + minimal_user, tools=None)
    p_templated = with_p - base_framing

    # 2. Tool Schemas component (S)
    if tools:
        with_ps = count_chat_tokens(tokenizer, system_msg + minimal_user, tools=tools)
        s_templated = with_ps - with_p
    else:
        with_ps = with_p
        s_templated = 0

    # Marginal templated tokens per tool
    per_tool_templated: dict[str, int] = {}
    for t in tools:
        name = t.get("function", {}).get("name", t.get("name", "unknown"))
        per_tool_templated[name] = count_chat_tokens(tokenizer, system_msg + minimal_user, tools=[t]) - with_p

    # 3. Optional distractor tools footprint
    distractors_summary: dict[str, Any] | None = None
    if distractor_tools:
        with_p_dist = count_chat_tokens(tokenizer, system_msg + minimal_user, tools=distractor_tools)
        with_p_all = count_chat_tokens(tokenizer, system_msg + minimal_user, tools=tools + distractor_tools)
        distractors_summary = {
            "num_distractors": len(distractor_tools),
            "S_distractors_only_templated": with_p_dist - with_p,
            "S_retail_plus_distractors_templated": with_p_all - with_p,
            "P_plus_S_retail_plus_distractors_first_turn": with_p_all,
        }

    # 4. History component (H) on provided trajectory/demo
    full_messages = system_msg + (history_messages if history_messages else minimal_user)
    total_prompt_tokens = count_chat_tokens(tokenizer, full_messages, tools=tools)
    p_plus_s_only = count_chat_tokens(tokenizer, system_msg, tools=tools)
    h_templated = max(0, total_prompt_tokens - p_plus_s_only)

    # 5. Literal Total Context Window feasibility (B_total)
    min_tool_tokens = min(per_tool_templated.values()) if per_tool_templated else 0
    budget_feasibility_total: dict[str, Any] = {}
    for label, limit in BUDGET_LEVELS.items():
        max_input = limit - output_reserve
        remaining_for_s_and_h = max_input - p_templated
        budget_feasibility_total[label] = {
            "total_window": limit,
            "output_reserve": output_reserve,
            "max_input_budget": max_input,
            "P_system_prompt_templated": p_templated,
            "remaining_for_S_plus_H": remaining_for_s_and_h,
            "feasible_for_any_single_retail_tool": remaining_for_s_and_h >= min_tool_tokens,
            "fits_full_P_and_S_all": with_ps <= max_input,
            "left_for_H_after_S_all": max_input - with_ps,
        }

    # 6. Allocatable (S + H) Budget feasibility (Decision D1: B_alloc)
    budget_feasibility_alloc: dict[str, Any] = {}
    for label, b_alloc in BUDGET_LEVELS.items():
        entry: dict[str, Any] = {
            "B_alloc_S_plus_H": b_alloc,
            "P_pinned_tokens": p_templated,
            "output_reserve": output_reserve,
            "total_vllm_window_required": b_alloc + p_templated + output_reserve,
            "fits_S_all_with_room_for_H": s_templated <= b_alloc,
            "remaining_for_H_when_S_all_kept": b_alloc - s_templated,
        }
        if oracle_rows:
            n_tasks = len(oracle_rows)
            fit_min = sum(1 for r in oracle_rows if (r["S_min"] + r["H_oracle"]) <= b_alloc)
            fit_all = sum(1 for r in oracle_rows if (s_templated + r["H_oracle"]) <= b_alloc)
            entry["oracle_S_min_plus_H_fit_count"] = fit_min
            entry["oracle_S_min_plus_H_fit_total"] = n_tasks
            entry["oracle_S_min_plus_H_fit_pct"] = round(100.0 * fit_min / n_tasks, 1)
            entry["oracle_S_all_plus_H_fit_count"] = fit_all
            entry["oracle_S_all_plus_H_fit_pct"] = round(100.0 * fit_all / n_tasks, 1)
        budget_feasibility_alloc[label] = entry

    result: dict[str, Any] = {
        "model_tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
        "components": {
            "P_policy_raw_bpe_tokens": p_raw_tokens,
            "P_system_prompt_templated": p_templated,
            "S_all_tools_templated": s_templated,
            "S_num_tools": len(tools),
            "P_plus_S_plus_framing_first_turn": with_ps,
            "H_sample_history_templated": h_templated,
            "H_sample_num_messages": len(history_messages),
            "total_sample_prompt_tokens": total_prompt_tokens,
        },
        "per_tool_templated": per_tool_templated,
        "budget_feasibility_B_total": budget_feasibility_total,
        "budget_feasibility_B_alloc_S_plus_H": budget_feasibility_alloc,
    }
    if distractors_summary is not None:
        result["distractors_summary"] = distractors_summary
    if oracle_rows:
        result["oracle_summary"] = {
            "total_tasks": len(oracle_rows),
            "H_oracle": _summarize_distribution([r["H_oracle"] for r in oracle_rows]),
            "S_min": _summarize_distribution([r["S_min"] for r in oracle_rows]),
        }
    return result


def _demo_payload() -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    policy = (
        "# Retail Agent Policy\n"
        "1. You must authenticate the user by email or name + zip code before inspecting or modifying orders.\n"
        "2. You must ask for explicit user confirmation (yes/no) before calling any database-modifying tool.\n"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "find_user_id_by_email",
                "description": "Find user id by email address.",
                "parameters": {
                    "type": "object",
                    "properties": {"email": {"type": "string", "description": "Customer email"}},
                    "required": ["email"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Get order status, items, and tracking information.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string", "description": "Order ID starting with #W"}},
                    "required": ["order_id"],
                },
            },
        },
    ]
    history = [
        {"role": "user", "content": "Hi, check order #W9988776. My email is alice@example.com."}
    ]
    return policy, tools, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Authoritative Qwen3 BPE Token Accounting (CBA-6 / CBA-8).")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF Tokenizer model ID")
    parser.add_argument("--trajectory-file", type=str, default=None, help="Path to saved trajectory JSON")
    parser.add_argument("--system-prompt-file", type=str, default=None, help="Path to full formatted system prompt")
    parser.add_argument("--policy-file", type=str, default=None, help="Path to raw retail policy.md")
    parser.add_argument("--tools-file", type=str, default=None, help="Path to OpenAI tools JSON")
    parser.add_argument("--distractors-file", type=str, default=None, help="Optional distractor OpenAI tools JSON")
    parser.add_argument("--oracle-profile", type=str, default=None, help="Optional precomputed oracle_profile.json")
    parser.add_argument("--output-reserve", type=int, default=512, help="Reserved tokens for generation")
    parser.add_argument("--out", type=str, default=None, help="Path to write token breakdown JSON")
    args = parser.parse_args()

    print(f"[INFO] Loading tokenizer for {args.model} ...")
    tokenizer = load_tokenizer(args.model)

    raw_policy_text: str | None = None
    if args.trajectory_file:
        traj_data = json.loads(Path(args.trajectory_file).read_text(encoding="utf-8"))
        system_prompt = traj_data.get("system_prompt") or traj_data.get("policy", "")
        tools = traj_data.get("tools", [])
        messages = [m for m in traj_data.get("messages", []) if m.get("role") != "system"]
    else:
        system_prompt, tools, messages = _demo_payload()
        if args.policy_file:
            raw_policy_text = Path(args.policy_file).read_text(encoding="utf-8")
            if not args.system_prompt_file:
                system_prompt = raw_policy_text
        if args.system_prompt_file:
            system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
            if raw_policy_text is None:
                sibling_policy = Path(args.system_prompt_file).parent / "retail_policy.md"
                if sibling_policy.exists():
                    raw_policy_text = sibling_policy.read_text(encoding="utf-8")
        if args.tools_file:
            tools = json.loads(Path(args.tools_file).read_text(encoding="utf-8"))

    distractor_tools = (
        json.loads(Path(args.distractors_file).read_text(encoding="utf-8"))
        if args.distractors_file
        else None
    )
    oracle_rows = (
        json.loads(Path(args.oracle_profile).read_text(encoding="utf-8"))
        if args.oracle_profile
        else None
    )

    report = compute_component_breakdown(
        tokenizer=tokenizer,
        system_prompt_text=system_prompt,
        tools=tools,
        history_messages=messages,
        raw_policy_text=raw_policy_text,
        distractor_tools=distractor_tools,
        oracle_rows=oracle_rows,
        output_reserve=args.output_reserve,
    )

    print(json.dumps(report, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[DONE] Token breakdown written to {out_path}")


if __name__ == "__main__":
    main()
