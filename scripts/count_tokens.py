#!/usr/bin/env python3
"""
[CBA-6] Authoritative Token Accounting with Qwen3 Native Chat Template.

Measures exact BPE token consumption across the three competing context components:
  - S (Tool Schemas): Base retail tool definitions + optional distractor tools
  - P (Policy Context): System instructions and retail operational policy rules
  - H (Interaction History): Multi-turn user/assistant/tool messages

Also checks compatibility against the proposal's fixed budgets: 2k (2048), 4k (4096), 8k (8192).
Runs 100% on CPU (requires only `transformers` and `tokenizers`).
"""

from __future__ import annotations

import argparse
import json
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
    """Count exact serialized tokens using Qwen3's native chat template."""
    token_ids = tokenizer.apply_chat_template(
        messages,
        tools=tools if tools else None,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


def compute_component_breakdown(
    tokenizer: Any,
    policy_text: str,
    tools: list[dict[str, Any]],
    history_messages: list[dict[str, Any]],
    output_reserve: int = 512,
) -> dict[str, Any]:
    """
    Decompose total prompt tokens into Policy (P), Tool Schemas (S),
    Interaction History (H), and template framing overhead.
    """
    minimal_user = [{"role": "user", "content": "Hello"}]
    base_framing = count_chat_tokens(tokenizer, minimal_user, tools=None) - len(
        tokenizer.encode("Hello", add_special_tokens=False)
    )

    # 1. Policy component (P)
    system_msg = [{"role": "system", "content": policy_text}]
    p_raw_tokens = len(tokenizer.encode(policy_text, add_special_tokens=False))
    p_templated = count_chat_tokens(tokenizer, system_msg + minimal_user, tools=None) - count_chat_tokens(
        tokenizer, minimal_user, tools=None
    )

    # 2. Tool Schemas component (S)
    if tools:
        s_templated = count_chat_tokens(tokenizer, system_msg + minimal_user, tools=tools) - count_chat_tokens(
            tokenizer, system_msg + minimal_user, tools=None
        )
    else:
        s_templated = 0

    # Per-tool breakdown
    per_tool_tokens: dict[str, int] = {}
    for t in tools:
        name = t.get("function", {}).get("name", t.get("name", "unknown"))
        t_json = json.dumps(t, ensure_ascii=False)
        per_tool_tokens[name] = len(tokenizer.encode(t_json, add_special_tokens=False))

    # 3. Full prompt with History (P + S + H)
    full_messages = system_msg + (history_messages if history_messages else minimal_user)
    total_prompt_tokens = count_chat_tokens(tokenizer, full_messages, tools=tools)

    # 4. History component (H)
    p_plus_s_tokens = count_chat_tokens(tokenizer, system_msg, tools=tools)
    h_templated = max(0, total_prompt_tokens - p_plus_s_tokens)

    # 5. Budget feasibility analysis across 2k / 4k / 8k
    budget_analysis = {}
    for label, limit in BUDGET_LEVELS.items():
        max_input = limit - output_reserve
        remaining_for_s_and_h = max_input - p_templated
        budget_analysis[label] = {
            "total_window": limit,
            "output_reserve": output_reserve,
            "max_input_budget": max_input,
            "policy_pinned_tokens": p_templated,
            "remaining_for_S_plus_H": remaining_for_s_and_h,
            "fits_full_P_and_S": (p_templated + s_templated) <= max_input,
            "fits_current_prompt": total_prompt_tokens <= max_input,
            "overflow_tokens": max(0, total_prompt_tokens - max_input),
        }

    return {
        "model_tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
        "total_serialized_prompt_tokens": total_prompt_tokens,
        "components": {
            "P_policy_templated_tokens": p_templated,
            "P_policy_raw_bpe_tokens": p_raw_tokens,
            "S_schemas_templated_tokens": s_templated,
            "S_num_tools": len(tools),
            "H_history_templated_tokens": h_templated,
            "H_num_messages": len(history_messages),
            "template_base_framing_tokens": base_framing,
        },
        "shares_pct": {
            "P_pct": round(100.0 * p_templated / max(1, total_prompt_tokens), 2),
            "S_pct": round(100.0 * s_templated / max(1, total_prompt_tokens), 2),
            "H_pct": round(100.0 * h_templated / max(1, total_prompt_tokens), 2),
        },
        "per_tool_bpe_tokens": per_tool_tokens,
        "budget_feasibility_2k_4k_8k": budget_analysis,
    }


def _demo_payload() -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    policy = (
        "# Retail Agent Policy\n"
        "1. You must authenticate the user by email or name + zip code before inspecting or modifying orders.\n"
        "2. You must ask for explicit user confirmation (yes/no) before calling any database-modifying tool "
        "(cancel, modify, return, exchange).\n"
        "3. Pending orders can only be cancelled or modified once. Delivered orders can be returned or exchanged.\n"
        "4. Transfer to a human agent only if the request falls outside the available tools or policy rules."
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
        {"role": "user", "content": "Hi, I need to cancel order #W9988776. My email is alice@example.com."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "find_user_id_by_email",
                        "arguments": '{"email": "alice@example.com"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "find_user_id_by_email", "content": "user_alice_01"},
    ]
    return policy, tools, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Authoritative Qwen3 BPE Token Accounting (CBA-6).")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF Tokenizer model ID")
    parser.add_argument("--trajectory-file", type=str, default=None, help="Path to saved trajectory JSON")
    parser.add_argument("--policy-file", type=str, default=None, help="Optional path to retail policy.md")
    parser.add_argument("--tools-file", type=str, default=None, help="Optional path to tools JSON")
    parser.add_argument("--output-reserve", type=int, default=512, help="Reserved tokens for generation")
    parser.add_argument("--out", type=str, default=None, help="Path to write token breakdown JSON")
    args = parser.parse_args()

    print(f"[INFO] Loading tokenizer for {args.model} ...")
    tokenizer = load_tokenizer(args.model)

    if args.trajectory_file:
        traj_data = json.loads(Path(args.trajectory_file).read_text(encoding="utf-8"))
        policy = traj_data.get("system_prompt") or traj_data.get("policy", "")
        tools = traj_data.get("tools", [])
        messages = [m for m in traj_data.get("messages", []) if m.get("role") != "system"]
    else:
        policy, tools, messages = _demo_payload()
        if args.policy_file:
            policy = Path(args.policy_file).read_text(encoding="utf-8")
        if args.tools_file:
            tools = json.loads(Path(args.tools_file).read_text(encoding="utf-8"))

    report = compute_component_breakdown(
        tokenizer=tokenizer,
        policy_text=policy,
        tools=tools,
        history_messages=messages,
        output_reserve=args.output_reserve,
    )

    print(json.dumps(report, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[DONE] Token breakdown written to {out_path}")


if __name__ == "__main__":
    main()
