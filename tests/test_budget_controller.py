"""[CBA-9] Offline unit tests for ContextBudgetController using real Qwen3-1.7B tokenizer & frozen configs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import count_tokens  # noqa: E402
from src.budget_controller import (  # noqa: E402
    BudgetConfig,
    ContextBudgetController,
    DistractorPool,
)


@pytest.fixture(scope="module")
def tokenizer():
    return count_tokens.load_tokenizer("Qwen/Qwen3-1.7B")


@pytest.fixture(scope="module")
def retail_assets():
    sys_prompt = (CONFIGS / "retail_system_prompt.txt").read_text(encoding="utf-8")
    raw_policy = (CONFIGS / "retail_policy.md").read_text(encoding="utf-8")
    tools = json.loads((CONFIGS / "retail_tools_openai.json").read_text(encoding="utf-8"))
    pool = DistractorPool.from_configs_dir(CONFIGS)
    return sys_prompt, raw_policy, tools, pool


def _build_long_retail_history() -> list[dict]:
    """Construct a realistic ~4,500-token multi-turn retail conversation with tool-call pairs."""
    order_blob = json.dumps(
        {
            "order_id": "#W2378156",
            "status": "delivered",
            "items": [
                {
                    "name": f"Product Variant {i}",
                    "product_id": f"16563670{i:02d}",
                    "item_id": f"11512936{i:02d}",
                    "price": 125.50 + i,
                    "options": {"color": "black", "size": "full", "switch": "linear", "backlight": "RGB"},
                }
                for i in range(25)
            ],
            "payment_history": [{"transaction_type": "payment", "amount": 3250.0, "payment_method_id": "credit_card_9513926"}],
        }
    )
    catalog_blob = json.dumps(
        {
            "product_id": "1656367028",
            "name": "Mechanical Keyboard",
            "variants": {
                f"7706410{i:03d}": {
                    "item_id": f"7706410{i:03d}",
                    "available": i % 2 == 0,
                    "price": 270.0 + i,
                    "options": {"switch type": "clicky" if i % 2 == 0 else "tactile", "backlight": "RGB", "size": "full size"},
                }
                for i in range(45)
            },
        }
    )
    return [
        {"role": "user", "content": "Hi! I am Yusuf Rossi (zip 19122). I want to exchange the keyboard in order #W2378156 for a clicky switch one, and maybe cancel order #W4776164."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "find_user_id_by_name_zip", "arguments": '{"first_name":"Yusuf","last_name":"Rossi","zip":"19122"}'}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "find_user_id_by_name_zip", "content": "yusuf_rossi_9620"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "get_user_details", "arguments": '{"user_id":"yusuf_rossi_9620"}'}}],
        },
        {"role": "tool", "tool_call_id": "call_2", "name": "get_user_details", "content": '{"user_id":"yusuf_rossi_9620","orders":["#W2378156","#W4776164"],"payment_methods":{"credit_card_9513926":{"brand":"mastercard"}}}'} ,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "get_order_details", "arguments": '{"order_id":"#W2378156"}'}}],
        },
        {"role": "tool", "tool_call_id": "call_3", "name": "get_order_details", "content": order_blob},
        {"role": "user", "content": "Please check available clicky RGB full-size mechanical keyboards first."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_4", "type": "function", "function": {"name": "get_product_details", "arguments": '{"product_id":"1656367028"}'}}],
        },
        {"role": "tool", "tool_call_id": "call_4", "name": "get_product_details", "content": catalog_blob},
        {"role": "user", "content": "Great, please cancel my pending order #W4776164 instead because it is no longer needed."},
    ]


@pytest.mark.parametrize("b_alloc", [2048, 4096, 8192])
@pytest.mark.parametrize("schema_ratio", [0.75, 0.50, 0.25])
def test_strict_token_ceilings_across_full_grid(tokenizer, retail_assets, b_alloc: int, schema_ratio: float) -> None:
    sys_prompt, raw_policy, tools, pool = retail_assets
    ctrl = ContextBudgetController(tokenizer, distractor_pool=pool)
    history = _build_long_retail_history()

    cfg = BudgetConfig(
        b_alloc=b_alloc,
        schema_ratio=schema_ratio,
        num_distractors=23,
        distractor_source="in_dist",
    )
    res = ctrl.allocate(sys_prompt, tools, history, cfg, raw_policy=raw_policy)
    tel = res.telemetry

    assert tel.p_tokens == 1492
    assert tel.s_tokens <= tel.b_s
    assert tel.h_tokens <= tel.b_h
    assert tel.within_budget is True
    assert res.trimmed_history[0]["content"] == history[0]["content"]


def test_unconstrained_baseline_keeps_all(tokenizer, retail_assets) -> None:
    sys_prompt, raw_policy, tools, pool = retail_assets
    ctrl = ContextBudgetController(tokenizer, distractor_pool=pool)
    history = _build_long_retail_history()

    cfg = BudgetConfig(b_alloc=8192, schema_ratio=None, num_distractors=0)
    res = ctrl.allocate(sys_prompt, tools, history, cfg, raw_policy=raw_policy)

    assert res.telemetry.num_tools_kept == 16
    assert res.telemetry.s_tokens == 3382
    assert res.telemetry.num_messages_kept == len(history)
    assert res.telemetry.within_budget is True


def test_atomic_tool_call_pair_integrity(tokenizer, retail_assets) -> None:
    sys_prompt, raw_policy, tools, pool = retail_assets
    ctrl = ContextBudgetController(tokenizer, distractor_pool=pool)
    history = _build_long_retail_history()

    for b_alloc, ratio in [(2048, 0.50), (4096, 0.50), (4096, 0.75)]:
        cfg = BudgetConfig(b_alloc=b_alloc, schema_ratio=ratio, num_distractors=10)
        res = ctrl.allocate(sys_prompt, tools, history, cfg, raw_policy=raw_policy)
        msgs = res.trimmed_history

        pending_ids: set[str] = set()
        for msg in msgs:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                assert not pending_ids, "Previous assistant tool_calls were left without tool responses!"
                pending_ids = {tc["id"] for tc in msg["tool_calls"]}
            elif role == "tool":
                assert msg["tool_call_id"] in pending_ids, f"Orphaned tool message {msg['tool_call_id']}!"
                pending_ids.discard(msg["tool_call_id"])
            else:
                assert not pending_ids, f"Unclosed tool_calls {pending_ids} before {role} message!"
        assert not pending_ids


def test_oversized_latest_tool_output_truncation(tokenizer, retail_assets) -> None:
    sys_prompt, raw_policy, tools, pool = retail_assets
    ctrl = ContextBudgetController(tokenizer, distractor_pool=pool)
    history = _build_long_retail_history()[:-1]  # Ends directly on massive ~2,000-token get_product_details tool output

    cfg = BudgetConfig(b_alloc=2048, schema_ratio=0.75, num_distractors=0)  # B_H = 512
    res = ctrl.allocate(sys_prompt, tools, history, cfg, raw_policy=raw_policy)

    assert res.telemetry.b_h == 512
    assert res.telemetry.h_tokens <= 512
    assert res.telemetry.truncated_tool_outputs >= 1
    assert res.trimmed_history[-1]["role"] == "tool"
    assert "...[truncated for context budget]..." in res.trimmed_history[-1]["content"]
    assert res.telemetry.within_budget is True


def test_distractor_injection_and_bm25_relevance(tokenizer, retail_assets) -> None:
    sys_prompt, raw_policy, tools, pool = retail_assets
    ctrl = ContextBudgetController(tokenizer, distractor_pool=pool)
    history = [{"role": "user", "content": "I need to cancel pending order #W4776164 because it is no longer needed."}]

    # Deterministic in-dist injection
    cfg_a = BudgetConfig(b_alloc=4096, schema_ratio=0.50, num_distractors=23, distractor_source="in_dist", seed=42)
    cfg_b = BudgetConfig(b_alloc=4096, schema_ratio=0.50, num_distractors=23, distractor_source="in_dist", seed=42)
    res_a = ctrl.allocate(sys_prompt, tools, history, cfg_a, raw_policy=raw_policy)
    res_b = ctrl.allocate(sys_prompt, tools, history, cfg_b, raw_policy=raw_policy)

    assert res_a.telemetry.num_tools_candidate == 39  # 16 retail + 23 in_dist
    assert res_a.telemetry.selected_tool_names == res_b.telemetry.selected_tool_names
    assert len(set(res_a.telemetry.selected_tool_names)) == len(res_a.telemetry.selected_tool_names)
    assert "cancel_pending_order" in res_a.telemetry.selected_tool_names

    # Unseen banking_knowledge injection
    cfg_unseen = BudgetConfig(b_alloc=4096, schema_ratio=0.50, num_distractors=13, distractor_source="unseen", seed=42)
    res_unseen = ctrl.allocate(sys_prompt, tools, history, cfg_unseen, raw_policy=raw_policy)
    assert res_unseen.telemetry.num_tools_candidate == 29  # 16 retail + 13 unseen
    assert "cancel_pending_order" in res_unseen.telemetry.selected_tool_names


def test_canonical_sort_prefix_cache_stability(tokenizer, retail_assets) -> None:
    sys_prompt, raw_policy, tools, pool = retail_assets
    ctrl = ContextBudgetController(tokenizer, distractor_pool=pool)
    history = _build_long_retail_history()

    cfg = BudgetConfig(b_alloc=4096, schema_ratio=0.50, num_distractors=23, canonical_sort_tools=True)
    res = ctrl.allocate(sys_prompt, tools, history, cfg, raw_policy=raw_policy)
    assert res.telemetry.selected_tool_names == sorted(res.telemetry.selected_tool_names)


def test_compact_schema_mode_packs_more_tools(tokenizer, retail_assets) -> None:
    sys_prompt, raw_policy, tools, pool = retail_assets
    ctrl = ContextBudgetController(tokenizer, distractor_pool=pool)
    history = [{"role": "user", "content": "Help me exchange delivered order items or cancel pending order."}]

    cfg_full = BudgetConfig(b_alloc=2048, schema_ratio=0.25, schema_mode="select", num_distractors=0)
    cfg_comp = BudgetConfig(b_alloc=2048, schema_ratio=0.25, schema_mode="compact_select", num_distractors=0)

    res_full = ctrl.allocate(sys_prompt, tools, history, cfg_full, raw_policy=raw_policy)
    res_comp = ctrl.allocate(sys_prompt, tools, history, cfg_comp, raw_policy=raw_policy)

    assert res_full.telemetry.s_tokens <= 512
    assert res_comp.telemetry.s_tokens <= 512
    assert res_comp.telemetry.num_tools_kept > res_full.telemetry.num_tools_kept


def test_rq1b_policy_retrieval_reduces_p_tokens(tokenizer, retail_assets) -> None:
    sys_prompt, raw_policy, tools, pool = retail_assets
    ctrl = ContextBudgetController(tokenizer, distractor_pool=pool)
    history = [{"role": "user", "content": "I want to cancel my pending order because I ordered by mistake."}]

    cfg = BudgetConfig(b_alloc=2048, schema_ratio=0.50, policy_mode="retrieved", policy_top_k=2, num_distractors=0)
    res = ctrl.allocate(sys_prompt, tools, history, cfg, raw_policy=raw_policy)

    assert "<instructions>" in res.system_prompt
    assert "<policy>" in res.system_prompt
    assert "## Cancel pending order" in res.system_prompt
    assert res.telemetry.p_tokens < 1000  # Significantly smaller than pinned 1,492


def test_h4_summarize_dropped_mode(tokenizer, retail_assets) -> None:
    sys_prompt, raw_policy, tools, pool = retail_assets
    ctrl = ContextBudgetController(tokenizer, distractor_pool=pool)
    history = _build_long_retail_history()

    cfg = BudgetConfig(b_alloc=2048, schema_ratio=0.25, history_mode="summarize_dropped", num_distractors=0)  # B_H = 1536
    res = ctrl.allocate(sys_prompt, tools, history, cfg, raw_policy=raw_policy)

    assert res.telemetry.h_tokens <= 1536
    assert res.telemetry.within_budget is True
    assert any("[Earlier conversation summary]" in str(m.get("content", "")) for m in res.trimmed_history)
