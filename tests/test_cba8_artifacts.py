"""[CBA-8] Automated verification tests for frozen retail artifacts, token accounting, and split hygiene."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
NOTES = REPO_ROOT / "notes"
SCRIPTS = REPO_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))
import count_tokens  # noqa: E402


EXPECTED_TASKS_SHA256 = "f44563d8f36362e2dc58282bb02cfb85010d22f27d9439808758d3e9b2bc16d2"
EXPECTED_VACUOUS_IDS = {"10", "12", "25", "50", "57", "65"}


def test_retail_tools_and_metadata() -> None:
    tools = json.loads((CONFIGS / "retail_tools_openai.json").read_text(encoding="utf-8"))
    meta = json.loads((CONFIGS / "retail_tool_metadata.json").read_text(encoding="utf-8"))

    assert isinstance(tools, list)
    assert len(tools) == 16
    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 16
    assert "get_item_details" in names

    assert meta["total_tools"] == 16
    assert len(meta["write_tools"]) == 7
    assert len(meta["read_tools"]) == 7
    assert len(meta["generic_tools"]) == 2


def test_distractor_tools_disjoint_and_valid_openai_schemas() -> None:
    retail = json.loads((CONFIGS / "retail_tools_openai.json").read_text(encoding="utf-8"))
    in_dist = json.loads((CONFIGS / "distractor_tools_in_dist.json").read_text(encoding="utf-8"))
    unseen = json.loads((CONFIGS / "distractor_tools_unseen.json").read_text(encoding="utf-8"))

    assert len(in_dist) >= 20
    assert len(unseen) >= 10

    for item in in_dist + unseen:
        assert item.get("type") == "function"
        fn = item.get("function", {})
        assert isinstance(fn.get("name"), str) and fn["name"]
        assert "parameters" in fn

    r_names = {t["function"]["name"] for t in retail}
    i_names = {t["function"]["name"] for t in in_dist}
    u_names = {t["function"]["name"] for t in unseen}

    assert len(i_names) == len(in_dist)
    assert len(u_names) == len(unseen)
    assert r_names.isdisjoint(i_names)
    assert r_names.isdisjoint(u_names)
    assert i_names.isdisjoint(u_names)


def test_count_chat_tokens_immune_to_transformers5_bug() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = count_tokens.load_tokenizer("Qwen/Qwen3-1.7B")
    sys_prompt = (CONFIGS / "retail_system_prompt.txt").read_text(encoding="utf-8")
    tools = json.loads((CONFIGS / "retail_tools_openai.json").read_text(encoding="utf-8"))

    user_only = [{"role": "user", "content": "Hi"}]
    with_p = [{"role": "system", "content": sys_prompt}, *user_only]

    n_base = count_tokens.count_chat_tokens(tok, user_only)
    n_with_p = count_tokens.count_chat_tokens(tok, with_p)
    n_with_ps = count_tokens.count_chat_tokens(tok, with_p, tools=tools)

    assert n_with_p - n_base == 1492
    assert n_with_ps - n_with_p == 3382
    assert n_with_ps == 4883


def test_retail_task_split_leak_free_and_non_vacuous() -> None:
    split_data = json.loads((CONFIGS / "retail_task_split.json").read_text(encoding="utf-8"))

    assert split_data["source_tasks_sha256"] == EXPECTED_TASKS_SHA256
    assert split_data["pinned_release"] == "v1.0.1"
    assert split_data["pinned_commit"] == "fc0055dc4e0a316c3f83133267fbd6faaa770992"

    counts = split_data["counts"]
    assert counts["total_raw_tasks"] == 114
    assert counts["excluded_vacuous_pass"] == 6
    assert counts["valid_tasks"] == 108
    assert counts["eval_split_count"] == 50
    assert counts["train_split_count"] == 58
    assert counts["persona_overlap_count"] == 0

    excluded_ids = {e["task_id"] for e in split_data["excluded_tasks"]}
    assert excluded_ids == EXPECTED_VACUOUS_IDS

    eval_ids = set(split_data["eval_task_ids"])
    train_ids = set(split_data["train_task_ids"])

    assert len(eval_ids) == 50
    assert len(train_ids) == 58
    assert eval_ids.isdisjoint(train_ids)
    assert EXPECTED_VACUOUS_IDS.isdisjoint(eval_ids | train_ids)
    assert "24" in (eval_ids | train_ids)

    meta = split_data["task_metadata"]
    eval_personas = {meta[tid]["known_info"] for tid in eval_ids}
    train_personas = {meta[tid]["known_info"] for tid in train_ids}
    assert eval_personas.isdisjoint(train_personas)


def test_retail_token_profile_numbers() -> None:
    profile = json.loads((NOTES / "retail_token_profile.json").read_text(encoding="utf-8"))
    comp = profile["components"]
    assert comp["P_policy_raw_bpe_tokens"] == 1407
    assert comp["P_system_prompt_templated"] == 1492
    assert comp["S_all_tools_templated"] == 3382
    assert comp["S_num_tools"] == 16
    assert comp["P_plus_S_plus_framing_first_turn"] == 4883

    b_total = profile["budget_feasibility_B_total"]
    assert b_total["2k"]["remaining_for_S_plus_H"] == 44
    assert b_total["2k"]["feasible_for_any_single_retail_tool"] is False

    b_alloc = profile["budget_feasibility_B_alloc_S_plus_H"]
    assert b_alloc["2k"]["oracle_S_min_plus_H_fit_count"] == 42
    assert b_alloc["4k"]["oracle_S_min_plus_H_fit_count"] == 82
    assert b_alloc["8k"]["oracle_S_min_plus_H_fit_count"] == 113
