"""Offline unit tests for `src/eval_runner.py` and `src/serving_profiler.py`."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from src.compliance_checkers import RetailComplianceEngine  # noqa: E402
from src.eval_runner import (  # noqa: E402
    BudgetedTurnExecutor,
    DEFAULT_GREETING,
    NLAssertionJudgeCache,
    OpenAICompatibleAgentClient,
    SweepCondition,
    evaluate_and_build_episode_record,
    evaluate_communicate_substring,
    evaluate_dual_metrics,
    expand_sweep_grid,
    load_task_ids_from_split,
    run_evaluation_suite,
    strip_boilerplate_greeting,
)
from src.serving_profiler import (  # noqa: E402
    VLLMMetricsScraper,
    parse_prometheus_metrics,
    summarize_episode_serving,
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


def _make_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: tuple[int, int] | None = None,
) -> Any:
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    usage_obj = (
        SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1])
        if usage is not None
        else None
    )
    return SimpleNamespace(choices=[choice] if finish_reason or content or reasoning_content or tool_calls else [], usage=usage_obj)


class _MockOpenAIClient:
    """Deterministic fake OpenAI streaming client recording all outbound requests."""

    def __init__(self, scripted_streams: list[list[Any]]) -> None:
        self._streams = list(scripted_streams)
        self.recorded_requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.recorded_requests.append(kwargs)
        assert self._streams, "MockOpenAIClient ran out of scripted streams"
        return iter(self._streams.pop(0))


def test_prometheus_metrics_parser_and_episode_delta() -> None:
    before_text = """
# HELP vllm:gpu_prefix_cache_hit_rate GPU prefix cache block hit rate
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate{model_name="Qwen/Qwen3-1.7B"} 0.25
vllm:gpu_prefix_cache_queries_total{model_name="Qwen/Qwen3-1.7B"} 1000.0
vllm:gpu_prefix_cache_hits_total{model_name="Qwen/Qwen3-1.7B"} 250.0
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen3-1.7B"} 0.12
vllm:time_to_first_token_seconds_sum{model_name="Qwen/Qwen3-1.7B"} 1.0
vllm:time_to_first_token_seconds_count{model_name="Qwen/Qwen3-1.7B"} 10.0
"""
    after_text = """
vllm:gpu_prefix_cache_hit_rate{model_name="Qwen/Qwen3-1.7B"} 0.60
vllm:gpu_prefix_cache_queries_total{model_name="Qwen/Qwen3-1.7B"} 3000.0
vllm:gpu_prefix_cache_hits_total{model_name="Qwen/Qwen3-1.7B"} 1750.0
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen3-1.7B"} 0.18
vllm:time_to_first_token_seconds_sum{model_name="Qwen/Qwen3-1.7B"} 1.6
vllm:time_to_first_token_seconds_count{model_name="Qwen/Qwen3-1.7B"} 14.0
"""
    b = parse_prometheus_metrics(before_text)
    a = parse_prometheus_metrics(after_text)
    delta = VLLMMetricsScraper.compute_episode_delta(b, a)

    # Delta hits = 1500, delta queries = 2000 => 0.75 exact episode hit rate
    assert delta["delta_prefix_cache_queries"] == 2000.0
    assert delta["delta_prefix_cache_hits"] == 1500.0
    assert delta["prefix_cache_hit_rate_delta"] == 0.75
    assert delta["prefix_cache_hit_rate_gauge"] == 0.6
    # Delta TTFT = 0.6s / 4 = 150ms
    assert delta["server_mean_ttft_ms"] == 150.0
    assert delta["gpu_cache_usage_perc"] == 0.18


def test_streaming_client_reassembles_tool_calls_and_catches_malformed_json() -> None:
    tc_chunk_1 = SimpleNamespace(
        index=0,
        id="call_abc",
        function=SimpleNamespace(name="find_user_id_", arguments='{"email":'),
    )
    tc_chunk_2 = SimpleNamespace(
        index=0,
        id=None,
        function=SimpleNamespace(name="by_email", arguments=' "yusuf@example.com"}'),
    )
    bad_tc_chunk = SimpleNamespace(
        index=0,
        id="call_bad",
        function=SimpleNamespace(name="cancel_pending_order", arguments='{"order_id": "#W1"'),  # truncated JSON
    )

    mock = _MockOpenAIClient(
        [
            [
                _make_chunk(reasoning_content="Authenticate user first."),
                _make_chunk(tool_calls=[tc_chunk_1]),
                _make_chunk(tool_calls=[tc_chunk_2], finish_reason="tool_calls"),
                _make_chunk(usage=(2200, 28)),
            ],
            [
                _make_chunk(tool_calls=[bad_tc_chunk], finish_reason="length"),
                _make_chunk(usage=(2500, 12)),
            ],
        ]
    )
    client = OpenAICompatibleAgentClient(model="Qwen/Qwen3-1.7B", enable_thinking=False, client=mock)

    resp1 = client.generate(messages=[{"role": "user", "content": "Hi"}], tools=[{"type": "function", "function": {"name": "find_user_id_by_email"}}], turn_index=0, seed=42)
    assert resp1.content is None
    assert resp1.reasoning_content == "Authenticate user first."
    assert len(resp1.tool_calls) == 1
    assert resp1.tool_calls[0] == {
        "id": "call_abc",
        "name": "find_user_id_by_email",
        "arguments": {"email": "yusuf@example.com"},
    }
    assert resp1.malformed_tool_calls == 0
    assert resp1.latency.prompt_tokens == 2200
    assert resp1.latency.completion_tokens == 28
    assert resp1.latency.ttft_ms >= 0.0
    assert mock.recorded_requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    resp2 = client.generate(messages=[{"role": "user", "content": "Cancel"}], tools=None, turn_index=1)
    assert resp2.malformed_tool_calls == 1
    assert resp2.tool_calls[0]["name"] == "cancel_pending_order"
    assert resp2.tool_calls[0]["arguments"] == {}


def test_strip_boilerplate_greeting_preserves_true_user_anchor() -> None:
    hist = [
        {"role": "assistant", "content": DEFAULT_GREETING},
        {"role": "user", "content": "Cancel order #W4776164, my email is yusuf@example.com."},
    ]
    stripped = strip_boilerplate_greeting(hist)
    assert len(stripped) == 1
    assert stripped[0]["role"] == "user"
    assert "yusuf@example.com" in stripped[0]["content"]

    # Does not strip non-greeting assistant messages
    hist_other = [
        {"role": "assistant", "content": "Custom intro."},
        {"role": "user", "content": "Hello"},
    ]
    assert strip_boilerplate_greeting(hist_other) == hist_other


def test_dual_metric_grading_and_nl_judge_cache(tmp_path: Path) -> None:
    # 1. <think> content must NOT satisfy communicate_info substring check
    msgs_think_only = [
        {"role": "user", "content": "What are the materials?"},
        {"role": "assistant", "content": "<think>The materials are polyester and cotton.</think>Let me check for you."},
    ]
    rew_fail, _ = evaluate_communicate_substring(msgs_think_only, ["polyester", "cotton"])
    assert rew_fail == 0.0

    msgs_good = [
        {"role": "user", "content": "What are the materials?"},
        {"role": "assistant", "content": "<think>Checked items.</think>The two t-shirts are made of polyester and cotton (total 1,234)."},
    ]
    rew_ok, checks = evaluate_communicate_substring(msgs_good, ["polyester", "cotton", "1234"])
    assert rew_ok == 1.0
    assert all(c["met"] for c in checks)

    # 2. NLAssertionJudgeCache caches verdicts on disk and avoids duplicate LLM calls
    call_counter = {"n": 0}

    def _fake_judge(sys_p: str, usr_p: str) -> str:
        call_counter["n"] += 1
        return json.dumps(
            {
                "results": [
                    {
                        "expectedOutcome": "Agent tells user materials are polyester and cotton",
                        "reasoning": "Agent stated both materials clearly.",
                        "metExpectation": True,
                    }
                ]
            }
        )

    cache_file = tmp_path / "nl_cache.json"
    cache = NLAssertionJudgeCache(cache_path=cache_file, judge_fn=_fake_judge)

    res1 = evaluate_dual_metrics(
        task_id="24",
        messages=msgs_good,
        termination_reason="user_stop",
        db_reward=1.0,
        communicate_info=["polyester", "cotton"],
        nl_assertions=["Agent tells user materials are polyester and cotton"],
        reward_basis=["DB", "NL_ASSERTION"],
        nl_judge_cache=cache,
        run_nl_judge=True,
    )
    assert res1.deterministic_success is True
    assert res1.official_tau2_success is True
    assert res1.nl_judge_cached is False
    assert call_counter["n"] == 1

    # Reload cache from disk -> second evaluation hits cache with 0 API calls
    cache2 = NLAssertionJudgeCache(cache_path=cache_file, judge_fn=_fake_judge)
    res2 = evaluate_dual_metrics(
        task_id="24",
        messages=msgs_good,
        termination_reason="user_stop",
        db_reward=1.0,
        communicate_info=["polyester", "cotton"],
        nl_assertions=["Agent tells user materials are polyester and cotton"],
        reward_basis=["DB", "NL_ASSERTION"],
        nl_judge_cache=cache2,
        run_nl_judge=True,
    )
    assert res2.official_tau2_success is True
    assert res2.nl_judge_cached is True
    assert call_counter["n"] == 1

    # 3. Non-clean termination (max_steps / too_many_errors) forces both scores to 0
    res_timeout = evaluate_dual_metrics(
        task_id="24",
        messages=msgs_good,
        termination_reason="max_steps",
        db_reward=1.0,
        communicate_info=["polyester", "cotton"],
        nl_assertions=None,
        reward_basis=["DB", "NL_ASSERTION"],
    )
    assert res_timeout.terminated_cleanly is False
    assert res_timeout.deterministic_success is False
    assert res_timeout.official_tau2_success is False


def test_split_loading_and_sweep_grid_expansion() -> None:
    eval_ids, _ = load_task_ids_from_split(CONFIGS / "retail_task_split.json", split="eval")
    train_ids, _ = load_task_ids_from_split(CONFIGS / "retail_task_split.json", split="train")
    assert len(eval_ids) == 50
    assert len(train_ids) == 58
    assert set(eval_ids).isdisjoint(set(train_ids))

    # Refuses excluded vacuous task IDs
    with pytest.raises(ValueError, match="vacuous-pass"):
        load_task_ids_from_split(CONFIGS / "retail_task_split.json", explicit_task_ids=["0", "10"])

    # Grid expansion checks across all presets
    pilot = expand_sweep_grid(CONFIGS / "sweep_grid.json", "baseline_pilot")
    assert len(pilot) == 1
    assert pilot[0].budget_config.schema_ratio is None

    rq1 = expand_sweep_grid(CONFIGS / "sweep_grid.json", "rq1_primary")
    assert len(rq1) == 3 * 4 * 3 * 3  # 3 budgets x 4 ratios x 3 distractors x 3 seeds = 108

    rq1_filtered = expand_sweep_grid(
        CONFIGS / "sweep_grid.json",
        "rq1_primary",
        budgets_filter=[2048],
        ratios_filter=[0.5],
        distractors_filter=[0, 23],
        seeds_filter=[42],
    )
    assert len(rq1_filtered) == 2

    rq1b = expand_sweep_grid(CONFIGS / "sweep_grid.json", "rq1b_policy_retrieval")
    assert len(rq1b) == 4  # 2 budgets x 2 policy_modes

    rq4 = expand_sweep_grid(CONFIGS / "sweep_grid.json", "rq4_prefix_cache_h4")
    assert len(rq4) == 8  # 2 budgets x 2 history_modes x 2 prefix_caching

    m7 = expand_sweep_grid(CONFIGS / "sweep_grid.json", "milestone7_unseen_tools")
    assert len(m7) == 2
    assert all(c.budget_config.distractor_source == "unseen" and c.budget_config.num_distractors == 13 for c in m7)


def test_end_to_end_budgeted_executor_and_suite_resume(tokenizer, retail_assets, tmp_path: Path) -> None:
    sys_prompt, raw_policy, tools, pool = retail_assets
    controller = ContextBudgetController(tokenizer, distractor_pool=pool)
    compliance = RetailComplianceEngine()

    cond = SweepCondition(
        condition_id="rq1_b4096_r50_d10_s42",
        budget_config=BudgetConfig(b_alloc=4096, schema_ratio=0.50, num_distractors=10, distractor_source="in_dist", seed=42),
    )

    executed_calls: list[str] = []

    def _mock_episode_runner(condition: SweepCondition, tid: str) -> dict[str, Any]:
        executed_calls.append(tid)
        # Simulate a 2-turn conversation: Turn 1 calls find_user_id_by_email, Turn 2 confirms cancellation
        tc1 = SimpleNamespace(
            index=0,
            id="c1",
            function=SimpleNamespace(name="find_user_id_by_email", arguments='{"email": "yusuf@example.com"}'),
        )
        mock_openai = _MockOpenAIClient(
            [
                [_make_chunk(tool_calls=[tc1], finish_reason="tool_calls"), _make_chunk(usage=(2800, 18))],
                [_make_chunk(content="Order #W4776164 is cancelled.", finish_reason="stop"), _make_chunk(usage=(3000, 12))],
            ]
        )
        client = OpenAICompatibleAgentClient(model="Qwen/Qwen3-1.7B", client=mock_openai)
        executor = BudgetedTurnExecutor(
            budget_controller=controller,
            agent_client=client,
            system_prompt=sys_prompt,
            raw_policy=raw_policy,
            base_tools=tools,
            budget_config=condition.budget_config,
            seed=condition.budget_config.seed,
        )

        hist: list[dict[str, Any]] = [
            {"role": "assistant", "content": DEFAULT_GREETING},
            {"role": "user", "content": "Hi, cancel pending order #W4776164. My email is yusuf@example.com."},
        ]
        r1, alloc1 = executor.step(hist)
        assert alloc1.telemetry.within_budget is True
        assert alloc1.trimmed_history[0]["role"] == "user"  # Greeting stripped!
        hist.append(r1.to_openai_message())
        hist.append({"role": "tool", "tool_call_id": "c1", "name": "find_user_id_by_email", "content": "yusuf_rossi_9620"})

        r2, alloc2 = executor.step(hist)
        assert alloc2.telemetry.within_budget is True
        hist.append(r2.to_openai_message())

        prom_b = {"prefix_cache_queries_total": 100.0, "prefix_cache_hits_total": 50.0, "gpu_prefix_cache_hit_rate": 0.5}
        prom_a = {"prefix_cache_queries_total": 500.0, "prefix_cache_hits_total": 370.0, "gpu_prefix_cache_hit_rate": 0.74}

        return evaluate_and_build_episode_record(
            run_id="test_run",
            condition=condition,
            model_name="Qwen/Qwen3-1.7B",
            tokenizer_name="Qwen/Qwen3-1.7B",
            task_id=tid,
            trajectory=hist,
            termination_reason="user_stop",
            db_reward=1.0 if tid == "0" else 0.0,
            communicate_info=None,
            nl_assertions=None,
            reward_basis=["DB", "NL_ASSERTION"],
            allow_transfer=False,
            turn_allocations=executor.turn_allocations,
            turn_latencies=executor.turn_latencies,
            malformed_tool_call_count=executor.malformed_tool_call_count,
            compliance_engine=compliance,
            prom_before=prom_b,
            prom_after=prom_a,
        )

    out_dir = tmp_path / "run_01"
    summary1 = run_evaluation_suite(
        run_id="test_run",
        out_dir=out_dir,
        conditions=[cond],
        task_ids=["0", "1"],
        episode_runner=_mock_episode_runner,
        resume=True,
    )
    assert executed_calls == ["0", "1"]
    assert summary1["total_episodes"] == 2
    assert summary1["overall"]["deterministic_pass_rate"] == 0.5
    assert summary1["overall"]["passes_15pct_floor_gate"] is True
    assert summary1["overall"]["all_turns_within_budget_rate"] == 1.0
    assert summary1["overall"]["mean_prefix_cache_hit_rate_delta"] == 0.8  # (370-50)/(500-100) = 0.8

    # Verify results.csv and summary.json exist and have expected rows
    assert (out_dir / "summary.json").exists()
    with (out_dir / "results.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["task_id"] == "0"
    assert rows[0]["deterministic_success"] == "1"

    # Resume with an added task "2": must NOT re-run "0" or "1"!
    summary2 = run_evaluation_suite(
        run_id="test_run",
        out_dir=out_dir,
        conditions=[cond],
        task_ids=["0", "1", "2"],
        episode_runner=_mock_episode_runner,
        resume=True,
    )
    assert executed_calls == ["0", "1", "2"]
    assert summary2["total_episodes"] == 3
