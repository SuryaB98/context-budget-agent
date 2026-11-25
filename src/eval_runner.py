#!/usr/bin/env python3
"""
End-to-End Context-Budgeted Evaluation Harness (`src/eval_runner.py`).

Wires together:
  - `ContextBudgetController` (`src/budget_controller.py`) with exact Qwen3 BPE budgeting
  - OpenAI-compatible vLLM endpoint (streaming for exact client-side TTFT + Hermes tool calls)
  - `tau2-bench@v1.0.1` Retail `Orchestrator` + `UserSimulator` + `Environment` (imported lazily!)
  - Decision D3 Dual-Metric Grader:
      1. Strict Deterministic `DB + COMMUNICATE` substring score (Hard Gate G3)
      2. Official `tau2` `DB x NL_ASSERTION` score with disk-cached `gpt-4.1-2025-04-14` judge
  - `RetailComplianceEngine` (`src/compliance_checkers.py`, Rules `R1`–`R8` & Silent Violations)
  - `VLLMMetricsScraper` (`src/serving_profiler.py`, PagedAttention prefix-cache hit rate & TTFT)

Supports single-condition runs, named grids from `configs/sweep_grid.json`, `--resume` crash
recovery, LoRA adapter evaluation (`--model` vs `--tokenizer`), and 100% offline unit testing.
"""

from __future__ import annotations

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from src.budget_controller import (
    AllocationResult,
    BudgetConfig,
    ContextBudgetController,
    DistractorPool,
)
from src.compliance_checkers import (
    RetailComplianceEngine,
    strip_think_blocks,
)
from src.serving_profiler import (
    EpisodeServingMetrics,
    TurnLatencyRecord,
    VLLMMetricsScraper,
    summarize_episode_serving,
)


DEFAULT_GREETING = "Hi! How can I help you today?"
CLEAN_TERMINATION_REASONS = frozenset({"user_stop", "agent_stop"})

NL_ASSERTION_SYSTEM_PROMPT = """
TASK
- You will be given a list of expected outcomes and a conversation that was collected during a test case run.
- The conversation is between an agent and a customer.
- Your job is to evaluate whether the agent satisfies each of the expected outcomes.
- Grade each expected outcome individually.

FORMAT
- Your response should be a JSON object with the following fields:
- `reasoning`: a short explanation for your classification
- `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
- `expectedOutcome`: repeat the expectation from the input that you are grading

Example response structure:
{
    "results": [
        {
            "expectedOutcome": "<one of the expected outcomes from the input>",
            "reasoning": "<reasoning trace>",
            "metExpectation": false
        }
    ]
}
""".strip()


@dataclass
class AgentTurnResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]
    latency: TurnLatencyRecord
    reasoning_content: str = ""
    malformed_tool_calls: int = 0
    finish_reason: str | None = None

    def to_openai_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": self.content or "",
        }
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


@dataclass
class DualMetricResult:
    terminated_cleanly: bool
    termination_reason: str
    db_reward: float
    communicate_reward: float
    communicate_checks: list[dict[str, Any]]
    deterministic_success: bool
    nl_reward: float | None
    nl_checks: list[dict[str, Any]]
    nl_judge_cached: bool
    official_tau2_success: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepCondition:
    condition_id: str
    budget_config: BudgetConfig
    prefix_caching_expected: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "budget_config": asdict(self.budget_config),
            "prefix_caching_expected": self.prefix_caching_expected,
        }


def strip_boilerplate_greeting(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove tau2's static Turn-0 assistant greeting (`Hi! How can I help you today?`)
    before passing history to `ContextBudgetController.allocate()` so that `blocks[0]`
    anchors on the actual initial user goal turn rather than the static greeting.
    """
    if (
        len(history) >= 2
        and history[0].get("role") == "assistant"
        and not history[0].get("tool_calls")
        and str(history[0].get("content", "")).strip() == DEFAULT_GREETING
        and history[1].get("role") == "user"
    ):
        return history[1:]
    return history


def evaluate_communicate_substring(
    messages: list[dict[str, Any]],
    communicate_info: list[str] | None,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Exact deterministic tau2 `CommunicateEvaluator` substring check across assistant
    user-facing text turns (stripping Qwen3 `<think>...</think>` blocks first).
    """
    if not communicate_info:
        return 1.0, []

    checks: list[dict[str, Any]] = []
    for info_str in communicate_info:
        needle = str(info_str).lower()
        found = False
        matched_snippet = ""
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            text = strip_think_blocks(msg.get("content"))
            if not text:
                continue
            if needle in text.lower().replace(",", ""):
                found = True
                matched_snippet = text[:200]
                break
        checks.append(
            {
                "info": info_str,
                "met": found,
                "matched_snippet": matched_snippet,
            }
        )
    reward = 1.0 if all(c["met"] for c in checks) else 0.0
    return reward, checks


class NLAssertionJudgeCache:
    """
    Disk-cached LLM judge for tau2 `nl_assertions` (`gpt-4.1-2025-04-14`, `temperature=0.0`).
    Guarantees reproducible re-grading and zero duplicate judge API charges.
    """

    def __init__(
        self,
        cache_path: Path | str | None = None,
        judge_model: str = "gpt-4.1-2025-04-14",
        judge_fn: Callable[[str, str], str] | None = None,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self.judge_model = judge_model
        self._judge_fn = judge_fn
        self._cache: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._cache = loaded
            except json.JSONDecodeError:
                self._cache = {}

    @staticmethod
    def _build_trajectory_str(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")
            if role == "assistant":
                content = strip_think_blocks(content) if content else None
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _cache_key(self, task_id: str, trajectory_str: str, nl_assertions: list[str]) -> str:
        payload = json.dumps(
            {
                "task_id": str(task_id),
                "judge_model": self.judge_model,
                "nl_assertions": nl_assertions,
                "trajectory_str": trajectory_str,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _default_judge_call(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or '{"results": []}'

    def _save(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2) + "\n", encoding="utf-8")

    def evaluate(
        self,
        task_id: str,
        messages: list[dict[str, Any]],
        nl_assertions: list[str] | None,
    ) -> tuple[float, list[dict[str, Any]], bool]:
        if not nl_assertions:
            return 1.0, [], True

        traj_str = self._build_trajectory_str(messages)
        key = self._cache_key(task_id, traj_str, nl_assertions)
        if key in self._cache:
            cached = self._cache[key]
            return float(cached["reward"]), list(cached["checks"]), True

        user_prompt = f"conversation:\n{traj_str}\n\nexpectedOutcomes:\n{nl_assertions}"
        call_fn = self._judge_fn if self._judge_fn is not None else self._default_judge_call
        raw_json = call_fn(NL_ASSERTION_SYSTEM_PROMPT, user_prompt)
        parsed = json.loads(raw_json)
        checks = [
            {
                "nl_assertion": r.get("expectedOutcome", ""),
                "met": bool(r.get("metExpectation", False)),
                "justification": r.get("reasoning", ""),
            }
            for r in parsed.get("results", [])
        ]
        reward = 1.0 if (checks and all(c["met"] for c in checks)) else 0.0
        self._cache[key] = {"reward": reward, "checks": checks}
        self._save()
        return reward, checks, False


def evaluate_dual_metrics(
    task_id: str,
    messages: list[dict[str, Any]],
    termination_reason: str,
    db_reward: float,
    communicate_info: list[str] | None,
    nl_assertions: list[str] | None,
    reward_basis: list[str] | None,
    nl_judge_cache: NLAssertionJudgeCache | None = None,
    run_nl_judge: bool = True,
) -> DualMetricResult:
    """
    [Decision D3] Compute both:
      1. Strict Deterministic `DB + COMMUNICATE` substring score (Hard Gate G3)
      2. Official `tau2` `DB x NL_ASSERTION` score (cached `gpt-4.1-2025-04-14`)
    """
    terminated_cleanly = str(termination_reason).lower() in CLEAN_TERMINATION_REASONS
    if not terminated_cleanly:
        return DualMetricResult(
            terminated_cleanly=False,
            termination_reason=str(termination_reason),
            db_reward=0.0,
            communicate_reward=0.0,
            communicate_checks=[],
            deterministic_success=False,
            nl_reward=0.0 if (run_nl_judge or not nl_assertions) else None,
            nl_checks=[],
            nl_judge_cached=True,
            official_tau2_success=False,
        )

    comm_reward, comm_checks = evaluate_communicate_substring(messages, communicate_info)
    det_success = (db_reward >= 1.0) and (comm_reward >= 1.0)

    basis_upper = {str(b).upper() for b in (reward_basis or ["DB", "NL_ASSERTION"])}
    needs_nl = ("NL_ASSERTION" in basis_upper) and bool(nl_assertions)

    if not needs_nl:
        nl_reward: float | None = 1.0
        nl_checks: list[dict[str, Any]] = []
        nl_cached = True
        official_success: bool | None = db_reward >= 1.0
    elif not run_nl_judge:
        nl_reward = None
        nl_checks = []
        nl_cached = False
        official_success = None
    else:
        judge = nl_judge_cache or NLAssertionJudgeCache()
        nl_reward, nl_checks, nl_cached = judge.evaluate(task_id, messages, nl_assertions)
        official_success = (db_reward >= 1.0) and (nl_reward >= 1.0)

    return DualMetricResult(
        terminated_cleanly=True,
        termination_reason=str(termination_reason),
        db_reward=float(db_reward),
        communicate_reward=float(comm_reward),
        communicate_checks=comm_checks,
        deterministic_success=det_success,
        nl_reward=nl_reward,
        nl_checks=nl_checks,
        nl_judge_cached=nl_cached,
        official_tau2_success=official_success,
    )


def _safe_parse_tool_args(raw_args: str) -> tuple[dict[str, Any], bool]:
    if not raw_args or not raw_args.strip():
        return {}, False
    try:
        parsed = json.loads(raw_args)
        if isinstance(parsed, dict):
            return parsed, False
        return {"_raw": parsed}, True
    except json.JSONDecodeError:
        return {}, True


class OpenAICompatibleAgentClient:
    """
    Streams `/v1/chat/completions` from vLLM (or a mock stream iterable in tests)
    to capture exact client-side TTFT, ITL, token usage, and structured Hermes tool calls.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 512,
        enable_thinking: bool | None = None,
        timeout: float = 120.0,
        client: Any = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        turn_index: int = 0,
        seed: int | None = None,
    ) -> AgentTurnResponse:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if seed is not None:
            kwargs["seed"] = seed
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }

        t_start = time.perf_counter()
        t_first_token: float | None = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_accum: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        prompt_tokens = 0
        completion_tokens = 0
        non_empty_chunks = 0

        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            d_content = getattr(delta, "content", None)
            d_reasoning = getattr(delta, "reasoning_content", None)
            d_tcs = getattr(delta, "tool_calls", None)

            has_token = bool(d_content) or bool(d_reasoning) or bool(d_tcs)
            if has_token:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                non_empty_chunks += 1

            if d_content:
                content_parts.append(d_content)
            if d_reasoning:
                reasoning_parts.append(d_reasoning)
            if d_tcs:
                for tc_delta in d_tcs:
                    idx = int(getattr(tc_delta, "index", 0) or 0)
                    slot = tool_accum.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    tc_id = getattr(tc_delta, "id", None)
                    if tc_id:
                        slot["id"] = tc_id
                    fn = getattr(tc_delta, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            slot["arguments"] += fn.arguments

        t_end = time.perf_counter()
        e2e_ms = (t_end - t_start) * 1000.0
        ttft_ms = ((t_first_token - t_start) * 1000.0) if t_first_token is not None else e2e_ms

        eff_comp_tokens = completion_tokens if completion_tokens > 0 else max(1, non_empty_chunks)
        decode_ms = max(0.0, e2e_ms - ttft_ms)
        itl_ms = (decode_ms / max(1, eff_comp_tokens - 1)) if eff_comp_tokens > 1 else 0.0
        tps = (eff_comp_tokens / (e2e_ms / 1000.0)) if e2e_ms > 0 else 0.0

        parsed_tool_calls: list[dict[str, Any]] = []
        malformed_count = 0
        for idx in sorted(tool_accum.keys()):
            slot = tool_accum[idx]
            args_dict, malformed = _safe_parse_tool_args(slot["arguments"])
            if malformed:
                malformed_count += 1
            parsed_tool_calls.append(
                {
                    "id": slot["id"] or f"call_{turn_index}_{idx}",
                    "name": slot["name"].strip(),
                    "arguments": args_dict,
                }
            )

        raw_content = "".join(content_parts)
        reasoning_str = "".join(reasoning_parts)
        final_content: str | None = raw_content if raw_content.strip() else None

        latency = TurnLatencyRecord(
            turn_index=turn_index,
            ttft_ms=round(ttft_ms, 3),
            e2e_latency_ms=round(e2e_ms, 3),
            itl_ms=round(itl_ms, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_sec=round(tps, 3),
            streamed=True,
        )
        return AgentTurnResponse(
            content=final_content,
            tool_calls=parsed_tool_calls,
            latency=latency,
            reasoning_content=reasoning_str,
            malformed_tool_calls=malformed_count,
            finish_reason=finish_reason,
        )


class BudgetedTurnExecutor:
    """
    Framework-agnostic agent turn runner that applies `ContextBudgetController.allocate()`
    before every model call and logs per-turn allocation telemetry and latency.
    """

    def __init__(
        self,
        budget_controller: ContextBudgetController,
        agent_client: OpenAICompatibleAgentClient,
        system_prompt: str,
        raw_policy: str,
        base_tools: list[dict[str, Any]],
        budget_config: BudgetConfig,
        seed: int | None = None,
    ) -> None:
        self.budget_controller = budget_controller
        self.agent_client = agent_client
        self.system_prompt = system_prompt
        self.raw_policy = raw_policy
        self.base_tools = base_tools
        self.budget_config = budget_config
        self.seed = seed
        self.turn_allocations: list[dict[str, Any]] = []
        self.turn_latencies: list[TurnLatencyRecord] = []
        self.malformed_tool_call_count: int = 0

    def reset_episode_logs(self) -> None:
        self.turn_allocations = []
        self.turn_latencies = []
        self.malformed_tool_call_count = 0

    def step(self, full_openai_history: list[dict[str, Any]]) -> tuple[AgentTurnResponse, AllocationResult]:
        turn_idx = len(self.turn_allocations)
        history_for_budget = strip_boilerplate_greeting(full_openai_history)

        alloc = self.budget_controller.allocate(
            system_prompt=self.system_prompt,
            base_tools=self.base_tools,
            history=history_for_budget,
            config=self.budget_config,
            raw_policy=self.raw_policy,
        )
        messages_to_send = [{"role": "system", "content": alloc.system_prompt}] + alloc.trimmed_history
        resp = self.agent_client.generate(
            messages=messages_to_send,
            tools=alloc.selected_tools,
            turn_index=turn_idx,
            seed=self.seed,
        )
        self.turn_allocations.append({"turn_index": turn_idx, **alloc.telemetry.to_dict()})
        self.turn_latencies.append(resp.latency)
        self.malformed_tool_call_count += resp.malformed_tool_calls
        return resp, alloc


def _tau2_messages_to_openai(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert tau2 `Message` objects into OpenAI chat-completions dict format."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.role
        if role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if getattr(msg, "tool_calls", None):
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            out.append(item)
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(msg, "id", ""),
                    "content": msg.content or "",
                }
            )
        elif role == "user":
            out.append({"role": "user", "content": msg.content or ""})
    return out


def _tau2_messages_to_log_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    """Serialize tau2 `Message` objects cleanly for JSONL storage and compliance checking."""
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.role
        entry: dict[str, Any] = {"role": role, "content": msg.content}
        if role == "assistant" and getattr(msg, "tool_calls", None):
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        elif role == "tool":
            entry["tool_call_id"] = getattr(msg, "id", "")
            entry["error"] = bool(getattr(msg, "error", False))
        serialized.append(entry)
    return serialized


def run_tau2_episode(
    task: Any,
    executor: BudgetedTurnExecutor,
    user_llm: str = "openai/gpt-4o-mini",
    user_llm_args: dict[str, Any] | None = None,
    max_steps: int = 60,
    max_errors: int = 10,
    seed: int = 42,
    timeout: float | None = 300.0,
) -> tuple[list[dict[str, Any]], str, float, list[str], list[str], list[str], bool]:
    """
    Run a single live `tau2-bench@v1.0.1` Retail episode using `BudgetedTurnExecutor`.
    Lazily imports `tau2` so offline unit tests never trigger `litellm`.
    """
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.agent.llm_agent import LLMAgentState
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        SystemMessage,
        ToolCall,
    )
    from tau2.domains.retail.environment import get_environment
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user.user_simulator import UserSimulator

    class _Tau2BudgetedAgent(HalfDuplexAgent[LLMAgentState]):
        def __init__(self, tools: list[Any], domain_policy: str) -> None:
            super().__init__(tools=tools, domain_policy=domain_policy)
            self._seed: int | None = seed

        def set_seed(self, new_seed: int) -> None:
            self._seed = new_seed
            executor.seed = new_seed

        def get_init_state(self, message_history: list[Any] | None = None) -> LLMAgentState:
            return LLMAgentState(
                system_messages=[SystemMessage(role="system", content=executor.system_prompt)],
                messages=list(message_history or []),
            )

        def generate_next_message(
            self, message: Any, state: LLMAgentState
        ) -> tuple[AssistantMessage, LLMAgentState]:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            elif message is not None:
                state.messages.append(message)

            openai_hist = _tau2_messages_to_openai(state.messages)
            resp, _ = executor.step(openai_hist)

            tau2_tool_calls: list[ToolCall] | None = None
            if resp.tool_calls:
                tau2_tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["name"],
                        arguments=tc["arguments"],
                        requestor="assistant",
                    )
                    for tc in resp.tool_calls
                ]

            content = resp.content
            if content is None and not tau2_tool_calls:
                # Defensive non-empty fallback so tau2's `agent_msg.validate()` never crashes a sweep
                content = "I apologize, I encountered an issue processing your request."

            assistant_msg = AssistantMessage(
                role="assistant",
                content=content,
                tool_calls=tau2_tool_calls,
                usage={
                    "prompt_tokens": resp.latency.prompt_tokens,
                    "completion_tokens": resp.latency.completion_tokens,
                },
                generation_time_seconds=resp.latency.e2e_latency_ms / 1000.0,
            )
            state.messages.append(assistant_msg)
            return assistant_msg, state

    executor.reset_episode_logs()
    env = get_environment()
    agent = _Tau2BudgetedAgent(tools=env.get_tools(), domain_policy=env.get_policy())
    user = UserSimulator(
        tools=None,
        instructions=str(task.user_scenario),
        llm=user_llm,
        llm_args={"temperature": 0.0, **(user_llm_args or {})},
    )
    orch = Orchestrator(
        domain="retail",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        solo_mode=False,
        validate_communication=False,
        timeout=timeout,
    )
    sim = orch.run()
    term_reason = str(sim.termination_reason)

    db_reward = 0.0
    if term_reason.lower() in CLEAN_TERMINATION_REASONS:
        env_eval = EnvironmentEvaluator.calculate_reward(
            environment_constructor=get_environment,
            task=task,
            full_trajectory=sim.messages,
            solo_mode=False,
            strict_replay=False,
        )
        db_reward = float(env_eval.db_check.db_reward if env_eval.db_check else env_eval.reward)

    eval_crit = task.evaluation_criteria
    comm_info = list(eval_crit.communicate_info or []) if eval_crit else []
    nl_assertions = list(eval_crit.nl_assertions or []) if eval_crit else []
    reward_basis = [str(b.value if hasattr(b, "value") else b) for b in (eval_crit.reward_basis or [])] if eval_crit else []
    allow_transfer = any(
        getattr(a, "name", "") == "transfer_to_human_agents"
        for a in (eval_crit.actions or [])
    ) if eval_crit else False

    traj_dicts = _tau2_messages_to_log_dicts(sim.messages)
    return traj_dicts, term_reason, db_reward, comm_info, nl_assertions, reward_basis, allow_transfer


def load_task_ids_from_split(
    split_path: Path | str = "configs/retail_task_split.json",
    split: str = "eval",
    explicit_task_ids: list[str] | None = None,
    max_tasks: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    split_data = json.loads(Path(split_path).read_text(encoding="utf-8"))
    excluded = {str(e["task_id"]) for e in split_data.get("excluded_tasks", [])}
    task_meta = split_data.get("task_metadata", {})

    if explicit_task_ids:
        chosen = [str(t).strip() for t in explicit_task_ids if str(t).strip()]
        bad = [t for t in chosen if t in excluded]
        if bad:
            raise ValueError(f"Refusing to run excluded vacuous-pass task IDs: {bad}")
    elif split == "eval":
        chosen = [str(t) for t in split_data["eval_task_ids"]]
    elif split == "train":
        chosen = [str(t) for t in split_data["train_task_ids"]]
    elif split == "all_valid":
        chosen = [str(t) for t in split_data["eval_task_ids"]] + [str(t) for t in split_data["train_task_ids"]]
    else:
        raise ValueError(f"Unknown split '{split}'. Expected one of: eval, train, all_valid.")

    if max_tasks is not None and max_tasks > 0:
        chosen = chosen[:max_tasks]
    return chosen, task_meta


def expand_sweep_grid(
    grid_path: Path | str = "configs/sweep_grid.json",
    grid_name: str = "baseline_pilot",
    budgets_filter: list[int] | None = None,
    ratios_filter: list[float | None] | None = None,
    distractors_filter: list[int] | None = None,
    seeds_filter: list[int] | None = None,
) -> list[SweepCondition]:
    """Expand a named experiment preset from `configs/sweep_grid.json` into `SweepCondition`s."""
    grid = json.loads(Path(grid_path).read_text(encoding="utf-8"))
    out_reserve = int(grid.get("output_reserve", 512))
    conditions: list[SweepCondition] = []

    def _ratio_tag(r: float | None) -> str:
        return "unconstrained" if r is None else f"r{int(round(r * 100))}"

    if grid_name == "baseline_pilot":
        seeds = seeds_filter or [42]
        for seed in seeds:
            cfg = BudgetConfig(
                b_alloc=8192,
                schema_ratio=None,
                output_reserve=out_reserve,
                num_distractors=0,
                distractor_source="none",
                schema_mode="select",
                history_mode="sliding_window",
                policy_mode="pinned",
                seed=seed,
            )
            conditions.append(SweepCondition(f"baseline_8k_unconstrained_s{seed}", cfg, True))
        return conditions

    if grid_name == "rq1_primary":
        sec = grid["rq1_primary"]
        budgets = budgets_filter or sec["budgets"]
        ratios = ratios_filter if ratios_filter is not None else sec["schema_ratios"]
        distractors = distractors_filter if distractors_filter is not None else sec["distractor_counts"]
        seeds = seeds_filter or sec["seeds"]
        for b in budgets:
            for r in ratios:
                for n_dist in distractors:
                    for seed in seeds:
                        cfg = BudgetConfig(
                            b_alloc=int(b),
                            schema_ratio=r,
                            output_reserve=out_reserve,
                            num_distractors=int(n_dist),
                            distractor_source=sec["distractor_source"] if n_dist > 0 else "none",
                            schema_mode=sec["schema_mode"],
                            history_mode=sec["history_mode"],
                            policy_mode=sec["policy_mode"],
                            seed=int(seed),
                        )
                        cid = f"rq1_b{b}_{_ratio_tag(r)}_d{n_dist}_s{seed}"
                        conditions.append(SweepCondition(cid, cfg, True))
        return conditions

    if grid_name == "rq1_schema_compression_ablation":
        sec = grid["rq1_schema_compression_ablation"]
        budgets = budgets_filter or sec["budgets"]
        ratios = ratios_filter if ratios_filter is not None else sec["schema_ratios"]
        distractors = distractors_filter if distractors_filter is not None else sec["distractor_counts"]
        seeds = seeds_filter or [42]
        for b in budgets:
            for r in ratios:
                for n_dist in distractors:
                    for seed in seeds:
                        cfg = BudgetConfig(
                            b_alloc=int(b),
                            schema_ratio=r,
                            output_reserve=out_reserve,
                            num_distractors=int(n_dist),
                            distractor_source="in_dist" if n_dist > 0 else "none",
                            schema_mode=sec["schema_mode"],
                            history_mode="sliding_window",
                            policy_mode="pinned",
                            seed=int(seed),
                        )
                        cid = f"rq1comp_b{b}_{_ratio_tag(r)}_d{n_dist}_s{seed}"
                        conditions.append(SweepCondition(cid, cfg, True))
        return conditions

    if grid_name == "rq1b_policy_retrieval":
        sec = grid["rq1b_policy_retrieval"]
        budgets = budgets_filter or sec["budgets"]
        ratios = ratios_filter if ratios_filter is not None else [0.5]
        seeds = seeds_filter or [42]
        for b in budgets:
            for r in ratios:
                for pmode in sec["policy_modes"]:
                    for seed in seeds:
                        cfg = BudgetConfig(
                            b_alloc=int(b),
                            schema_ratio=r,
                            output_reserve=out_reserve,
                            num_distractors=0,
                            distractor_source="none",
                            schema_mode="select",
                            history_mode="sliding_window",
                            policy_mode=pmode,
                            policy_top_k=int(sec.get("policy_top_k", 3)),
                            seed=int(seed),
                        )
                        cid = f"rq1b_b{b}_{_ratio_tag(r)}_p{pmode}_s{seed}"
                        conditions.append(SweepCondition(cid, cfg, True))
        return conditions

    if grid_name == "rq4_prefix_cache_h4":
        sec = grid["rq4_prefix_cache_h4"]
        budgets = budgets_filter or sec["budgets"]
        ratios = ratios_filter if ratios_filter is not None else [0.5]
        seeds = seeds_filter or [42]
        for b in budgets:
            for r in ratios:
                for hmode in sec["history_modes"]:
                    for pcache in sec["prefix_caching"]:
                        for seed in seeds:
                            cfg = BudgetConfig(
                                b_alloc=int(b),
                                schema_ratio=r,
                                output_reserve=out_reserve,
                                num_distractors=0,
                                distractor_source="none",
                                schema_mode="select",
                                history_mode=hmode,
                                policy_mode="pinned",
                                seed=int(seed),
                            )
                            cid = f"rq4_b{b}_{_ratio_tag(r)}_{hmode}_cache{'on' if pcache else 'off'}_s{seed}"
                            conditions.append(SweepCondition(cid, cfg, bool(pcache)))
        return conditions

    if grid_name == "milestone7_unseen_tools":
        sec = grid["milestone7_unseen_tools"]
        budgets = budgets_filter or sec["budgets"]
        ratios = ratios_filter if ratios_filter is not None else [0.5]
        seeds = seeds_filter or [42]
        n_dist = int(sec["num_distractors"])
        for b in budgets:
            for r in ratios:
                for seed in seeds:
                    cfg = BudgetConfig(
                        b_alloc=int(b),
                        schema_ratio=r,
                        output_reserve=out_reserve,
                        num_distractors=n_dist,
                        distractor_source=sec["distractor_source"],
                        schema_mode="select",
                        history_mode="sliding_window",
                        policy_mode="pinned",
                        seed=int(seed),
                    )
                    cid = f"m7unseen_b{b}_{_ratio_tag(r)}_d{n_dist}_s{seed}"
                    conditions.append(SweepCondition(cid, cfg, True))
        return conditions

    raise ValueError(f"Unknown sweep grid preset '{grid_name}'.")


def evaluate_and_build_episode_record(
    *,
    run_id: str,
    condition: SweepCondition,
    model_name: str,
    tokenizer_name: str,
    task_id: str,
    trajectory: list[dict[str, Any]],
    termination_reason: str,
    db_reward: float,
    communicate_info: list[str] | None,
    nl_assertions: list[str] | None,
    reward_basis: list[str] | None,
    allow_transfer: bool,
    turn_allocations: list[dict[str, Any]],
    turn_latencies: list[TurnLatencyRecord],
    malformed_tool_call_count: int,
    compliance_engine: RetailComplianceEngine,
    nl_judge_cache: NLAssertionJudgeCache | None = None,
    run_nl_judge: bool = True,
    prom_before: dict[str, float] | None = None,
    prom_after: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Combine trajectory, dual rewards, compliance audit, token telemetry, and serving metrics."""
    episode_id = f"{condition.condition_id}__task_{task_id}"
    dual = evaluate_dual_metrics(
        task_id=task_id,
        messages=trajectory,
        termination_reason=termination_reason,
        db_reward=db_reward,
        communicate_info=communicate_info,
        nl_assertions=nl_assertions,
        reward_basis=reward_basis,
        nl_judge_cache=nl_judge_cache,
        run_nl_judge=run_nl_judge,
    )
    comp = compliance_engine.evaluate_episode(
        trajectory=trajectory,
        episode_id=episode_id,
        task_id=task_id,
        task_success=dual.deterministic_success,
        allow_transfer=allow_transfer,
    )
    serving: EpisodeServingMetrics = summarize_episode_serving(
        turns=turn_latencies,
        prom_before=prom_before,
        prom_after=prom_after,
    )

    p_vals = [t["p_tokens"] for t in turn_allocations]
    s_vals = [t["s_tokens"] for t in turn_allocations]
    h_vals = [t["h_tokens"] for t in turn_allocations]
    tot_vals = [t["total_prompt_tokens"] for t in turn_allocations]
    all_within = all(bool(t["within_budget"]) for t in turn_allocations) if turn_allocations else True

    official_sub_silent = (
        bool(dual.official_tau2_success) and not comp.substantive_compliant
        if dual.official_tau2_success is not None
        else None
    )

    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "condition_id": condition.condition_id,
        "model": model_name,
        "tokenizer": tokenizer_name,
        "task_id": str(task_id),
        "seed": condition.budget_config.seed,
        "budget_config": asdict(condition.budget_config),
        "prefix_caching_expected": condition.prefix_caching_expected,
        "dual_metrics": dual.to_dict(),
        "compliance": comp.to_dict(),
        "official_substantive_silent_violation": official_sub_silent,
        "serving": serving.to_dict(),
        "budget_summary": {
            "num_turns": len(turn_allocations),
            "mean_p_tokens": round(statistics.mean(p_vals), 2) if p_vals else 0.0,
            "mean_s_tokens": round(statistics.mean(s_vals), 2) if s_vals else 0.0,
            "mean_h_tokens": round(statistics.mean(h_vals), 2) if h_vals else 0.0,
            "max_total_prompt_tokens": max(tot_vals) if tot_vals else 0,
            "all_turns_within_budget": all_within,
            "total_messages_dropped": sum(int(t["num_messages_dropped"]) for t in turn_allocations),
            "total_truncated_tool_outputs": sum(int(t["truncated_tool_outputs"]) for t in turn_allocations),
            "malformed_tool_calls": malformed_tool_call_count,
        },
        "turn_allocations": turn_allocations,
        "trajectory": trajectory,
    }


def _episode_flat_row(rec: dict[str, Any]) -> dict[str, Any]:
    dm = rec["dual_metrics"]
    cp = rec["compliance"]
    sv = rec["serving"]
    bs = rec["budget_summary"]
    bc = rec["budget_config"]
    return {
        "run_id": rec["run_id"],
        "episode_id": rec["episode_id"],
        "condition_id": rec["condition_id"],
        "model": rec["model"],
        "task_id": rec["task_id"],
        "seed": rec["seed"],
        "b_alloc": bc["b_alloc"],
        "schema_ratio": bc["schema_ratio"],
        "num_distractors": bc["num_distractors"],
        "distractor_source": bc["distractor_source"],
        "schema_mode": bc["schema_mode"],
        "history_mode": bc["history_mode"],
        "policy_mode": bc["policy_mode"],
        "termination_reason": dm["termination_reason"],
        "db_reward": dm["db_reward"],
        "communicate_reward": dm["communicate_reward"],
        "deterministic_success": int(bool(dm["deterministic_success"])),
        "nl_reward": "" if dm["nl_reward"] is None else dm["nl_reward"],
        "official_tau2_success": "" if dm["official_tau2_success"] is None else int(bool(dm["official_tau2_success"])),
        "substantive_compliant": int(bool(cp["substantive_compliant"])),
        "unenforced_compliant": int(bool(cp["unenforced_compliant"])),
        "substantive_silent_violation": int(bool(cp["substantive_silent_violation"])),
        "any_silent_violation": int(bool(cp["any_silent_violation"])),
        "attempted_violation": int(bool(cp["attempted_violation"])),
        "num_turns": bs["num_turns"],
        "mean_p_tokens": bs["mean_p_tokens"],
        "mean_s_tokens": bs["mean_s_tokens"],
        "mean_h_tokens": bs["mean_h_tokens"],
        "max_total_prompt_tokens": bs["max_total_prompt_tokens"],
        "all_turns_within_budget": int(bool(bs["all_turns_within_budget"])),
        "malformed_tool_calls": bs["malformed_tool_calls"],
        "mean_ttft_ms": sv["mean_ttft_ms"],
        "p95_ttft_ms": sv["p95_ttft_ms"],
        "total_agent_latency_ms": sv["total_agent_latency_ms"],
        "prefix_cache_hit_rate_delta": "" if sv["prefix_cache_hit_rate_delta"] is None else sv["prefix_cache_hit_rate_delta"],
        "prefix_cache_hit_rate_gauge": "" if sv["prefix_cache_hit_rate_gauge"] is None else sv["prefix_cache_hit_rate_gauge"],
    }


def aggregate_run_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-condition and overall summaries matching Hard Gates G3 & G5."""
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_cond.setdefault(r["condition_id"], []).append(r)

    def _summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        if n == 0:
            return {"num_episodes": 0}
        det_succ = sum(1 for r in items if r["dual_metrics"]["deterministic_success"])
        off_eval = [r for r in items if r["dual_metrics"]["official_tau2_success"] is not None]
        off_succ = sum(1 for r in off_eval if r["dual_metrics"]["official_tau2_success"])
        db_succ = sum(1 for r in items if r["dual_metrics"]["db_reward"] >= 1.0)
        comm_succ = sum(1 for r in items if r["dual_metrics"]["communicate_reward"] >= 1.0)

        sub_silent = sum(1 for r in items if r["compliance"]["substantive_silent_violation"])
        any_silent = sum(1 for r in items if r["compliance"]["any_silent_violation"])
        attempted = sum(1 for r in items if r["compliance"]["attempted_violation"])
        within_b = sum(1 for r in items if r["budget_summary"]["all_turns_within_budget"])

        ttfts = [r["serving"]["mean_ttft_ms"] for r in items if r["serving"]["num_agent_turns"] > 0]
        p95s = [r["serving"]["p95_ttft_ms"] for r in items if r["serving"]["num_agent_turns"] > 0]
        pc_deltas = [
            r["serving"]["prefix_cache_hit_rate_delta"]
            for r in items
            if r["serving"]["prefix_cache_hit_rate_delta"] is not None
        ]
        pc_gauges = [
            r["serving"]["prefix_cache_hit_rate_gauge"]
            for r in items
            if r["serving"]["prefix_cache_hit_rate_gauge"] is not None
        ]

        det_rate = round(det_succ / n, 4)
        off_rate = round(off_succ / len(off_eval), 4) if off_eval else None
        return {
            "num_episodes": n,
            "deterministic_pass_count": det_succ,
            "deterministic_pass_rate": det_rate,
            "official_tau2_evaluated": len(off_eval),
            "official_tau2_pass_count": off_succ if off_eval else None,
            "official_tau2_pass_rate": off_rate,
            "db_pass_rate": round(db_succ / n, 4),
            "communicate_pass_rate": round(comm_succ / n, 4),
            "passes_15pct_floor_gate": det_rate >= 0.15,
            "substantive_silent_violation_count": sub_silent,
            "substantive_silent_violation_rate_overall": round(sub_silent / n, 4),
            "substantive_silent_violation_rate_given_success": round(sub_silent / det_succ, 4) if det_succ else 0.0,
            "any_silent_violation_count": any_silent,
            "any_silent_violation_rate_given_success": round(any_silent / det_succ, 4) if det_succ else 0.0,
            "attempted_violation_rate": round(attempted / n, 4),
            "all_turns_within_budget_rate": round(within_b / n, 4),
            "mean_p_tokens": round(statistics.mean(r["budget_summary"]["mean_p_tokens"] for r in items), 2),
            "mean_s_tokens": round(statistics.mean(r["budget_summary"]["mean_s_tokens"] for r in items), 2),
            "mean_h_tokens": round(statistics.mean(r["budget_summary"]["mean_h_tokens"] for r in items), 2),
            "mean_ttft_ms": round(statistics.mean(ttfts), 3) if ttfts else 0.0,
            "mean_p95_ttft_ms": round(statistics.mean(p95s), 3) if p95s else 0.0,
            "mean_prefix_cache_hit_rate_delta": round(statistics.mean(pc_deltas), 4) if pc_deltas else None,
            "mean_prefix_cache_hit_rate_gauge": round(statistics.mean(pc_gauges), 4) if pc_gauges else None,
        }

    return {
        "total_episodes": len(records),
        "overall": _summarize_group(records),
        "by_condition": {cid: _summarize_group(items) for cid, items in by_cond.items()},
    }


def load_existing_episodes(jsonl_path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, int]]]:
    records: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if not jsonl_path.exists():
        return records, completed
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            completed.add((str(rec["condition_id"]), str(rec["task_id"]), int(rec["seed"])))
    return records, completed


def write_run_artifacts(out_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_run_records(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = out_dir / "results.csv"
    rows = [_episode_flat_row(r) for r in records]
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def run_evaluation_suite(
    *,
    run_id: str,
    out_dir: Path,
    conditions: list[SweepCondition],
    task_ids: list[str],
    episode_runner: Callable[[SweepCondition, str], dict[str, Any]],
    resume: bool = True,
) -> dict[str, Any]:
    """
    Execute all `(condition, task_id)` pairs with crash-safe JSONL persistence,
    `--resume` skipping, flat CSV export, and summary aggregation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "episodes.jsonl"

    records, completed = load_existing_episodes(jsonl_path) if resume else ([], set())
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()

    with jsonl_path.open("a", encoding="utf-8") as jf:
        for cond in conditions:
            for tid in task_ids:
                key = (cond.condition_id, str(tid), int(cond.budget_config.seed))
                if key in completed:
                    continue
                rec = episode_runner(cond, str(tid))
                jf.write(json.dumps(rec) + "\n")
                jf.flush()
                records.append(rec)
                completed.add(key)

    return write_run_artifacts(out_dir, records)


def _parse_optional_ratios(raw: str | None) -> list[float | None] | None:
    if not raw:
        return None
    out: list[float | None] = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in {"none", "null", "unconstrained"}:
            out.append(None)
        else:
            out.append(float(tok))
    return out


def _parse_int_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Context-Budgeted Evaluation Harness for tau2-bench@v1.0.1 Retail."
    )
    parser.add_argument("--run-id", default="eval_run_01", help="Run identifier under --out-root")
    parser.add_argument("--out-root", default="runs", help="Root directory for saved runs")
    parser.add_argument("--configs-dir", default="configs", help="Directory containing frozen configs")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Served vLLM model or LoRA adapter name")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer ID (defaults to --model)")
    parser.add_argument("--agent-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible vLLM endpoint")
    parser.add_argument("--agent-api-key", default="EMPTY", help="API key for vLLM endpoint")
    parser.add_argument("--user-llm", default="openai/gpt-4o-mini", help="tau2 UserSimulator LLM model")
    parser.add_argument("--split", default="eval", choices=["eval", "train", "all_valid"], help="Task split from retail_task_split.json")
    parser.add_argument("--task-ids", default=None, help="Optional comma-separated explicit task IDs (e.g. 0,1,2)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional cap on number of tasks to run")
    parser.add_argument(
        "--grid",
        default="baseline_pilot",
        choices=[
            "baseline_pilot",
            "rq1_primary",
            "rq1_schema_compression_ablation",
            "rq1b_policy_retrieval",
            "rq4_prefix_cache_h4",
            "milestone7_unseen_tools",
            "custom",
        ],
        help="Named sweep preset from configs/sweep_grid.json or 'custom'",
    )
    # Optional filters / custom single-condition overrides
    parser.add_argument("--budgets", default=None, help="Comma-separated B_alloc filter (e.g. 2048,4096)")
    parser.add_argument("--ratios", default=None, help="Comma-separated schema ratios (e.g. 0.75,0.5,0.25,null)")
    parser.add_argument("--distractors", default=None, help="Comma-separated distractor counts (e.g. 0,10,23)")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds (e.g. 42)")
    parser.add_argument("--distractor-source", default="in_dist", choices=["none", "in_dist", "unseen"])
    parser.add_argument("--schema-mode", default="select", choices=["select", "compact_select"])
    parser.add_argument("--history-mode", default="sliding_window", choices=["sliding_window", "summarize_dropped"])
    parser.add_argument("--policy-mode", default="pinned", choices=["pinned", "retrieved"])
    parser.add_argument("--policy-top-k", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", default=None, choices=["true", "false"], help="Optional Qwen3 chat_template thinking toggle")
    parser.add_argument("--skip-nl-judge", action="store_true", help="Skip gpt-4.1 nl_assertions LLM judge (deterministic-only)")
    parser.add_argument("--nl-judge-model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--nl-judge-cache", default="runs/nl_judge_cache.json")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing episodes.jsonl instead of resuming")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configs_dir = Path(args.configs_dir)
    out_dir = Path(args.out_root) / args.run_id

    explicit_ids = [x.strip() for x in args.task_ids.split(",")] if args.task_ids else None
    task_ids, _ = load_task_ids_from_split(
        split_path=configs_dir / "retail_task_split.json",
        split=args.split,
        explicit_task_ids=explicit_ids,
        max_tasks=args.max_tasks,
    )

    budgets = _parse_int_list(args.budgets)
    ratios = _parse_optional_ratios(args.ratios)
    distractors = _parse_int_list(args.distractors)
    seeds = _parse_int_list(args.seeds)

    if args.grid == "custom":
        b_list = budgets or [4096]
        r_list = ratios if ratios is not None else [0.5]
        d_list = distractors if distractors is not None else [0]
        s_list = seeds or [42]
        conditions = [
            SweepCondition(
                condition_id=f"custom_b{b}_{'unconstrained' if r is None else f'r{int(round(r * 100))}'}_d{d}_s{s}",
                budget_config=BudgetConfig(
                    b_alloc=b,
                    schema_ratio=r,
                    num_distractors=d,
                    distractor_source=args.distractor_source if d > 0 else "none",
                    schema_mode=args.schema_mode,
                    history_mode=args.history_mode,
                    policy_mode=args.policy_mode,
                    policy_top_k=args.policy_top_k,
                    seed=s,
                ),
            )
            for b in b_list
            for r in r_list
            for d in d_list
            for s in s_list
        ]
    else:
        conditions = expand_sweep_grid(
            grid_path=configs_dir / "sweep_grid.json",
            grid_name=args.grid,
            budgets_filter=budgets,
            ratios_filter=ratios,
            distractors_filter=distractors,
            seeds_filter=seeds,
        )

    tokenizer_name = args.tokenizer or args.model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    system_prompt = (configs_dir / "retail_system_prompt.txt").read_text(encoding="utf-8")
    raw_policy = (configs_dir / "retail_policy.md").read_text(encoding="utf-8")
    base_tools = json.loads((configs_dir / "retail_tools_openai.json").read_text(encoding="utf-8"))
    distractor_pool = DistractorPool.from_configs_dir(configs_dir)

    controller = ContextBudgetController(tokenizer=tokenizer, distractor_pool=distractor_pool)
    compliance_engine = RetailComplianceEngine()
    scraper = VLLMMetricsScraper(args.agent_base_url)
    nl_cache = NLAssertionJudgeCache(cache_path=args.nl_judge_cache, judge_model=args.nl_judge_model)

    enable_thinking: bool | None = None
    if args.enable_thinking is not None:
        enable_thinking = args.enable_thinking == "true"

    from tau2.domains.retail.environment import get_tasks

    all_tasks = {str(t.id): t for t in get_tasks(task_split_name=None)}

    def _run_one(cond: SweepCondition, tid: str) -> dict[str, Any]:
        task = all_tasks[tid]
        client = OpenAICompatibleAgentClient(
            model=args.model,
            base_url=args.agent_base_url,
            api_key=args.agent_api_key,
            temperature=args.temperature,
            max_tokens=cond.budget_config.output_reserve,
            enable_thinking=enable_thinking,
        )
        executor = BudgetedTurnExecutor(
            budget_controller=controller,
            agent_client=client,
            system_prompt=system_prompt,
            raw_policy=raw_policy,
            base_tools=base_tools,
            budget_config=cond.budget_config,
            seed=cond.budget_config.seed,
        )
        prom_before = scraper.snapshot()
        traj, term, db_rew, comm_info, nl_asserts, rew_basis, allow_xfer = run_tau2_episode(
            task=task,
            executor=executor,
            user_llm=args.user_llm,
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            seed=cond.budget_config.seed,
        )
        prom_after = scraper.snapshot()
        rec = evaluate_and_build_episode_record(
            run_id=args.run_id,
            condition=cond,
            model_name=args.model,
            tokenizer_name=tokenizer_name,
            task_id=tid,
            trajectory=traj,
            termination_reason=term,
            db_reward=db_rew,
            communicate_info=comm_info,
            nl_assertions=nl_asserts,
            reward_basis=rew_basis,
            allow_transfer=allow_xfer,
            turn_allocations=executor.turn_allocations,
            turn_latencies=executor.turn_latencies,
            malformed_tool_call_count=executor.malformed_tool_call_count,
            compliance_engine=compliance_engine,
            nl_judge_cache=nl_cache,
            run_nl_judge=not args.skip_nl_judge,
            prom_before=prom_before,
            prom_after=prom_after,
        )
        print(
            f"[{cond.condition_id}] task={tid} term={term} "
            f"det={rec['dual_metrics']['deterministic_success']} "
            f"off={rec['dual_metrics']['official_tau2_success']} "
            f"silent={rec['compliance']['substantive_silent_violation']} "
            f"ttft={rec['serving']['mean_ttft_ms']}ms",
            flush=True,
        )
        return rec

    summary = run_evaluation_suite(
        run_id=args.run_id,
        out_dir=out_dir,
        conditions=conditions,
        task_ids=task_ids,
        episode_runner=_run_one,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
