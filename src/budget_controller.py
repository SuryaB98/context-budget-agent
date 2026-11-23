"""
[CBA-9] Per-Component Context Budget Controller (`S`, `P`, `H` + Distractors).

Enforces exact Qwen3 BPE serialized token limits across:
  - Policy Context (`P`): pinned in RQ1 (`1,492` tokens) or BM25-retrieved in RQ1b
  - Tool Schemas (`S <= B_S`): 16 retail tools + `N` distractors, BM25-ranked, canonical-sorted
  - Interaction History (`H <= B_H`): atomic `(assistant_tool_call, tool_response)` blocks,
    preserving Turn 0 anchor + newest tail, with middle-truncation safeguard and optional H4
    prefix-cache-breaking summarization of dropped turns.

Zero `tau2` or `litellm` imports so allocation runs offline in milliseconds.
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from rank_bm25 import BM25Okapi


TRUNCATION_MARKER = "\n...[truncated for context budget]...\n"


def count_chat_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Count exact serialized BPE tokens safely across transformers 4.x and 5.x."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools if tools else None,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError(f"Expected str from apply_chat_template(tokenize=False), got {type(rendered).__name__}")
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    count = len(token_ids)
    if messages and count <= 5:
        raise RuntimeError(f"Unexpectedly low token count ({count}) for non-empty messages.")
    return count


def _tokenize_bm25(text: str) -> list[str]:
    """Split identifiers (snake_case / camelCase) and prose into lowercase BM25 tokens."""
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text.replace("_", " "))
    return re.findall(r"[a-z0-9]+", expanded.lower())


def _tool_to_search_tokens(tool: dict[str, Any]) -> list[str]:
    fn = tool.get("function", tool)
    parts: list[str] = [str(fn.get("name", "")), str(fn.get("description", ""))]
    params = fn.get("parameters", {})
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    for pname, pspec in props.items():
        parts.append(str(pname))
        if isinstance(pspec, dict) and pspec.get("description"):
            parts.append(str(pspec["description"]))
    return _tokenize_bm25(" ".join(parts))


def compact_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Strip verbose parameter descriptions while preserving name, types, enums, and required keys."""
    out = copy.deepcopy(tool)
    fn = out.get("function", out)
    desc = str(fn.get("description", "")).strip()
    if desc:
        first_sentence = desc.split(". ")[0].rstrip(".") + "."
        fn["description"] = first_sentence[:120]

    def _clean_prop(spec: Any) -> Any:
        if not isinstance(spec, dict):
            return spec
        cleaned: dict[str, Any] = {}
        for key in ("type", "enum"):
            if key in spec:
                cleaned[key] = spec[key]
        if "items" in spec and isinstance(spec["items"], dict):
            cleaned["items"] = _clean_prop(spec["items"])
        return cleaned

    params = fn.get("parameters")
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        params["properties"] = {
            k: _clean_prop(v) for k, v in params["properties"].items()
        }
    return out


def group_atomic_blocks(history: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """
    Partition messages into indivisible blocks so an assistant `tool_calls` message
    is never separated from its subsequent `role == "tool"` response messages.
    """
    blocks: list[list[dict[str, Any]]] = []
    i = 0
    n = len(history)
    while i < n:
        msg = copy.deepcopy(history[i])
        role = msg.get("role")
        if role == "tool":
            # Drop orphaned leading tool output defensively
            i += 1
            continue
        if role == "assistant" and msg.get("tool_calls"):
            block = [msg]
            j = i + 1
            while j < n and history[j].get("role") == "tool":
                block.append(copy.deepcopy(history[j]))
                j += 1
            blocks.append(block)
            i = j
        else:
            blocks.append([msg])
            i += 1
    return blocks


def middle_truncate_text(text: str, max_chars: int) -> str:
    """Truncate long text keeping head (60%) and tail (40%) separated by an explicit marker."""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(TRUNCATION_MARKER) + 8:
        return text[:max(1, max_chars)]
    avail = max_chars - len(TRUNCATION_MARKER)
    head = int(avail * 0.6)
    tail = avail - head
    if tail <= 0:
        return text[:head] + TRUNCATION_MARKER.strip()
    return text[:head] + TRUNCATION_MARKER + text[-tail:]


def _extract_user_query(history: list[dict[str, Any]]) -> str:
    user_texts = [
        str(m.get("content", "")).strip()
        for m in history
        if m.get("role") == "user" and m.get("content")
    ]
    if not user_texts:
        return "retail order customer request"
    if len(user_texts) == 1:
        return user_texts[0]
    # Weight initial goal + latest user request (2x)
    return f"{user_texts[0]} {user_texts[-1]} {user_texts[-1]}"


def retrieve_policy_prompt(
    full_system_prompt: str,
    raw_policy: str | None,
    query: str,
    top_k: int = 3,
) -> str:
    """
    [RQ1b] Split retail policy by markdown section headings, retain the preamble
    plus the top-k BM25-matched sections in original order, and preserve wrapper tags.
    """
    instr_match = re.search(r"<instructions>(.*?)</instructions>", full_system_prompt, flags=re.S)
    policy_match = re.search(r"<policy>(.*?)</policy>", full_system_prompt, flags=re.S)

    instructions_body = instr_match.group(1).strip() if instr_match else ""
    policy_body = (raw_policy if raw_policy is not None else (policy_match.group(1) if policy_match else full_system_prompt)).strip()

    sections = [s.strip() for s in re.split(r"(?=^#{2,3}\s+)", policy_body, flags=re.M) if s.strip()]
    if len(sections) <= 1 + top_k:
        selected_policy = policy_body
    else:
        preamble = sections[0]
        candidates = sections[1:]
        corpus = [_tokenize_bm25(sec) for sec in candidates]
        q_tokens = _tokenize_bm25(query)
        if q_tokens and any(corpus):
            scores = BM25Okapi(corpus).get_scores(q_tokens)
            ranked = sorted(range(len(candidates)), key=lambda idx: (-float(scores[idx]), idx))
            chosen_indices = sorted(ranked[:top_k])
        else:
            chosen_indices = list(range(min(top_k, len(candidates))))
        selected_policy = "\n\n".join([preamble] + [candidates[idx] for idx in chosen_indices])

    if instructions_body:
        return f"<instructions>\n{instructions_body}\n</instructions>\n<policy>\n{selected_policy}\n</policy>"
    return selected_policy


def _default_extractive_summary(dropped_blocks: list[list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for blk in dropped_blocks:
        for msg in blk:
            role = msg.get("role", "unknown")
            if role == "assistant" and msg.get("tool_calls"):
                calls = [
                    f"{tc.get('function', {}).get('name', '?')}({tc.get('function', {}).get('arguments', '')})"
                    for tc in msg["tool_calls"]
                ]
                lines.append(f"Called: {', '.join(calls)}")
            elif role == "tool":
                snippet = str(msg.get("content", "")).replace("\n", " ")[:100]
                lines.append(f"Tool result: {snippet}")
            else:
                snippet = str(msg.get("content", "")).replace("\n", " ")[:120]
                if snippet:
                    lines.append(f"{role}: {snippet}")
    return " | ".join(lines) if lines else "Earlier turns omitted."


@dataclass(frozen=True)
class BudgetConfig:
    b_alloc: int = 4096
    schema_ratio: float | None = 0.50
    output_reserve: int = 512
    num_distractors: int = 0
    distractor_source: Literal["none", "in_dist", "unseen"] = "in_dist"
    schema_mode: Literal["select", "compact_select"] = "select"
    history_mode: Literal["sliding_window", "summarize_dropped"] = "sliding_window"
    policy_mode: Literal["pinned", "retrieved"] = "pinned"
    policy_top_k: int = 3
    canonical_sort_tools: bool = True
    seed: int = 42


@dataclass
class AllocationTelemetry:
    b_alloc: int
    b_s: int | None
    b_h: int | None
    p_tokens: int
    s_tokens: int
    h_tokens: int
    total_prompt_tokens: int
    base_framing_tokens: int
    num_tools_candidate: int
    num_tools_kept: int
    selected_tool_names: list[str]
    dropped_tool_names: list[str]
    num_messages_in: int
    num_messages_kept: int
    num_messages_dropped: int
    truncated_tool_outputs: int
    schema_mode: str
    history_mode: str
    policy_mode: str
    within_budget: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AllocationResult:
    system_prompt: str
    selected_tools: list[dict[str, Any]]
    trimmed_history: list[dict[str, Any]]
    telemetry: AllocationTelemetry


class DistractorPool:
    """Loads frozen in-distribution and unseen distractor OpenAI tool schemas."""

    def __init__(
        self,
        in_dist_path: Path | None = None,
        unseen_path: Path | None = None,
    ) -> None:
        self.in_dist: list[dict[str, Any]] = (
            json.loads(in_dist_path.read_text(encoding="utf-8"))
            if in_dist_path and in_dist_path.exists()
            else []
        )
        self.unseen: list[dict[str, Any]] = (
            json.loads(unseen_path.read_text(encoding="utf-8"))
            if unseen_path and unseen_path.exists()
            else []
        )

    @classmethod
    def from_configs_dir(cls, configs_dir: Path | str = "configs") -> "DistractorPool":
        root = Path(configs_dir)
        return cls(
            in_dist_path=root / "distractor_tools_in_dist.json",
            unseen_path=root / "distractor_tools_unseen.json",
        )

    def sample(
        self,
        n: int,
        source: Literal["none", "in_dist", "unseen"] = "in_dist",
        seed: int = 42,
        exclude_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if n <= 0 or source == "none":
            return []
        pool = self.in_dist if source == "in_dist" else self.unseen
        excluded = exclude_names or set()
        candidates = [
            copy.deepcopy(t)
            for t in pool
            if t.get("function", {}).get("name") not in excluded
        ]
        candidates.sort(key=lambda t: t["function"]["name"])
        if n >= len(candidates):
            return candidates
        rng = random.Random(seed)
        return rng.sample(candidates, n)


class ContextBudgetController:
    """Deterministic per-component BPE token budget controller for multi-turn tool agents."""

    def __init__(
        self,
        tokenizer: Any,
        distractor_pool: DistractorPool | None = None,
        summarizer_fn: Callable[[list[list[dict[str, Any]]]], str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.distractor_pool = distractor_pool
        self.summarizer_fn = summarizer_fn
        self._minimal_user = [{"role": "user", "content": "Hi"}]
        self._base_framing = count_chat_tokens(self.tokenizer, self._minimal_user, tools=None)

    def _measure_p_tokens(self, system_prompt: str) -> tuple[str, int, int]:
        sys_msg = [{"role": "system", "content": system_prompt}]
        with_p = count_chat_tokens(self.tokenizer, sys_msg + self._minimal_user, tools=None)
        return system_prompt, with_p, with_p - self._base_framing

    def _measure_s_tokens(self, sys_msg: list[dict[str, Any]], with_p: int, tools: list[dict[str, Any]]) -> int:
        if not tools:
            return 0
        with_ps = count_chat_tokens(self.tokenizer, sys_msg + self._minimal_user, tools=tools)
        return max(0, with_ps - with_p)

    def _rank_tools(
        self,
        tools: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        if not tools:
            return []
        corpus = [_tool_to_search_tokens(t) for t in tools]
        q_tokens = _tokenize_bm25(query)
        if q_tokens and any(corpus):
            scores = BM25Okapi(corpus).get_scores(q_tokens)
            indexed = sorted(
                range(len(tools)),
                key=lambda idx: (-float(scores[idx]), tools[idx]["function"]["name"]),
            )
            return [tools[idx] for idx in indexed]
        return sorted(tools, key=lambda t: t["function"]["name"])

    def _select_tools(
        self,
        sys_msg: list[dict[str, Any]],
        with_p: int,
        candidate_tools: list[dict[str, Any]],
        query: str,
        b_s: int | None,
        schema_mode: str,
        canonical_sort: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        prepared = [
            compact_tool_schema(t) if schema_mode == "compact_select" else copy.deepcopy(t)
            for t in candidate_tools
        ]
        if b_s is None:
            selected = sorted(prepared, key=lambda t: t["function"]["name"]) if canonical_sort else prepared
            return selected, self._measure_s_tokens(sys_msg, with_p, selected)

        ranked = self._rank_tools(prepared, query)
        chosen_ranked: list[dict[str, Any]] = []
        for tool in ranked:
            trial = chosen_ranked + [tool]
            if self._measure_s_tokens(sys_msg, with_p, trial) <= b_s:
                chosen_ranked.append(tool)

        if canonical_sort and chosen_ranked:
            selected = sorted(chosen_ranked, key=lambda t: t["function"]["name"])
            s_tok = self._measure_s_tokens(sys_msg, with_p, selected)
            # Guard against 1-2 token order-dependent boundary jitter
            if s_tok > b_s:
                selected_names = {t["function"]["name"] for t in selected}
                for drop_cand in reversed(chosen_ranked):
                    if s_tok <= b_s:
                        break
                    selected_names.discard(drop_cand["function"]["name"])
                    selected = [t for t in selected if t["function"]["name"] in selected_names]
                    s_tok = self._measure_s_tokens(sys_msg, with_p, selected)
            return selected, s_tok

        s_tok = self._measure_s_tokens(sys_msg, with_p, chosen_ranked)
        return chosen_ranked, s_tok

    def _shrink_message_content_to_fit(
        self,
        msg: dict[str, Any],
        prefix_msgs: list[dict[str, Any]],
        suffix_msgs: list[dict[str, Any]],
        budget: int,
        h_fn: Callable[[list[dict[str, Any]]], int],
    ) -> bool:
        """Binary-search middle truncation on `msg['content']` until `h_fn(...) <= budget`."""
        orig = str(msg.get("content") or "")
        if not orig:
            return False
        lo, hi = 24, len(orig)
        best: str | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            msg["content"] = middle_truncate_text(orig, mid)
            if h_fn(prefix_msgs + [msg] + suffix_msgs) <= budget:
                best = msg["content"]
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            msg["content"] = best
            return True
        msg["content"] = middle_truncate_text(orig, 24)
        return h_fn(prefix_msgs + [msg] + suffix_msgs) <= budget

    def _trim_history(
        self,
        sys_msg: list[dict[str, Any]],
        selected_tools: list[dict[str, Any]],
        history: list[dict[str, Any]],
        b_h: int | None,
        history_mode: str,
    ) -> tuple[list[dict[str, Any]], int, int]:
        blocks = group_atomic_blocks(history)
        if not blocks:
            return [], 0, 0

        p_plus_s_only = count_chat_tokens(self.tokenizer, sys_msg, tools=selected_tools)

        def _h_tokens(msgs: list[dict[str, Any]]) -> int:
            if not msgs:
                return 0
            return max(0, count_chat_tokens(self.tokenizer, sys_msg + msgs, tools=selected_tools) - p_plus_s_only)

        flat_all = [m for b in blocks for m in b]
        if b_h is None:
            return flat_all, _h_tokens(flat_all), 0

        if _h_tokens(flat_all) <= b_h:
            return flat_all, _h_tokens(flat_all), 0

        truncated_count = 0
        anchor = copy.deepcopy(blocks[0])

        if len(blocks) == 1:
            if _h_tokens(anchor) > b_h:
                for msg in sorted(anchor, key=lambda m: len(str(m.get("content") or "")), reverse=True):
                    if _h_tokens(anchor) <= b_h:
                        break
                    idx = anchor.index(msg)
                    if self._shrink_message_content_to_fit(msg, anchor[:idx], anchor[idx + 1 :], b_h, _h_tokens):
                        truncated_count += 1
            return anchor, _h_tokens(anchor), truncated_count

        # Ensure anchor + newest block fits inside b_h
        newest = copy.deepcopy(blocks[-1])
        if _h_tokens(anchor + newest) > b_h:
            # Cap oversized anchor first to at most ~1/3 of b_h
            anchor_budget = max(64, b_h // 3)
            if _h_tokens(anchor) > anchor_budget:
                for msg in sorted(anchor, key=lambda m: len(str(m.get("content") or "")), reverse=True):
                    if _h_tokens(anchor) <= anchor_budget:
                        break
                    idx = anchor.index(msg)
                    if self._shrink_message_content_to_fit(msg, anchor[:idx], anchor[idx + 1 :], anchor_budget, _h_tokens):
                        truncated_count += 1

            # Middle-truncate longest messages (tool outputs first) inside newest block
            if _h_tokens(anchor + newest) > b_h:
                candidates = sorted(
                    range(len(newest)),
                    key=lambda idx: (0 if newest[idx].get("role") == "tool" else 1, -len(str(newest[idx].get("content") or ""))),
                )
                for idx in candidates:
                    if _h_tokens(anchor + newest) <= b_h:
                        break
                    if newest[idx].get("content"):
                        if self._shrink_message_content_to_fit(
                            newest[idx],
                            anchor + newest[:idx],
                            newest[idx + 1 :],
                            b_h,
                            _h_tokens,
                        ):
                            truncated_count += 1
                        elif _h_tokens(anchor + newest) > b_h:
                            truncated_count += 1

        # Greedily add preceding tail blocks backwards while preserving anchor
        kept_tail_blocks: list[list[dict[str, Any]]] = [newest]
        dropped_blocks: list[list[dict[str, Any]]] = []
        stopped = False
        for idx in range(len(blocks) - 2, 0, -1):
            blk = copy.deepcopy(blocks[idx])
            if not stopped:
                trial_msgs = anchor + blk + [m for b in kept_tail_blocks for m in b]
                if _h_tokens(trial_msgs) <= b_h:
                    kept_tail_blocks.insert(0, blk)
                    continue
                stopped = True
            dropped_blocks.insert(0, blk)

        if history_mode == "summarize_dropped" and dropped_blocks:
            summary_raw = (
                self.summarizer_fn(dropped_blocks)
                if self.summarizer_fn is not None
                else _default_extractive_summary(dropped_blocks)
            )
            summary_msg: dict[str, Any] = {
                "role": "user",
                "content": f"[Earlier conversation summary] {summary_raw}",
            }
            # Drop older non-newest tail blocks if needed to make room for summary
            while len(kept_tail_blocks) > 1 and _h_tokens(anchor + [summary_msg] + [m for b in kept_tail_blocks for m in b]) > b_h:
                kept_tail_blocks.pop(0)

            tail_flat = [m for b in kept_tail_blocks for m in b]
            if _h_tokens(anchor + [summary_msg] + tail_flat) > b_h:
                self._shrink_message_content_to_fit(summary_msg, anchor, tail_flat, b_h, _h_tokens)

            if _h_tokens(anchor + [summary_msg] + tail_flat) <= b_h:
                final_msgs = anchor + [summary_msg] + tail_flat
                return final_msgs, _h_tokens(final_msgs), truncated_count

        final_msgs = anchor + [m for b in kept_tail_blocks for m in b]
        return final_msgs, _h_tokens(final_msgs), truncated_count

    def allocate(
        self,
        system_prompt: str,
        base_tools: list[dict[str, Any]],
        history: list[dict[str, Any]],
        config: BudgetConfig,
        raw_policy: str | None = None,
    ) -> AllocationResult:
        query = _extract_user_query(history)

        # 1. Policy stage (P)
        effective_sys = (
            retrieve_policy_prompt(system_prompt, raw_policy, query, top_k=config.policy_top_k)
            if config.policy_mode == "retrieved"
            else system_prompt
        )
        effective_sys, with_p, p_tokens = self._measure_p_tokens(effective_sys)
        sys_msg = [{"role": "system", "content": effective_sys}]

        # 2. Candidate tools + distractor injection
        base_names = {t["function"]["name"] for t in base_tools}
        distractors: list[dict[str, Any]] = []
        if config.num_distractors > 0 and config.distractor_source != "none" and self.distractor_pool is not None:
            distractors = self.distractor_pool.sample(
                n=config.num_distractors,
                source=config.distractor_source,
                seed=config.seed,
                exclude_names=base_names,
            )
        candidate_tools = [copy.deepcopy(t) for t in base_tools] + distractors

        # 3. Budget split (Decision D1: B_alloc across S + H)
        if config.schema_ratio is None:
            b_s: int | None = None
            b_h: int | None = None
        else:
            b_s = int(config.b_alloc * config.schema_ratio)
            b_h = config.b_alloc - b_s

        # 4. Schema allocation (S <= B_S)
        selected_tools, s_tokens = self._select_tools(
            sys_msg=sys_msg,
            with_p=with_p,
            candidate_tools=candidate_tools,
            query=query,
            b_s=b_s,
            schema_mode=config.schema_mode,
            canonical_sort=config.canonical_sort_tools,
        )

        # 5. Atomic history allocation (H <= B_H)
        trimmed_history, h_tokens, truncated_outputs = self._trim_history(
            sys_msg=sys_msg,
            selected_tools=selected_tools,
            history=history,
            b_h=b_h,
            history_mode=config.history_mode,
        )

        total_prompt = count_chat_tokens(
            self.tokenizer,
            sys_msg + (trimmed_history if trimmed_history else self._minimal_user),
            tools=selected_tools,
        )

        cand_names = [t["function"]["name"] for t in candidate_tools]
        kept_names = [t["function"]["name"] for t in selected_tools]
        kept_set = set(kept_names)
        dropped_names = [n for n in cand_names if n not in kept_set]

        within_budget = (
            True
            if config.schema_ratio is None
            else (
                s_tokens <= b_s  # type: ignore[operator]
                and h_tokens <= b_h  # type: ignore[operator]
                and total_prompt <= (p_tokens + self._base_framing + config.b_alloc)
            )
        )

        telemetry = AllocationTelemetry(
            b_alloc=config.b_alloc,
            b_s=b_s,
            b_h=b_h,
            p_tokens=p_tokens,
            s_tokens=s_tokens,
            h_tokens=h_tokens,
            total_prompt_tokens=total_prompt,
            base_framing_tokens=self._base_framing,
            num_tools_candidate=len(candidate_tools),
            num_tools_kept=len(selected_tools),
            selected_tool_names=kept_names,
            dropped_tool_names=dropped_names,
            num_messages_in=len(history),
            num_messages_kept=len(trimmed_history),
            num_messages_dropped=max(0, len(history) - len(trimmed_history)),
            truncated_tool_outputs=truncated_outputs,
            schema_mode=config.schema_mode,
            history_mode=config.history_mode,
            policy_mode=config.policy_mode,
            within_budget=within_budget,
        )

        return AllocationResult(
            system_prompt=effective_sys,
            selected_tools=selected_tools,
            trimmed_history=trimmed_history,
            telemetry=telemetry,
        )
