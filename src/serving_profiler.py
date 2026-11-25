"""
[Milestone 5 / RQ4] vLLM Serving & PagedAttention Prefix-Cache Profiler.

Records:
  1. Client-side streaming latency per agent turn:
     - Time-to-First-Token (`ttft_ms`)
     - End-to-end turn latency (`e2e_latency_ms`)
     - Inter-token latency (`itl_ms`) and throughput (`tokens_per_sec`)
  2. Server-side vLLM Prometheus `/metrics` snapshots before and after each episode:
     - Prefix cache hit rate (both exact episode counter delta `hits / queries` and gauge)
     - KV-cache utilization, prompt/generation token totals, and server-side TTFT mean

Zero `tau2` or GPU dependencies so profiling and unit tests run offline.
"""

from __future__ import annotations

import re
import statistics
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any


_PROM_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[Nn][Aa][Nn]|[+-]?[Ii][Nn][Ff])"
)

# Canonical metric keys -> candidate Prometheus metric names across vLLM v0.6–v0.9 (V0 & V1 engines)
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "gpu_prefix_cache_hit_rate": (
        "vllm:gpu_prefix_cache_hit_rate",
        "vllm:prefix_cache_hit_rate",
    ),
    "prefix_cache_queries_total": (
        "vllm:gpu_prefix_cache_queries_total",
        "vllm:prefix_cache_queries_total",
    ),
    "prefix_cache_hits_total": (
        "vllm:gpu_prefix_cache_hits_total",
        "vllm:prefix_cache_hits_total",
    ),
    "gpu_cache_usage_perc": (
        "vllm:gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc",
    ),
    "prompt_tokens_total": (
        "vllm:prompt_tokens_total",
    ),
    "generation_tokens_total": (
        "vllm:generation_tokens_total",
    ),
    "ttft_seconds_sum": (
        "vllm:time_to_first_token_seconds_sum",
    ),
    "ttft_seconds_count": (
        "vllm:time_to_first_token_seconds_count",
    ),
}


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """
    Parse Prometheus exposition text from vLLM `/metrics` into canonical scalar keys.
    If a metric appears multiple times with different label sets, values are summed
    for counters and averaged for gauges (`*_rate`, `*_perc`).
    """
    raw_values: dict[str, list[float]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _PROM_LINE_RE.match(stripped)
        if not match:
            continue
        name, val_str = match.group(1), match.group(2)
        try:
            val = float(val_str)
        except ValueError:
            continue
        if val != val:  # NaN guard
            continue
        raw_values.setdefault(name, []).append(val)

    parsed: dict[str, float] = {}
    for canonical, candidates in _METRIC_ALIASES.items():
        for prom_name in candidates:
            if prom_name in raw_values:
                vals = raw_values[prom_name]
                if canonical.endswith(("_rate", "_perc")):
                    parsed[canonical] = sum(vals) / len(vals)
                else:
                    parsed[canonical] = sum(vals)
                break
    return parsed


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


@dataclass
class TurnLatencyRecord:
    turn_index: int
    ttft_ms: float
    e2e_latency_ms: float
    itl_ms: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float
    streamed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeServingMetrics:
    num_agent_turns: int = 0
    mean_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    max_ttft_ms: float = 0.0
    mean_e2e_latency_ms: float = 0.0
    total_agent_latency_ms: float = 0.0
    mean_itl_ms: float = 0.0
    mean_tokens_per_sec: float = 0.0
    total_prompt_tokens_server: int = 0
    total_completion_tokens_server: int = 0
    prefix_cache_hit_rate_delta: float | None = None
    prefix_cache_hit_rate_gauge: float | None = None
    delta_prefix_cache_queries: float | None = None
    delta_prefix_cache_hits: float | None = None
    server_mean_ttft_ms: float | None = None
    gpu_cache_usage_perc: float | None = None
    turns: list[TurnLatencyRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["turns"] = [t.to_dict() for t in self.turns]
        return data


class VLLMMetricsScraper:
    """Fetches and diffs vLLM Prometheus `/metrics` around each evaluation episode."""

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self.metrics_url = f"{root}/metrics"
        self.timeout = timeout

    def snapshot(self) -> dict[str, float] | None:
        try:
            req = urllib.request.Request(self.metrics_url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.getcode() != 200:
                    return None
                body = resp.read().decode("utf-8", errors="replace")
            return parse_prometheus_metrics(body)
        except Exception:
            return None

    @staticmethod
    def compute_episode_delta(
        before: dict[str, float] | None,
        after: dict[str, float] | None,
    ) -> dict[str, float | None]:
        out: dict[str, float | None] = {
            "prefix_cache_hit_rate_delta": None,
            "prefix_cache_hit_rate_gauge": None,
            "delta_prefix_cache_queries": None,
            "delta_prefix_cache_hits": None,
            "server_mean_ttft_ms": None,
            "gpu_cache_usage_perc": None,
        }
        if not after:
            return out

        if "gpu_prefix_cache_hit_rate" in after:
            out["prefix_cache_hit_rate_gauge"] = round(after["gpu_prefix_cache_hit_rate"], 6)
        if "gpu_cache_usage_perc" in after:
            out["gpu_cache_usage_perc"] = round(after["gpu_cache_usage_perc"], 6)

        if before:
            if "prefix_cache_queries_total" in before and "prefix_cache_queries_total" in after:
                dq = max(0.0, after["prefix_cache_queries_total"] - before["prefix_cache_queries_total"])
                dh = max(0.0, after.get("prefix_cache_hits_total", 0.0) - before.get("prefix_cache_hits_total", 0.0))
                out["delta_prefix_cache_queries"] = dq
                out["delta_prefix_cache_hits"] = dh
                if dq > 0:
                    out["prefix_cache_hit_rate_delta"] = round(min(1.0, dh / dq), 6)

            if "ttft_seconds_count" in before and "ttft_seconds_count" in after:
                dc = after["ttft_seconds_count"] - before["ttft_seconds_count"]
                ds = after.get("ttft_seconds_sum", 0.0) - before.get("ttft_seconds_sum", 0.0)
                if dc > 0 and ds >= 0:
                    out["server_mean_ttft_ms"] = round((ds / dc) * 1000.0, 3)

        return out


def summarize_episode_serving(
    turns: list[TurnLatencyRecord],
    prom_before: dict[str, float] | None = None,
    prom_after: dict[str, float] | None = None,
) -> EpisodeServingMetrics:
    delta = VLLMMetricsScraper.compute_episode_delta(prom_before, prom_after)
    if not turns:
        return EpisodeServingMetrics(
            prefix_cache_hit_rate_delta=delta["prefix_cache_hit_rate_delta"],
            prefix_cache_hit_rate_gauge=delta["prefix_cache_hit_rate_gauge"],
            delta_prefix_cache_queries=delta["delta_prefix_cache_queries"],
            delta_prefix_cache_hits=delta["delta_prefix_cache_hits"],
            server_mean_ttft_ms=delta["server_mean_ttft_ms"],
            gpu_cache_usage_perc=delta["gpu_cache_usage_perc"],
        )

    ttfts = [t.ttft_ms for t in turns]
    e2es = [t.e2e_latency_ms for t in turns]
    itls = [t.itl_ms for t in turns if t.itl_ms > 0]
    tps = [t.tokens_per_sec for t in turns if t.tokens_per_sec > 0]

    return EpisodeServingMetrics(
        num_agent_turns=len(turns),
        mean_ttft_ms=round(statistics.mean(ttfts), 3),
        p50_ttft_ms=round(_percentile(ttfts, 50.0), 3),
        p95_ttft_ms=round(_percentile(ttfts, 95.0), 3),
        max_ttft_ms=round(max(ttfts), 3),
        mean_e2e_latency_ms=round(statistics.mean(e2es), 3),
        total_agent_latency_ms=round(sum(e2es), 3),
        mean_itl_ms=round(statistics.mean(itls), 3) if itls else 0.0,
        mean_tokens_per_sec=round(statistics.mean(tps), 3) if tps else 0.0,
        total_prompt_tokens_server=sum(t.prompt_tokens for t in turns),
        total_completion_tokens_server=sum(t.completion_tokens for t in turns),
        prefix_cache_hit_rate_delta=delta["prefix_cache_hit_rate_delta"],
        prefix_cache_hit_rate_gauge=delta["prefix_cache_hit_rate_gauge"],
        delta_prefix_cache_queries=delta["delta_prefix_cache_queries"],
        delta_prefix_cache_hits=delta["delta_prefix_cache_hits"],
        server_mean_ttft_ms=delta["server_mean_ttft_ms"],
        gpu_cache_usage_perc=delta["gpu_cache_usage_perc"],
        turns=turns,
    )
