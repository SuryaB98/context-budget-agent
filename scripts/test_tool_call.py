#!/usr/bin/env python3
"""
[CBA-4] Verify vLLM OpenAI Endpoint & Structured Tool-Calling Protocol.

Checks:
1. GET /health returns 200 OK
2. GET /v1/models lists the served Qwen3 model
3. POST /v1/chat/completions with a mock retail tool schema returns valid
   structured `tool_calls` JSON via the Hermes tool parser.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


MOCK_RETAIL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Get the status and items of a retail customer order by order_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order identifier starting with '#', e.g. '#W0000001'.",
                    }
                },
                "required": ["order_id"],
            },
        },
    }
]


def http_json(url: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    req = urllib.request.Request(url)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer EMPTY")
        resp = urllib.request.urlopen(req, data=data, timeout=60)
    else:
        resp = urllib.request.urlopen(req, timeout=15)
    status = resp.getcode()
    body = resp.read().decode("utf-8")
    return status, json.loads(body) if body.strip().startswith(("{", "[")) else body


def main() -> int:
    parser = argparse.ArgumentParser(description="Test vLLM Qwen3 tool calling endpoint.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM server root URL")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Served model identifier")
    parser.add_argument(
        "--out",
        default="notes/vllm_test_response.json",
        help="Path to save verification output JSON",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    print(f"[1/3] Checking health endpoint: {base}/health ...")
    status, _ = http_json(f"{base}/health")
    assert status == 200, f"Health check failed with status {status}"
    print("      -> 200 OK")

    print(f"[2/3] Checking models endpoint: {base}/v1/models ...")
    _, models_resp = http_json(f"{base}/v1/models")
    model_ids = [m["id"] for m in models_resp.get("data", [])]
    print(f"      -> Available models: {model_ids}")
    if args.model not in model_ids and model_ids:
        print(f"      [WARN] Requested {args.model} not in list; using {model_ids[0]}")
        args.model = model_ids[0]

    print(f"[3/3] Testing structured tool call on {args.model} ...")
    chat_payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a retail customer service agent. Always use the provided tools to look up order data.",
            },
            {
                "role": "user",
                "content": "Can you check the status of my order #W1234567?",
            },
        ],
        "tools": MOCK_RETAIL_TOOLS,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    _, chat_resp = http_json(f"{base}/v1/chat/completions", chat_payload)

    choice = chat_resp["choices"][0]["message"]
    tool_calls = choice.get("tool_calls") or []
    if not tool_calls:
        print("[ERROR] No structured tool_calls returned in response:", json.dumps(chat_resp, indent=2))
        return 1

    tc = tool_calls[0]["function"]
    parsed_args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
    print(f"      -> Success! Parsed tool call: {tc['name']}({json.dumps(parsed_args)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chat_resp, indent=2), encoding="utf-8")
    print(f"[DONE] Saved raw verification response to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
