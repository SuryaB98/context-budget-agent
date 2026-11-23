#!/usr/bin/env python3
"""
[CBA-8] Extract frozen static retail & distractor artifacts from tau2-bench@v1.0.1.

Exports pure JSON/text files under `configs/` so downstream token accounting,
budget control, task auditing, and pytest suites run offline in <2s without
importing `tau2` or triggering `litellm` background threads.
"""

from __future__ import annotations

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.environment.toolkit import ToolType, get_tool_types
from tau2.registry import registry


def _tool_type_str(tt: Any) -> str:
    if isinstance(tt, ToolType):
        return tt.value
    return str(tt)


def _extract_domain_schemas(domain: str) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    if domain == "banking_knowledge":
        try:
            env = registry.get_env_constructor(domain)(retrieval_variant="no_knowledge")
        except Exception:
            from tau2.domains.banking_knowledge.environment import get_db
            from tau2.domains.banking_knowledge.tools import KnowledgeTools

            toolkit = KnowledgeTools(get_db())
            schemas = [t.openai_schema for t in toolkit.get_tools().values()]
            raw_types = get_tool_types(toolkit)
            tool_types = {name: _tool_type_str(tt) for name, tt in raw_types.items()}
            return schemas, tool_types, ""
    else:
        env = registry.get_env_constructor(domain)()
    tools = env.get_tools()
    schemas = [t.openai_schema for t in tools]
    raw_types = get_tool_types(env.tools) if env.tools is not None else {}
    tool_types = {name: _tool_type_str(tt) for name, tt in raw_types.items()}
    policy = env.get_policy() or ""
    return schemas, tool_types, policy


def extract_all(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary retail artifacts
    retail_schemas, retail_types, retail_policy = _extract_domain_schemas("retail")
    retail_names = {s["function"]["name"] for s in retail_schemas}

    write_tools = sorted(name for name, t in retail_types.items() if t == "write")
    read_tools = sorted(name for name, t in retail_types.items() if t == "read")
    generic_tools = sorted(name for name, t in retail_types.items() if t not in {"write", "read"})

    system_prompt = SYSTEM_PROMPT.format(
        domain_policy=retail_policy,
        agent_instruction=AGENT_INSTRUCTION,
    )

    (out_dir / "retail_tools_openai.json").write_text(
        json.dumps(retail_schemas, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "retail_tool_metadata.json").write_text(
        json.dumps(
            {
                "domain": "retail",
                "total_tools": len(retail_schemas),
                "tool_types": retail_types,
                "write_tools": write_tools,
                "read_tools": read_tools,
                "generic_tools": generic_tools,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "retail_system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (out_dir / "retail_policy.md").write_text(retail_policy, encoding="utf-8")

    # 2. In-distribution distractor tools (airline + telecom, deduplicated against retail)
    seen_names = set(retail_names)
    in_dist_schemas: list[dict[str, Any]] = []
    in_dist_sources: dict[str, list[str]] = {}

    for dom in ("airline", "telecom"):
        dom_schemas, _, _ = _extract_domain_schemas(dom)
        kept: list[str] = []
        for schema in dom_schemas:
            name = schema["function"]["name"]
            if name not in seen_names:
                seen_names.add(name)
                in_dist_schemas.append(schema)
                kept.append(name)
        in_dist_sources[dom] = kept

    (out_dir / "distractor_tools_in_dist.json").write_text(
        json.dumps(in_dist_schemas, indent=2) + "\n", encoding="utf-8"
    )

    # 3. Unseen tool catalogue (banking_knowledge, deduplicated against retail + in_dist)
    bk_schemas, _, _ = _extract_domain_schemas("banking_knowledge")
    unseen_schemas: list[dict[str, Any]] = []
    unseen_names: list[str] = []
    for schema in bk_schemas:
        name = schema["function"]["name"]
        if name not in seen_names:
            seen_names.add(name)
            unseen_schemas.append(schema)
            unseen_names.append(name)

    (out_dir / "distractor_tools_unseen.json").write_text(
        json.dumps(unseen_schemas, indent=2) + "\n", encoding="utf-8"
    )

    distractor_meta = {
        "in_dist_total": len(in_dist_schemas),
        "in_dist_by_domain": {k: len(v) for k, v in in_dist_sources.items()},
        "in_dist_tool_names": in_dist_sources,
        "unseen_total": len(unseen_schemas),
        "unseen_by_domain": {"banking_knowledge": len(unseen_names)},
        "unseen_tool_names": {"banking_knowledge": unseen_names},
    }
    (out_dir / "distractor_metadata.json").write_text(
        json.dumps(distractor_meta, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "retail_tools": len(retail_schemas),
        "write_tools": len(write_tools),
        "read_tools": len(read_tools),
        "generic_tools": len(generic_tools),
        "in_dist_distractors": len(in_dist_schemas),
        "unseen_distractors": len(unseen_schemas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract static retail and distractor artifacts from tau2.")
    parser.add_argument("--out-dir", default="configs", help="Output configs directory")
    args = parser.parse_args()

    summary = extract_all(Path(args.out_dir))
    print(json.dumps(summary, indent=2))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    os._exit(0)
