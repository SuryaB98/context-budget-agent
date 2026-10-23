# Benchmark Selection & Version Pinning Memo

**Project:** Token Exchange Rates: How Small Model Agents Should Spend a Fixed Context Budget  
**Degree / Course:** Amrita Online MCA (Major Project 21CSA391A)  
**Date:** 2026-10-23  
**Status:** Pre-registered & Locked  
**Ticket:** [CBA-2] (Week 1 Milestone)  

---

## 1. Executive Summary & Decision

We formally select **τ³-bench Retail** (the bug-fixed release of `sierra-research/tau2-bench`) under an **"Extend"** strategy. We lock the benchmark to release tag **`v1.0.1`** (commit checkpoint `pre-v1.0.1` / `v1.0.1`).

- **Strategy:** Extend (retaining tasks, mock database, and deterministic state checks; wrapping with custom token allocation, schema distractor injection, and policy compliance monitors).
- **Primary Domain:** `retail` (multi-turn tool interaction with explicit customer service policy).
- **Fallback Contingency:** **BFCL v3/v4 Multi-turn** (scripted user turns, zero LLM-user cost) if τ³ encounters critical local serving failure.
- **Rejected Alternatives:** SOP-Bench (Amazon), ToolSandbox (Apple), ToolBench, AppWorld (excessive setup overhead, unvetted small-model floors, or non-deterministic LLM-judge grading).

---

## 2. Technical Audit & Environment Specifications

### 2.1 Repository & Immutable Pinning
| Attribute | Pinned Value |
| :--- | :--- |
| **Repository URL** | `https://github.com/sierra-research/tau2-bench` |
| **Pinned Release Tag** | **`v1.0.1`** |
| **Branch / Target** | `v1.0.1` release tag (frozen 1.x benchmark) |
| **License** | MIT License (commercial and academic research permitted) |

### 2.2 Why `v1.0.1` over `2.0.0`?
In October 2026, Sierra released τ-bench 2.0.0, which added voice mode and changed text communication grading to use a non-deterministic semantic LLM judge (`gpt-5.4-mini` / `gpt-5.5`).  
For scientific validity in this thesis:
1. **Hard Gate G3 Compliance:** `v1.0.1` uses **deterministic database state checks** and substring/rule communication verification, satisfying Hard Gate G3 (identical scores on trajectory regrading).
2. **Retail Bug Fixes:** `v1.0.1` includes the 26 critical retail domain bug fixes introduced in v1.0.0 without the non-deterministic LLM judge overhead.
3. **Task Volume:** Retains all 114 canonical retail tasks (compared to 50 in 2.0 English frozen subsets).

### 2.3 Runtime & Package Requirements
- **Python Version:** Python 3.12 (`>=3.12,<3.14` per `pyproject.toml`)
- **Package Manager:** `uv` (`uv sync` for text benchmark)
- **Serving Engine Routing:** Built on `litellm>=1.80.15`, providing native compatibility with OpenAI-compatible endpoints.

### 2.4 vLLM & Local Agent Configuration
The local model (`Qwen/Qwen3-1.7B`) served via vLLM will be targeted using LiteLLM provider routing:
- **CLI Flag:** `--agent-llm openai/Qwen/Qwen3-1.7B`
- **Environment Variables:**
  ```env
  OPENAI_API_BASE="http://localhost:8000/v1"
  OPENAI_API_KEY="EMPTY"
  ```
- **Tool Calling:** vLLM launched with `--enable-auto-tool-choice --tool-call-parser hermes`.

---

## 3. Grader Audit & Quality Filters (Issue #384)

In τ² / τ³ retail, reward is calculated as:
$$\text{Reward} = \text{DB\_Reward} \times \text{Communicate\_Reward}$$

### Task Filtering Pre-registration:
- **Issue #384 Risk:** Tasks with empty `reward_basis` or missing assertions pass trivially regardless of whether the agent performed valid actions.
- **Filter Rule:** Before running the 50-task pilot or evaluation split, every retail task definition under `data/tau2/domains/retail/tasks/` must be audited. Any task with `len(reward_basis) == 0` or empty `evaluation_criteria` will be excluded from the evaluation set.

---

## 4. User Simulator Strategy & Cost Management

1. **Role of Simulator:** Roleplays the customer based on task persona and goals across multiple turns.
2. **Simulator Model:** `openai/gpt-4o-mini` (or `gpt-4.1` with `temperature: 0.0` for repeatability).
3. **Cost Control:** Hard spending cap set at $10.00 with billing alert on OpenAI API.
4. **Communication Scorer:** Explicitly configured as deterministic substring matching (`--communicate-scorer substring`) to guarantee zero variance on regrading.

---

## 5. Formal Supervisor Note (For Interim Report)

> **Supervisor & Evaluator Note on Benchmark Versioning:**  
> The approved Major Project proposal designated $\tau^2$-bench as the primary evaluation testbed. During infrastructure initialization (Week 1), we audited the official repository (`sierra-research/tau2-bench`) and verified that the maintainers resolved 26 retail task evaluation discrepancies in the `v1.0.0` / `v1.0.1` release series.  
> To guarantee experimental reproducibility and data integrity, our study pins the codebase to release **`v1.0.1`**. This constitutes a technical bug-fix and version-lock update of the benchmark named in the proposal, preserving the identical retail domain, task schema, and policy mechanics while eliminating known false-pass artifacts.
