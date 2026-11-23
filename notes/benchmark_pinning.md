# Benchmark Selection & Version Pinning Memo

**Project:** Token Exchange Rates: How Small Model Agents Should Spend a Fixed Context Budget  
**Degree / Course:** Amrita Online MCA (Major Project 21CSA391A)  
**Date:** 2026-10-23 (Amended 2026-11-23 during `[CBA-8]` empirical audit)  
**Status:** Pre-registered, Empirically Audited & Locked  
**Tickets:** `[CBA-2]` (Week 1) · `[CBA-8]` (Retail Audit & Leak-Free Split)

---

## 1. Executive Summary & Decision

We formally select **τ³-bench Retail** (`sierra-research/tau2-bench`, pinned to release tag **`v1.0.1`**, commit **`fc0055dc4e0a316c3f83133267fbd6faaa770992`**) under an **"Extend"** strategy.

- **Strategy:** Extend (retaining the 16 retail tools, mock database, and deterministic DB state checks; wrapping with exact Qwen3 BPE token budgeting, distractor-tool injection, history trimming, and offline deterministic policy compliance checkers).
- **Primary Domain:** `retail` (`114` raw tasks, `108` strictly non-vacuous tasks after audit).
- **Fallback Contingency:** **BFCL v3/v4 Multi-turn** (scripted user turns, zero LLM-user cost) if any primary model falls below the `15%` floor gate.
- **Rejected Alternatives:** SOP-Bench (Amazon), ToolSandbox (Apple), ToolBench, AppWorld (excessive setup overhead or non-deterministic grading).

---

## 2. Technical Audit & Environment Specifications

### 2.1 Repository & Immutable Pinning
| Attribute | Pinned Value |
| :--- | :--- |
| **Repository URL** | `https://github.com/sierra-research/tau2-bench` |
| **Pinned Release Tag** | **`v1.0.1`** (annotated tag `b711c1ead46f55111bf765cf44d5da8bacc2d28c`) |
| **Pinned Commit SHA** | **`fc0055dc4e0a316c3f83133267fbd6faaa770992`** |
| **Retail `tasks.json` SHA-256** | `f44563d8f36362e2dc58282bb02cfb85010d22f27d9439808758d3e9b2bc16d2` |
| **Retail Tools Count** | **`16` tools** (`7` WRITE, `7` READ, `2` GENERIC; `3,382` templated Qwen3 tokens) |
| **License** | MIT License |

### 2.2 Why `v1.0.1` over `2.0.0`?
In October 2026, Sierra released τ-bench 2.0.0, adding voice mode and replacing default text communication grading with semantic LLM judging (`gpt-5.4-mini` / `gpt-5.5`). Pinning **`v1.0.1`**:
1. Preserves all `114` canonical retail tasks (vs. `50` English frozen tasks in 2.0.0) and bundles the `v1.0.0` retail task fixes plus `banking_knowledge` grading fixes.
2. Retains deterministic database state verification (`EnvironmentEvaluator`), while requiring explicit handling of `40/114` retail tasks that define `nl_assertions` (Decision D3 below).

### 2.3 Runtime, vLLM & Decision D1 (`--max-model-len 12288`)
- **Python Version:** Python 3.12 (`>=3.12,<3.14` per `pyproject.toml`), managed with `uv`.
- **Empirical Token Audit (`Qwen/Qwen3-1.7B` & `Qwen/Qwen3-4B` — identical tokenizers):**
  - Templated Policy `P` (`AGENT_INSTRUCTION` + `<policy>` + `policy.md`): **`1,492` tokens** (`1,407` raw BPE).
  - All `16` retail tools (`S_all`): **`3,382` tokens**; Turn 1 baseline (`P + S_all + framing`): **`4,883` tokens**.
  - **Decision D1 (Allocatable `S + H` Budget):** Under a literal `2,048` total window with `512` output reserve, pinning `P = 1,492` leaves only `44` tokens for `S + H` (less than the smallest tool `list_all_product_types` at `161` tokens). Therefore, primary allocation budgets **`2k (2048)`, `4k (4096)`, and `8k (8192)`** control the **allocatable `(S + H)` budget** with `P` pinned separately (`setup/run_vllm.sh` defaults to `--max-model-len 12288` to accommodate `1492 + 8192 + 512 = 10,196` tokens), while `RQ1b` tests compressing `P` itself via retrieval.

---

## 3. Grader Audit, Vacuous-Pass Filtering & Leak-Free Split (`[CBA-8]`)

### 3.1 Actual `v1.0.1` Grader Behavior & Decision D3 (Dual-Metric Reporting)
Empirical inspection of `src/tau2/evaluator/evaluator.py` and `tasks.json` in `v1.0.1` established:
- **`COMMUNICATE` in `reward_basis`:** **`0 / 114`** retail tasks include `COMMUNICATE` in `reward_basis` (there is **no** `--communicate-scorer` CLI flag in `tau2`). `112 / 114` tasks list `['DB', 'NL_ASSERTION']` and `2 / 114` (`33`, `34`) list `['DB']`.
- **`nl_assertions` LLM Judge:** **`40 / 114`** tasks (`29` official train, `11` official test) define non-empty `nl_assertions`, which invokes `NLAssertionsEvaluator` (`gpt-4.1-2025-04-14`, `temperature = 0.0`). The remaining `74` tasks short-circuit `NL_ASSERTION` to `1.0` without an LLM call.
- **`ci_only` Tasks:** `36` tasks specify `communicate_info`, of which `3` tasks (`30`, `31`, `32`) have `communicate_info = ['746342064230']` and `nl_assertions = None` (silently skipped by default `EvaluationType.ALL`).
- **Decision D3 (Dual-Metric Reporting for Hard Gate G3):** Every evaluation run will report **both**:
  1. **Strict Deterministic `DB + COMMUNICATE` Substring Score** across all `108` non-vacuous tasks (including `30, 31, 32`, zero LLM-judge variance, 100% Gate G3 compliant).
  2. **Official `tau2` `DB × NL_ASSERTION` Score** using pinned `gpt-4.1-2025-04-14` (`temperature = 0.0`, with judge verdicts cached per `(task_id, response)` for deterministic regrading).

### 3.2 True Vacuous-Pass Filter (`6` Tasks Excluded $\to$ `108` Valid)
- `0 / 114` tasks have missing `evaluation_criteria` or empty `reward_basis`.
- Instead, **`6` tasks (`10, 12, 25, 50, 57, 65`)** have **zero expected `WRITE` tool calls AND `nl_assertions = None`** (including transfer tasks `10, 12, 50` because `transfer_to_human_agents` is `ToolType.GENERIC` and leaves the DB unchanged). Under `EvaluationType.ALL`, an empty do-nothing trajectory scores `1.0` on these `6` tasks. All `6` are excluded, leaving **`108` strictly non-vacuous tasks** (retaining Task `24`, which has `actions = []` but has a graded `nl_assertion` and `communicate_info`).

### 3.3 Decision D2: Leak-Free Persona-Grouped Stratified Split (`50` Eval / `58` Train)
- Sierra's official `split_tasks.json` (`74` train / `40` test) leaks customer identities: all `114` tasks come from only `63` `known_info` personas, and **`20 / 31` multi-task persona groups cross the official train/test boundary**.
- `configs/retail_task_split.json` freezes a **persona-grouped, stratified split** (`seed = 42`, `50` Eval / `58` Train across the `108` valid tasks, **`0` persona overlap**), balanced across `has_nl_assertions` and `H_oracle` quartiles.

---

## 4. User Simulator Strategy & Cost Management

1. **Simulator Model:** `openai/gpt-4o-mini` (or pinned `gpt-4.1-2025-04-14` with `temperature = 0.0`), fixed across all experiments.
2. **Cost Control:** Hard spending cap set at \$10.00 for pilot runs with billing alert on OpenAI API.

---

## 5. Formal Supervisor Note (For Interim Report)

> **Supervisor & Evaluator Note on Benchmark Versioning & Audit:**  
> The approved Major Project proposal designated $\tau^2$-bench as the primary evaluation testbed. During infrastructure initialization and pre-flight auditing (`[CBA-2]`, `[CBA-8]`), we pinned the official repository (`sierra-research/tau2-bench`) to release **`v1.0.1`** (`fc0055dc4e0a316c3f83133267fbd6faaa770992`). Our automated audit uncovered `6` vacuous-pass tasks (excluded, leaving `108` valid tasks) and `20` customer-persona groups crossing the upstream `train`/`test` split. To ensure zero SFT train-to-eval leakage and deterministic grading parity, we froze a persona-grouped `50`-eval / `58`-train split (`configs/retail_task_split.json`) and adopted dual deterministic/official scoring.
