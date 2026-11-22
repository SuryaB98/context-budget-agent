# Context Budget Agent — Project Guide & Cross-Machine Context

This file is automatically loaded by Antigravity in every session inside this repository to preserve architectural decisions, empirical audit findings, guardrails, and current execution state across machines (Windows, macOS, GitHub Codespaces, and Cloud Linux GPU VMs).

---

## 1. Project Identity & Deadlines

- **Title:** *Token Exchange Rates: How Small Model Agents Should Spend a Fixed Context Budget*
- **Program / Course:** Amrita Online MCA (`21CSA391A`) · **Student:** Surya B (`AA.SC.P2MCA25014003`)
- **Repository:** `https://github.com/SuryaB98/context-budget-agent`
- **Hard Experiment Freeze:** **Sun 3 Jan 2027** (no new GPU experiments after this date)
- **Final Dissertation Submission:** **Thu 15 Jan 2027**
- **Research Questions:**
  - **RQ1 (Schema vs. History Trade-off):** Evaluate `S:H` allocation ratios (`75:25`, `50:50`, `25:75`, and unconstrained baseline) across `2k` (`2048`), `4k` (`4096`), and `8k` (`8192`) BPE token budgets with Policy (`P`) pinned.
  - **RQ1b (Policy Retrieval):** Compare pinned `P` vs. top-$k$ retrieved policy chunks at `2k` and `4k` budgets for task accuracy and silent policy violations.
  - **RQ2 (LoRA SFT Budget Recovery):** Evaluate verifier-filtered LoRA SFT in a `2x2` ablation grid (`DataSize` in `{50%, 100%}` x `Rank r` in `{8, 32}`) on `Qwen3-1.7B`, plus the winning rank on `Qwen3-4B`, measuring recovery on held-out `2k/4k` budgets, unseen tool catalogue generalisation, and 100-item general language capability retention.
  - **RQ4 (Serving Cost Reality):** Measure vLLM PagedAttention prefix-cache hit rates (`vllm:gpu_prefix_cache_hit_rate`), Time-to-First-Token (TTFT), and end-to-end latency across prompt manipulation strategies (e.g., sliding-window trimming vs. history summarization).

---

## 2. Pinned Stack & Compute Architecture

- **Candidate Models:** `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`
- **Pinned Benchmark:** `sierra-research/tau2-bench` strictly pinned to tag **`v1.0.1`**
  - Commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
  - Retail `tasks.json` SHA-256: `f44563d8f36362e2dc58282bb02cfb85010d22f27d9439808758d3e9b2bc16d2`
  - Contingency Backup: BFCL multi-turn suite
- **Python Environment:** Python `3.12` managed via `uv` (`transformers`, `tokenizers`, `openai`, `pytest`, `scipy`, `numpy`, `pandas`, `matplotlib`)
- **Serving Engine:** vLLM (`vllm/vllm-openai:v0.8.5`, `--enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching --max-model-len 12288`)
- **Environment Split & Rules:**
  - **Local Machine (macOS / CPU / Codespaces):** Development, BPE token accounting, task auditing, budget controller, deterministic compliance checkers, `pytest` suites, statistical analysis (`10k` bootstrap CIs), Matplotlib figures, and dissertation writing.
  - **Cloud Linux GPU VM (24 GB NVIDIA VRAM):** vLLM serving (`setup/run_vllm.sh`) and LoRA SFT training.
  - **Rules:**
    1. Do **not** run `setup/run_vllm.sh` on macOS (it targets a Linux CUDA Docker container).
    2. Never mix local macOS `mlx-lm`/`ollama` sanity checks into experimental metrics or paper tables.
    3. Do **not** rent a Cloud GPU until Block 1 (`CBA-8`, `CBA-9`, `CBA-10`) passes `pytest` locally.

---

## 3. Locked Empirical Findings & Design Decisions (CBA-8 Pre-Flight Audit)

Empirical measurement against `tau2-bench@v1.0.1` retail and `Qwen3-1.7B` / `Qwen3-4B` tokenizers (identical token counts across both models):
- **Templated Policy (`P`):** **`1,492` tokens** (`AGENT_INSTRUCTION` + `<policy>` + `policy.md`; raw `policy.md` is `1,407` tokens).
- **Templated Tools (`S_all`):** **`3,382` tokens** across all **16** retail tools (`7 WRITE / 7 READ / 2 GENERIC`, including `get_item_details`; earlier docs understated 15).
- **Turn-1 Baseline (`P + S_all + framing`):** **`4,883` tokens** before any user goal or dialogue history.
- **Oracle History (`H_oracle`, 114 tasks):** `p50 = 1,777` tokens, `mean = 2,305` tokens (`min = 66`, `p90 = 5,036`, `max = 7,016`).

Locked Design Decisions:
- **Decision D1 (Allocatable `(S + H)` Budget):** Treat `2k / 4k / 8k` as the **allocatable `(S + H)` budget** `B_alloc` in `{2048, 4096, 8192}`, with `P = 1,492` pinned separately and `R_out = 512`, and set vLLM `--max-model-len 12288`. (A literal `2048` total window with `R_out = 512` and `P = 1,492` leaves only `44` tokens—smaller than the cheapest `161`-token tool. Under `B_alloc`, `S_min + H_oracle` fits `36.8%` at `2k`, `71.9%` at `4k`, and `99.1%` at `8k`.)
- **Decision D2 (Leak-Free Persona-Grouped Split):** Exclude the **6 vacuous-pass tasks** (`10, 12, 25, 50, 57, 65`, which have zero expected `WRITE` calls and zero graded `nl_assertions`; **retain Task `24`** which has graded `nl_assertions`), leaving **108** valid tasks. Construct a seeded (`seed=42`) **persona-grouped stratified split** (`~50` Eval / `~58` Train) with **`0` shared `known_info` customer personas** (Sierra's official `split_tasks.json` leaks `20/31` multi-task customer personas across train/test).
- **Decision D3 (Dual-Metric Grading — Hard Gate G3):** In `v1.0.1` retail, `0/114` tasks use `COMMUNICATE` in `reward_basis`, `40/114` invoke LLM-judged `nl_assertions` (`gpt-4.1-2025-04-14`, `temperature=0.0`), and `3` tasks (`30, 31, 32`) are `ci_only` (`communicate_info` set, `nl_assertions=None`). Report **both** strict deterministic `DB + COMMUNICATE` substring score and official cached `DB x NL_ASSERTION`.

---

## 4. Strict Engineering Guardrails & Known Pitfalls

- **`transformers 5.x` Tokenizer Bug:** `tokenizer.apply_chat_template(..., tools=tools, tokenize=True)` returns a `BatchEncoding` dict (`len(...) == 2`) in `transformers 5.24.0`. Always pass `tokenize=False` and count `len(tokenizer.encode(rendered, add_special_tokens=False))`.
- **`tau2` / `litellm` Offline Hang:** Importing `tau2` triggers `litellm` background network fetches. Always set `LITELLM_LOCAL_MODEL_COST_MAP=True` when running `tau2`-importing scripts, and extract static retail artifacts (`configs/retail_tools_openai.json`, `configs/retail_system_prompt.txt`, `configs/retail_policy.md`) once so runtime controllers, tokenizers, and `pytest` run offline in `<2s` with zero `tau2` imports.
- **Zero Train/Eval Data Leakage:** SFT trajectory generation (`CBA-15`) must strictly use `train_task_ids` only (`0` task ID overlap and `0` persona overlap with `eval_task_ids`).
- **Deterministic Compliance (Hard Gate G3):** Policy compliance evaluation (`src/compliance_checkers.py`) must be 100% deterministic over trajectory logs covering `R1`–`R6` plus concrete retail policy sub-rules in `notes/CBA_8_Execution_Plan.md` §6 (single-tool-per-turn, auth even with user-provided ID, cancel reason enum, refund destination constraint, transfer sequence constraint). Never use an LLM judge for compliance.
- **Repository Hygiene:** Never commit `.env`, `runs/` artifacts, `Planning/`, `timeline/`, or model weights (`*.pt`, `*.safetensors`, `*.bin`, etc.).

---

## 5. Current Status & Immediate Next Queue (as of 2026-11-22)

- **Completed / Drafted:**
  - `CBA-1` (`1ad3f2b`): Directory layout, `.gitignore`, baseline `README.md`.
  - `CBA-2` (`63bcae9`): Benchmark selection & pinning memo (`notes/benchmark_pinning.md`).
  - `CBA-4` scripts (`bae464e`): `setup/run_vllm.sh`, `scripts/test_tool_call.py`, `.devcontainer/devcontainer.json`, `PORTABLE_WORK_PACKET.md`.
  - `CBA-6` draft (`bae464e`): `scripts/count_tokens.py` (needs `transformers 5.x` fix in `CBA-8`).
  - `CBA-7` draft (`bae464e`): 6-paper literature matrix & gap (`notes/paper_notes.md`).
  - `CBA-8` audit (`6975ae4`): Empirical audit & step-by-step plan (`notes/CBA_8_Execution_Plan.md`).
- **Next Up — Block 1 (Local CPU, Zero GPU Cost):**
  1. **`CBA-8` (Start Here):** Execute `notes/CBA_8_Execution_Plan.md` — add `scripts/extract_retail_artifacts.py`, fix `scripts/count_tokens.py`, upgrade `scripts/audit_retail_tasks.py`, amend `notes/benchmark_pinning.md` & `setup/run_vllm.sh` (`--max-model-len 12288`), add `tests/test_cba8_artifacts.py`, and pass `pytest`.
  2. **`CBA-9`:** Implement `src/budget_controller.py`, `configs/distractor_tools.json`, and `tests/test_budget_controller.py`.
  3. **`CBA-10`:** Implement `src/compliance_checkers.py` (`R1`–`R6` + sub-rules) and `tests/test_compliance_checkers.py`.
  4. **`CBA-11`:** Compile & submit Interim Report using `PORTABLE_WORK_PACKET.md` §4, `notes/paper_notes.md`, `notes/benchmark_pinning.md`, and `CBA-8` outputs.
- **Subsequent Blocks:**
  - **GPU Window 1:** `CBA-3` (24 GB VM setup) -> `CBA-4` (vLLM verify) -> `CBA-5` (1-task smoke + 50-task floor pilot).
  - **Local Calibration:** `CBA-12` (hand-label 30–50 pilot transcripts, verify `>= 90%` checker agreement).
  - **GPU Window 2:** `CBA-13` (primary `2k/4k/8k` x `S:H` sweep + latency/cache) -> `CBA-14` (RQ1b policy retrieval + H4 prefix-cache ON/OFF probe).
  - **GPU Window 3:** `CBA-15` (verifier-filtered train SFT dataset) -> `CBA-16` (`2x2` LoRA ablations on `1.7B` + `4B`) -> `CBA-17` (budget recovery, unseen tools, language retention evals).
  - **Final Wrap-up (Local CPU):** `CBA-18` (10k paired bootstrap CIs, 5-class failure taxonomy, Matplotlib figures, `v1.0.0` release tag, final dissertation & viva deck).

---

## 6. Source-of-Truth File Map

| Path | Purpose |
| :--- | :--- |
| `notes/CBA_8_Execution_Plan.md` | Authoritative empirical audit findings (D1–D3), 16-tool token table, and exact CBA-8 runbook |
| `notes/JIRA_MASTER_BACKLOG.md` | End-to-end execution backlog (`CBA-1` to `CBA-18`) with scope and acceptance criteria |
| `notes/benchmark_pinning.md` | `tau2-bench@v1.0.1` pinning rationale & user-simulator spec (to be amended in CBA-8) |
| `notes/paper_notes.md` | 6-paper literature synthesis matrix & 3-part Research Gap statement (`CBA-7`) |
| `PORTABLE_WORK_PACKET.md` | Self-contained project brief, `R1`–`R6` rule specs, Interim Report blueprint, 8-week schedule |
| `.devcontainer/devcontainer.json` | Python `3.12` + `uv` + core pip dependencies reference |
| `setup/run_vllm.sh` | CUDA Docker launcher for vLLM OpenAI-compatible server (`Qwen3-1.7B` / `4B`) |
| `scripts/count_tokens.py` | Native Qwen3 BPE component token accountant (`P`, `S`, `H`) |
| `scripts/audit_retail_tasks.py` | Retail task Issue #384 / vacuous-pass auditor & Train/Eval split generator |
| `scripts/test_tool_call.py` | End-to-end verification script for vLLM `/health`, `/v1/models`, and Hermes `tool_calls` |

---

## 7. Keep Context Fresh

At the completion of each ticket (`CBA-8`, `CBA-9`, etc.), update **Section 5 (Current Status & Immediate Next Queue)** in this `GEMINI.md` inside the same commit so every machine cloning or pulling `main` starts with accurate state.
