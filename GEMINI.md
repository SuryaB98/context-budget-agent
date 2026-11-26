# Context Budget Agent — Project Guide & Cross-Machine Context

This file is automatically loaded by Antigravity in every session inside this repository to preserve architectural decisions, empirical audit findings, guardrails, and current execution state across machines (Windows, GitHub Codespaces, RunPod Linux GPU Pods, and macOS).

---

## 1. Project Identity & Deadlines

- **Title:** *Token Exchange Rates: How Small Model Agents Should Spend a Fixed Context Budget*
- **Program / Course:** Amrita Online MCA (`21CSA391A`) · **Student:** Surya B (`AA.SC.P2MCA25014003`)
- **Repository:** `https://github.com/SuryaB98/context-budget-agent`
- **Hard Experiment Freeze:** **Sun 3 Jan 2027** (no new GPU experiments after this date)
- **Final Dissertation Submission:** **Thu 15 Jan 2027**
- **Research Questions:**
  - **RQ1 (Schema vs. History Trade-off):** Evaluate `S:H` allocation ratios (`75:25`, `50:50`, `25:75`, and unconstrained baseline) across `2k` (`2048`), `4k` (`4096`), and `8k` (`8192`) BPE token budgets with Policy (`P`) pinned and distractor counts `N in {0, 10, 23}` (`configs/sweep_grid.json`).
  - **RQ1b (Policy Retrieval):** Compare pinned `P` vs. BM25 top-$k$ (`k=3`) retrieved policy chunks at `2k` and `4k` budgets for task accuracy and silent policy violations.
  - **RQ2 (LoRA SFT Budget Recovery):** Evaluate verifier-filtered LoRA SFT in a `2x2` ablation grid (`DataSize` in `{50%, 100%}` x `Rank r` in `{8, 32}`) on `Qwen3-1.7B`, plus the winning rank on `Qwen3-4B`, measuring recovery on held-out `2k/4k` budgets, unseen tool catalogue generalisation (`13` `banking_knowledge` tools), and 100-item general language capability retention.
  - **RQ4 (Serving Cost Reality):** Measure vLLM PagedAttention prefix-cache hit rates (`vllm:gpu_prefix_cache_hit_rate`), Time-to-First-Token (TTFT), and end-to-end latency across prompt manipulation strategies (`sliding_window` vs. `summarize_dropped` with prefix caching ON/OFF).

---

## 2. Pinned Stack & Locked Two-Tier Compute Architecture

- **Candidate Models:** `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` (plus 45-min floor screen on `Qwen3-8B` / `14B` in GPU Window 1)
- **Pinned Benchmark:** `sierra-research/tau2-bench` strictly pinned to tag **`v1.0.1`**
  - Commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
  - Retail `tasks.json` SHA-256: `f44563d8f36362e2dc58282bb02cfb85010d22f27d9439808758d3e9b2bc16d2`
  - Contingency Backup: BFCL multi-turn suite
- **Python Environment:** Python `3.12` managed via `uv` (`transformers`, `tokenizers`, `rank-bm25`, `jinja2`, `openai`, `pytest`, `scipy`, `numpy`, `pandas`, `matplotlib`)
- **Serving Engine:** vLLM (`vllm/vllm-openai:v0.8.5`, `--enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching --max-model-len 12288`)
- **Locked Two-Tier Compute Architecture (Decision Locked 2026-11-23):**
  - **Tier 1 — Everyday CPU Dev, Tests & Writing (`$0`):** **GitHub Codespaces (Free)** until the new macOS laptop arrives (`.devcontainer/devcontainer.json` installs all offline dependencies so `pytest tests/ -v` passes immediately; use `agy` CLI in Codespaces terminal, or Antigravity IDE locally once on macOS).
  - **Tier 2 — Batched GPU Compute:** **RunPod Community Cloud + 40 GB Persistent Network Volume (`/workspace`)** (`~$2.80/mo` storage, prepaid hard spend cap `~$25`). Attach the same `/workspace` volume to an **A100 80 GB** for the 45-min 4-model floor screen (`Qwen3-1.7B / 4B / 8B / 14B`) and **RTX 4090 24 GB** / **RTX 3090 24 GB** for sweeps & LoRA SFT.
- **Operational Rules:**
  1. RunPod pods already run inside Docker — launch `vllm serve ...` directly instead of nested `docker run` (`setup/run_vllm.sh` is for bare-metal/VM Docker hosts).
  2. On first RunPod boot, persist Antigravity memory and Hugging Face cache on the Network Volume: `mkdir -p /workspace/.gemini /workspace/hf_cache && ln -sfn /workspace/.gemini ~/.gemini` and `export HF_HOME=/workspace/hf_cache`.
  3. Never leave a RunPod GPU pod idle — terminate the pod immediately after each batch and push code + compact `notes/` summaries to `origin/main` (keep raw `runs/` on `/workspace/runs/` and back them up periodically).
  4. Never run `setup/run_vllm.sh` on macOS, and never mix local macOS `mlx-lm`/`ollama` sanity checks into thesis metrics.

---

## 3. Locked Empirical Findings & Design Decisions (CBA-8 Pre-Flight Audit)

Empirical measurement against `tau2-bench@v1.0.1` retail and `Qwen3-1.7B` / `Qwen3-4B` tokenizers (identical token counts across both models):
- **Templated Policy (`P`):** **`1,492` tokens** (`AGENT_INSTRUCTION` + `<policy>` + `policy.md`; raw `policy.md` is `1,407` tokens).
- **Templated Tools (`S_all`):** **`3,382` tokens** across all **16** retail tools (`7 WRITE / 7 READ / 2 GENERIC`, including `get_item_details`).
- **Turn-1 Baseline (`P + S_all + framing`):** **`4,883` tokens** before any user goal or dialogue history.
- **Oracle History (`H_oracle`, 114 tasks):** `p50 = 1,777` tokens, `mean = 2,305` tokens (`min = 66`, `p90 = 5,036`, `max = 7,016`).

Locked Design Decisions:
- **Decision D1 (Allocatable `(S + H)` Budget):** Treat `2k / 4k / 8k` as the **allocatable `(S + H)` budget** `B_alloc` in `{2048, 4096, 8192}`, with `P = 1,492` pinned separately and `R_out = 512`, and set vLLM `--max-model-len 12288`. (Under `B_alloc`, `S_min + H_oracle` fits `36.8%` at `2k`, `71.9%` at `4k`, and `99.1%` at `8k`.)
- **Decision D2 (Leak-Free Persona-Grouped Split):** Exclude the **6 vacuous-pass tasks** (`10, 12, 25, 50, 57, 65`; **retain Task `24`**), leaving **108** valid tasks split via seeded (`seed=42`) **persona-grouped stratified split** into **`50` Eval / `58` Train** tasks (`29` Eval / `32` Train persona groups, **`0` shared `known_info` customer personas**).
- **Decision D3 (Dual-Metric Grading — Hard Gate G3):** In `v1.0.1` retail, `0/114` tasks use `COMMUNICATE` in `reward_basis`, `40/114` invoke LLM-judged `nl_assertions` (`gpt-4.1-2025-04-14`, `temperature=0.0`), and `3` tasks (`30, 31, 32`) are `ci_only`. Report **both** strict deterministic `DB + COMMUNICATE` substring score and official cached `DB x NL_ASSERTION`.

---

## 4. Strict Engineering Guardrails & Known Pitfalls

- **`transformers 5.x` Tokenizer Bug:** `tokenizer.apply_chat_template(..., tools=tools, tokenize=True)` returns a `BatchEncoding` dict (`len(...) == 2`). Always pass `tokenize=False` and count `len(tokenizer.encode(rendered, add_special_tokens=False))`.
- **`tau2` / `litellm` Offline Hang:** Always set `LITELLM_LOCAL_MODEL_COST_MAP=True` when importing `tau2`, and use static frozen configs (`configs/retail_tools_openai.json`, `configs/retail_system_prompt.txt`, `configs/retail_policy.md`) so controllers, tokenizers, and `pytest` run offline in `<2s` with zero `tau2` imports.
- **Zero Train/Eval Data Leakage:** SFT trajectory generation (`CBA-15`) must strictly use `train_task_ids` (`58` tasks) only (`0` task ID overlap and `0` persona overlap with `eval_task_ids`).
- **Distractor Hygiene:** `configs/distractor_tools_in_dist.json` (`23` tools: `11` airline + `12` telecom) is for RQ1 sweeps and SFT training; `configs/distractor_tools_unseen.json` (`13` `banking_knowledge` tools) is strictly held out for Milestone 7 unseen-tool generalisation (never use unseen distractors during RQ1 sweeps or SFT training).
- **8-Rule Deterministic Compliance Engine (Hard Gate G3):** `src/compliance_checkers.py` must remain 100% deterministic over trajectory logs (no LLM-as-a-judge):
  - **Tier A (Unenforced — drive Silent Policy Violations):** `R1_AUTH_BEFORE_ACCESS`, `R2_EXPLICIT_CONFIRMATION`, `R3_CHECK_ORDER_BEFORE_MUTATE`, `R4_SINGLE_USER_ISOLATION`, `R5_SINGLE_TOOL_PROTOCOL`, `R6_TRANSFER_PROTOCOL`.
  - **Tier B (Tool-Enforced / Hallucination Telemetry):** `R7_ATTEMPTED_INVALID_MUTATION`, `R8_UNGROUNDED_ARGUMENT`.
  - Use `compute_checker_agreement` and `load_hand_labels_csv` for `CBA-12` calibration (`>= 90%` gate).
- **Repository Hygiene:** Never commit `.env`, `runs/` artifacts, `Planning/`, `timeline/`, or model weights (`*.pt`, `*.safetensors`, `*.bin`, etc.).

---

## 5. Current Status & Immediate Next Queue (as of 2026-11-26)

- **Completed (Block 1 Engineering + Pre-GPU Harness + Interim Report Draft Ready — `40` offline `pytest` unit tests passing):**
  - `CBA-1` (`1ad3f2b`): Directory layout, `.gitignore`, baseline `README.md`.
  - `CBA-2` (`63bcae9`, amended `49928b1`): Benchmark pinning memo (`notes/benchmark_pinning.md`).
  - `CBA-4` scripts (`bae464e`, updated `49928b1`): `setup/run_vllm.sh` (`--max-model-len 12288`), `scripts/test_tool_call.py`.
  - `CBA-6` & `CBA-8` (`49928b1`): `scripts/extract_retail_artifacts.py`, fixed `scripts/count_tokens.py`, upgraded `scripts/audit_retail_tasks.py`, `configs/retail_task_split.json` (`50` Eval / `58` Train, `0` persona overlap, `6` vacuous excluded), `configs/retail_tools_openai.json` (`16` tools), `configs/distractor_tools_in_dist.json` (`23` tools), `configs/distractor_tools_unseen.json` (`13` tools), `notes/retail_token_profile.json`, `tests/test_cba8_artifacts.py`.
  - `CBA-7` (verified `2026-11-26`): 6-paper literature matrix & gap (`notes/paper_notes.md`, all 6 arXiv IDs, titles, authors, benchmarks & empirical numbers verified).
  - `CBA-9` (`0200033`): `src/budget_controller.py` (`ContextBudgetController` with exact Qwen3 BPE budgets, BM25 schema/policy selection, atomic history trimming, H4 summarization), `configs/sweep_grid.json`, `tests/test_budget_controller.py` (`17` unit tests).
  - `CBA-10` (`42d4141`): `src/compliance_checkers.py` (`8` deterministic rules `R1`–`R8` + calibration metrics), `tests/test_compliance_checkers.py` (`12` unit tests).
  - **`eval_runner` + `serving_profiler` + RunPod Launcher (`9b33279`):** `src/eval_runner.py`, `src/serving_profiler.py`, `setup/run_vllm_runpod.sh`, `tests/test_eval_runner.py` (`6` unit tests).
  - **`CBA-11` Interim Report Package + GPU Window 1 Runbook:** `notes/Interim_Report_21CSA699A_Surya_B.md`, `notes/Interim_Report_21CSA699A_Surya_B.docx`, `notes/figures/system_architecture.png`, `scripts/generate_interim_docx.py`, and `notes/GPU_WINDOW_1_RUNBOOK.md`.
- **Immediate Next Queue:**
  1. **Portal Submission (`CBA-11`):** Review/sign `notes/Interim_Report_21CSA699A_Surya_B.docx`, export to PDF, and submit on Amrita Online portal.
  2. **GPU Window 1 (`CBA-3` -> `CBA-4` -> `CBA-5` — RunPod + 40 GB Network Volume):** Follow `notes/GPU_WINDOW_1_RUNBOOK.md` to provision `/workspace` volume, launch `setup/run_vllm_runpod.sh`, verify Hermes tool calling (`scripts/test_tool_call.py`), run 45-min 4-model floor screen (`Qwen3-1.7B / 4B / 8B / 14B` on A100 80 GB) + 50-task baseline pilot (`python -m src.eval_runner --grid baseline_pilot ...`).
  3. **Local Calibration (`CBA-12` — Codespaces / Mac):** Hand-label 30–50 pilot transcripts and verify `>= 90%` checker agreement.
  4. **GPU Window 2 (`CBA-13`, `CBA-14` — RunPod):** Core `2k/4k/8k` sweep (`--grid rq1_primary`), RQ1b policy retrieval (`--grid rq1b_policy_retrieval`), H4 prefix-cache ON/OFF probe (`--grid rq4_prefix_cache_h4`).
  5. **GPU Window 3 (`CBA-15`, `CBA-16`, `CBA-17` — RunPod):** Verifier-filtered train-only SFT dataset, `2x2` LoRA ablations, Milestone 7 budget recovery / unseen tools (`--grid milestone7_unseen_tools`) / language retention evals.
  6. **Final Wrap-up (`CBA-18` — Mac / CPU):** 10k paired bootstrap CIs, 5-class failure taxonomy, Matplotlib figures, `v1.0.0` release tag, final dissertation & viva deck.

---

## 6. Source-of-Truth File Map

| Path | Purpose |
| :--- | :--- |
| `src/budget_controller.py` | Per-component BPE budget controller (`P`, `S <= B_S`, `H <= B_H`, BM25 selection, atomic trimming) |
| `src/compliance_checkers.py` | 8-rule deterministic retail policy compliance engine (`R1`–`R8`) & `CBA-12` calibration helpers |
| `src/eval_runner.py` | End-to-end evaluation harness (`BudgetedTurnExecutor`, streaming vLLM client, lazy `tau2` adapter, Decision D3 dual grader + `NLAssertionJudgeCache`, `--resume`, sweep presets) |
| `src/serving_profiler.py` | vLLM Prometheus `/metrics` scraper (`gpu_prefix_cache_hit_rate` delta/gauge) & streaming TTFT/ITL profiler |
| `configs/retail_task_split.json` | Frozen leak-free `50` Eval / `58` Train split (`6` vacuous excluded, `0` persona overlap) |
| `configs/sweep_grid.json` | Experiment matrix for `RQ1`, schema compression, `RQ1b`, `RQ4` cache probe, and unseen tools |
| `configs/retail_tools_openai.json` | Frozen OpenAI schemas for all `16` retail tools (`3,382` templated Qwen3 tokens) |
| `configs/distractor_tools_in_dist.json` | `23` in-distribution distractor schemas (`11` airline + `12` telecom) for RQ1 & SFT |
| `configs/distractor_tools_unseen.json` | `13` held-out `banking_knowledge` schemas reserved strictly for Milestone 7 |
| `notes/retail_token_profile.json` | Verified Qwen3 BPE token footprint (`P=1,492`, `S_all=3,382`, `H_oracle`, fit rates) |
| `notes/CBA_8_Execution_Plan.md` | Authoritative empirical audit findings (D1–D3), 16-tool token table, and CBA-8 runbook |
| `notes/JIRA_MASTER_BACKLOG.md` | End-to-end execution backlog (`CBA-1` to `CBA-18`) with scope and acceptance criteria |
| `notes/benchmark_pinning.md` | `tau2-bench@v1.0.1` pinning rationale, SHA-256 locks, and dual-metric grading spec |
| `notes/paper_notes.md` | 6-paper literature synthesis matrix & 3-part Research Gap statement (`CBA-7`) |
| `PORTABLE_WORK_PACKET.md` | Self-contained project brief, Interim Report blueprint, and 8-week schedule |
| `.devcontainer/devcontainer.json` | Codespaces Python `3.12` + `uv` + all offline dependencies (`rank-bm25`, `jinja2`, etc.) |
| `setup/run_vllm.sh` | CUDA Docker launcher for vLLM OpenAI-compatible server (bare-metal/VM hosts) |
| `setup/run_vllm_runpod.sh` | RunPod-native direct `vllm serve` launcher (prefix-cache ON/OFF, LoRA modules, `/workspace/hf_cache`) |
| `notes/GPU_WINDOW_1_RUNBOOK.md` | Copy-pasteable RunPod GPU Window 1 runbook (`CBA-3` → `CBA-4` → `CBA-5` + pod teardown) |
| `scripts/` & `tests/` | `extract_retail_artifacts.py`, `audit_retail_tasks.py`, `count_tokens.py`, `test_tool_call.py`, `40` unit tests |

---

## 7. Keep Context Fresh

At the completion of each ticket (`CBA-11`, `eval_runner`, `CBA-3/4/5`, etc.), update **Section 5 (Current Status & Immediate Next Queue)** in this `GEMINI.md` inside the same commit so every machine cloning or pulling `main` starts with accurate state.
