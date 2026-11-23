# 📋 Master JIRA Task Backlog: End-to-End Execution (`CBA-1` to `CBA-18`)

**Project:** Token Exchange Rates: How Small Model Agents Should Spend a Fixed Context Budget (`21CSA391A`)  
**Baseline:** `MCA_Project_Proposal_Amrita_Revised_Written_Final.pdf`  
**How to Use This File:** Execute tickets strictly in order from top to bottom. All **🟢 LOCAL CPU / BROWSER** tickets can be completed on any machine (or GitHub Codespaces / Colab) with zero GPU rental cost. Rent a 24 GB Cloud GPU only when entering a **🔵 GPU WINDOW** block.

---

## 📊 Master Kanban & Progress Tracker

| Ticket ID | Title | Proposal Milestone | Environment | Est. Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CBA-1** | Workspace Directory Structure & Git Ignore Verification | M1 | 🟢 Local / Browser | 0.5h | ✅ **DONE** (`1ad3f2b`) |
| **CBA-2** | Pin $\tau^3$-bench Retail `v1.0.1` & Benchmark Selection Memo | M1 | 🟢 Local / Browser | 1.5h | ✅ **DONE** (`63bcae9`) |
| **CBA-6** | Authoritative Token Accounting Script (`count_tokens.py`) | M2 | 🟢 Local CPU | 1.5h | ✅ **DONE** (`notes/retail_token_profile.json`) |
| **CBA-7** | Literature Review & Research Gap Qualification (`paper_notes.md`) | Interim | 🟢 Local / Browser | 2.0h | 🟡 **DRAFT DONE — REVIEW & VERIFY** (`bae464e`) |
| **CBA-8** | Audit $\tau^3$-bench `v1.0.1` Retail Tasks (Issue #384) & Freeze Train/Eval Split | M1 / M2 | 🟢 Local CPU | 1.0h | ✅ **DONE** (`configs/retail_task_split.json`) |
| **CBA-9** | Implement Per-Component Context Budget Controller (`S`, `P`, `H` + Distractors) | M2 | 🟢 Local CPU | 3.0h | ✅ **DONE** (`src/budget_controller.py`, `17` unit tests) |
| **CBA-10** | Implement 6 Deterministic Retail Policy Compliance Checkers (`R1`–`R6`) | M4 | 🟢 Local CPU | 3.0h | ✅ **DONE** (`src/compliance_checkers.py`, `8` rules, `12` tests) |
| **CBA-11** | Compile & Submit Interim Report Document | Interim | 🟢 Local / Docs | 2.5h | ⏭️ **NEXT UP (Do #1)** |
| **CBA-3** | Cloud GPU VM Provisioning (24 GB VRAM) & Driver Validation | M1 | 🔵 **GPU Window 1** | 1.0h | ⏳ Waiting for GPU Window 1 |
| **CBA-4** | Deploy `Qwen3-1.7B` & `4B` with vLLM & Verify Hermes Tool Calling | M1 | 🔵 **GPU Window 1** | 1.0h | ⏳ Waiting for GPU Window 1 (Scripts Ready) |
| **CBA-5** | $\tau^3$-bench Retail 1-Task Smoke Run & 50-Task Floor Pilot | M1 / M2 | 🔵 **GPU Window 1** | 3.0h | ⏳ Waiting for GPU Window 1 |
| **CBA-12** | Hand-Label 30–50 Pilot Transcripts & Calibrate Compliance Checkers ($\ge 90\%$) | M4 | 🟢 Local / Browser | 2.5h | ⏳ After CBA-5 |
| **CBA-13** | Primary Allocation Sweep (`2k/4k/8k` $\times$ $S:H$ Ratios $\times$ `1.7B/4B`) + Latency/Cache Logs | M3 / M5 | 🔵 **GPU Window 2** | 8.0h | ⏳ Week 7 |
| **CBA-14** | Secondary Policy-Retrieval Experiment (RQ1b) & Prefix-Cache ON vs OFF Probe | M4 / M5 | 🔵 **GPU Window 2** | 3.0h | ⏳ Week 7 |
| **CBA-15** | Construct Verifier-Filtered SFT Trajectory Dataset (From Train Split Only) | M6 | 🔵 **GPU Window 3** | 3.0h | ⏳ Week 8 |
| **CBA-16** | LoRA SFT Training & $2 \times 2$ Ablation Grid (Data Size $\times$ Rank on `1.7B` + `4B`) | M6 | 🔵 **GPU Window 3** | 5.0h | ⏳ Week 8–9 |
| **CBA-17** | Evaluate SFT Budget Recovery, Unseen Tool Catalogues & Language Retention | M7 | 🔵 **GPU Window 3** | 4.0h | ⏳ Week 10 |
| **CBA-18** | Paired Statistical Tests, 10k Bootstrap CIs, Failure Taxonomy & Final Report | M8 / M9 | 🟢 Local CPU | 12.0h | ⏳ Week 11–13 |

---

## 🟢 BLOCK 1: Immediate Local CPU / Browser Sprint (Zero GPU Required)

Finish `CBA-8` $\to$ `CBA-9` $\to$ `CBA-10` $\to$ `CBA-11` one by one right now (on any browser via GitHub Codespaces, Google Colab CPU, or your laptop).

---

### [CBA-8] Audit $\tau^3$-bench `v1.0.1` Retail Tasks (Issue #384) & Freeze Train/Eval Split
- **Status:** ✅ **DONE** (`50` Eval / `58` Train leak-free split frozen; `6` vacuous tasks excluded; `16` retail + `23` in-dist + `13` unseen schemas extracted)
- **Environment:** 🟢 Local CPU / GitHub Codespaces / Google Colab
- **Proposal Milestone:** Milestone 1 & Milestone 2
- **Description:**  
  Clone `sierra-research/tau2-bench` at tag `v1.0.1`, run `scripts/audit_retail_tasks.py` and `scripts/count_tokens.py` on the real retail `tasks.json` and `policy.md`, exclude any broken tasks with empty `reward_basis` (Issue #384), freeze the deterministic 50-Eval / ~64-Train task split, and record the exact BPE token profile of the retail domain.
- **In Scope:**
  - Auditing all 114 retail tasks in `v1.0.1` for non-empty deterministic grading criteria.
  - Generating and committing `configs/retail_task_split.json` (`seed=42`, 50 held-out eval tasks, remaining ~64 train tasks).
  - Running `scripts/count_tokens.py` on `data/tau2/domains/retail/policy.md` and the 15 base retail tool schemas to record exact Qwen3 BPE token counts in `notes/retail_token_profile.json`.
- **Not in Scope:**
  - Running live LLM inference or user simulator calls.
  - Auditing non-retail domains (`airline`, `telecom`, `banking`).
- **Commands to Execute:**
  ```bash
  git clone --branch v1.0.1 --depth 1 https://github.com/sierra-research/tau2-bench.git /tmp/tau2-bench
  python scripts/audit_retail_tasks.py \
    --tasks-json /tmp/tau2-bench/data/tau2/domains/retail/tasks.json \
    --out configs/retail_task_split.json
  python scripts/count_tokens.py \
    --policy-file /tmp/tau2-bench/data/tau2/domains/retail/policy.md \
    --out notes/retail_token_profile.json
  ```
- **Acceptance Criteria (DoD):**
  - [ ] `configs/retail_task_split.json` exists, contains SHA-256 hash of `tasks.json`, lists excluded Issue #384 tasks (if any), and locks `eval_task_ids` (50 tasks) and `train_task_ids` (~64 tasks).
  - [ ] `notes/retail_token_profile.json` records the exact Qwen3 BPE token count of the retail `policy.md` ($P$) and confirms remaining $(S+H)$ token capacity under `2k (2048)`, `4k (4096)`, and `8k (8192)` budgets.
- **Output Artifacts:** `configs/retail_task_split.json`, `notes/retail_token_profile.json`.

---

### [CBA-9] Implement Per-Component Context Budget Controller (`src/budget_controller.py`)
- **Status:** ✅ **DONE** (`src/budget_controller.py`, `configs/sweep_grid.json`, `17/17` tests passing)
- **Environment:** 🟢 Local CPU / GitHub Codespaces
- **Proposal Milestone:** Milestone 2 (Python evaluation harness & per-component token budgeting)
- **Description:**  
  Build the deterministic Python module that takes a target total budget $B \in \{2048, 4096, 8192\}$, output reserve $R_{\text{out}} = 512$, policy text $P$, candidate tool schemas $S$ (15 base retail tools + $N$ distractor schemas), interaction history $H$, and an allocation ratio $\alpha \in \{0.75, 0.50, 0.25, \text{unconstrained}\}$, and enforces exact token limits $T(S) \le B_S$ and $T(H) \le B_H$ using the Qwen3 tokenizer prior to generation.
- **In Scope:**
  - Calculating remaining budget $B_{\text{rem}} = B - R_{\text{out}} - T(P)$ and component quotas $B_S = \lfloor \alpha B_{\text{rem}} \rfloor$, $B_H = B_{\text{rem}} - B_S$.
  - **Schema Allocation ($S$):** Ranking base + distractor tool schemas by lexical overlap (BM25 / TF-IDF cosine similarity) with the user conversation context and packing top-$k$ schemas (plus compact description fallback at 2k budget) until $T(S) \le B_S$.
  - **Distractor Tool Injection:** Loading $N \in \{0, 10, 25\}$ distractor tool schemas from external domains (e.g., $\tau^3$-bench `airline`/`telecom` or BFCL schemas) to test selection difficulty vs. budget displacement.
  - **History Allocation ($H$):** Always preserving the first user task prompt (`messages[0]`) and greedily retaining the most recent turn pairs from the end of history (`messages[-1], messages[-2], ...`) without splitting `(assistant_tool_call, tool_output)` pairs until $T(H) \le B_H$.
  - Unit tests in `tests/test_budget_controller.py` verifying strict token ceiling compliance.
- **Not in Scope:**
  - RL-trained dynamic context controllers (explicitly out of scope per proposal).
  - Modifying $\tau^3$-bench's internal database engine.
- **Acceptance Criteria (DoD):**
  - [ ] `src/budget_controller.py` implemented with clean dataclasses and deterministic behavior.
  - [ ] For any input prompt and any budget $B \in \{2048, 4096, 8192\}$, `count_chat_tokens(allocated_prompt) <= B - R_out` holds 100% of the time.
  - [ ] `pytest tests/test_budget_controller.py` passes with $\ge 5$ unit test cases (including 2k extreme pressure and pair-integrity checks).
- **Output Artifacts:** `src/budget_controller.py`, `configs/distractor_tools.json`, `tests/test_budget_controller.py`.

---

### [CBA-10] Implement 6 Deterministic Retail Policy Compliance Checkers (`src/compliance_checkers.py`)
- **Status:** ✅ **DONE** (`src/compliance_checkers.py` — `8`-rule Tier A+B engine + calibration metrics, `12/12` tests passing)
- **Environment:** 🟢 Local CPU / GitHub Codespaces
- **Proposal Milestone:** Milestone 4 (Deterministic Policy Compliance Engine)
- **Description:**  
  Implement an offline rule verification engine that ingests saved trajectory JSON logs and deterministically evaluates the 6 retail operational policy rules (`R1`–`R6`) that $\tau^3$-bench's terminal DB state grader fails to catch, computing both per-rule violation/omission flags and the headline **Silent Policy Violation Rate**.
- **In Scope:**
  - Implementing the 6 deterministic checkers defined in `PORTABLE_WORK_PACKET.md`:
    1. `R1_AUTH_BEFORE_LOOKUP`: User authenticated via `find_user_id_by_email` or `find_user_id_by_name_zip` before inspecting/mutating orders.
    2. `R2_EXPLICIT_CONFIRMATION`: Assistant requested confirmation and received an affirmative user turn before calling any mutating tool (`cancel_*`, `modify_*`, `return_*`, `exchange_*`).
    3. `R3_SINGLE_ORDER_MODIFICATION`: At most one item modification or exchange call per `order_id` in an episode.
    4. `R4_PENDING_VS_DELIVERED_STATUS`: Cancel/modify only invoked on `pending` orders; return/exchange only on `delivered` orders.
    5. `R5_PAYMENT_METHOD_VALIDITY`: Payment method ID used in return/exchange/modify belongs to the user's profile or original order.
    6. `R6_PREMATURE_HUMAN_TRANSFER`: `transfer_to_human_agents` called only when ground-truth criteria allow human transfer.
  - Computing `SilentViolation(traj) = (task_success == 1) and (any_rule_violated == True)`.
  - Unit tests in `tests/test_compliance_checkers.py` using synthetic compliant and non-compliant trajectories.
- **Not in Scope:**
  - Using an LLM-as-a-judge to score policy compliance (violates Hard Gate G3 and Proposal Declaration 4).
- **Acceptance Criteria (DoD):**
  - [ ] `src/compliance_checkers.py` implements all 6 rules and outputs a structured JSON compliance report per trajectory and aggregate dataset statistics.
  - [ ] `pytest tests/test_compliance_checkers.py` passes 100% of synthetic violation and compliance test cases.
- **Output Artifacts:** `src/compliance_checkers.py`, `tests/test_compliance_checkers.py`.

---

### [CBA-11] Compile & Submit Interim Report Document
- **Status:** ⏭️ **NEXT UP (#1)**
- **Environment:** 🟢 Local / Google Docs / Word (Can be done on any device using `PORTABLE_WORK_PACKET.md`)
- **Proposal Milestone:** Official Interim Report Submission
- **Description:**  
  Assemble the Interim Report using the copy-ready blueprint in `PORTABLE_WORK_PACKET.md` (Section 4), `notes/paper_notes.md` (`CBA-7`), `notes/benchmark_pinning.md` (`CBA-2`), and the token profile from `CBA-8`.
- **In Scope:**
  - Chapter 1: Problem Statement, Motivation, Research Questions (`RQ1`, `RQ1b`, `RQ2`, `RQ4`), and Hypotheses (`H1`, `H1b`, `H2`, `H4`).
  - Chapter 2: Literature Review & Comparative Matrix (6 core papers) + Formal 3-part Research Gap.
  - Chapter 3: Benchmark Selection & Pinning ($\tau^3$-bench Retail `v1.0.1`, Issue #384 audit, frozen 50-Eval / 64-Train split, exact BPE token profile).
  - Chapter 4: High-Level Design (HLD), System Architecture Diagram, Mathematical Budget Allocation Equations, Deterministic Compliance Rule Table (`R1`–`R6`), and Verifier-Filtered LoRA SFT Pipeline Specification.
- **Not in Scope:**
  - Final experimental results, LoRA ablation curves, or final bootstrap statistical tables (those belong to the Final Report in January).
- **Acceptance Criteria (DoD):**
  - [ ] Formatted Interim Report PDF/DOCX exported and submitted to Amrita portal / supervisor.
- **Output Artifacts:** `Interim_Report_Surya_B_AA.SC.P2MCA25014003.pdf`.

---

## 🔵 BLOCK 2: Cloud GPU Window 1 — Smoke Test & 50-Task Floor Pilot (~4–6 GPU Hours)

> **Pragmatic Rule:** Do not rent the GPU for Block 2 until `CBA-8`, `CBA-9`, and `CBA-10` pass `pytest` locally.

---

### [CBA-3] Cloud GPU VM Provisioning (24 GB VRAM) & Driver Validation
- **Status:** ⏳ **WAITING FOR GPU WINDOW 1**
- **Environment:** 🔵 Cloud Linux GPU VM (RunPod / Lightning.ai / Vast.ai / Lambda — RTX 3090 / 4090 / A10G / L4, Ubuntu 22.04)
- **Proposal Milestone:** Milestone 1
- **Description:**  
  Rent a single 24 GB VRAM Linux GPU instance, set a hard provider spending cap, verify `nvidia-smi` and Docker GPU passthrough, and clone `https://github.com/SuryaB98/context-budget-agent.git`.
- **In Scope:**
  - Provisioning 1x 24 GB GPU instance, verifying CUDA drivers, setting a billing alert/auto-stop timer, and logging hardware specs in `notes/vm_environment.md`.
- **Not in Scope:**
  - Multi-GPU clusters, Kubernetes orchestration, or custom Dockerfile builds.
- **Acceptance Criteria (DoD):**
  - [ ] `nvidia-smi` reports $\ge 24\text{ GB}$ VRAM.
  - [ ] Hardware specs (GPU model, driver, CUDA version, $/hr cost) recorded in `notes/vm_environment.md`.
- **Output Artifacts:** `notes/vm_environment.md`.

---

### [CBA-4] Deploy `Qwen3-1.7B` & `4B` with vLLM & Verify Hermes Tool Calling
- **Status:** ⏳ **WAITING FOR GPU WINDOW 1 (Scripts Already Implemented in `bae464e`)**
- **Environment:** 🔵 Cloud GPU Window 1
- **Proposal Milestone:** Milestone 1
- **Description:**  
  Execute `bash setup/run_vllm.sh Qwen/Qwen3-1.7B 8000` and `python scripts/test_tool_call.py` to confirm vLLM serves `Qwen3-1.7B` (and subsequently `Qwen3-4B`) with PagedAttention prefix caching enabled and parses structured tool calls via the Hermes parser.
- **In Scope:**
  - Running `setup/run_vllm.sh` and verifying `/health`, `/v1/models`, and structured `tool_calls` output via `scripts/test_tool_call.py`.
- **Not in Scope:**
  - Custom vLLM CUDA kernel modifications.
- **Acceptance Criteria (DoD):**
  - [ ] `python scripts/test_tool_call.py` exits with code `0` and saves `notes/vllm_test_response.json`.
- **Output Artifacts:** `notes/vllm_test_response.json`.

---

### [CBA-5] $\tau^3$-bench Retail 1-Task Smoke Run & 50-Task Baseline Floor Pilot
- **Status:** ⏳ **WAITING FOR GPU WINDOW 1**
- **Environment:** 🔵 Cloud GPU Window 1
- **Proposal Milestone:** Milestone 1 & Milestone 2
- **Description:**  
  Execute 1 retail task end-to-end (`runs/smoke_test_run_01/`) and then run the 50 held-out evaluation tasks (`eval_task_ids` from `configs/retail_task_split.json`) at unconstrained/8k baseline context on `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` (with `gpt-4o-mini` at `temperature=0.0` as the user simulator and hard $10 API cap) to verify the $\ge 15\%$ Floor Gate (G5).
- **In Scope:**
  - Running 1 smoke task + 50 eval tasks $\times$ 2 models at baseline 8k context.
  - Saving full trajectories, DB reward, communication reward, and token breakdowns (`scripts/count_tokens.py`).
- **Not in Scope:**
  - Running the full $S:H$ budget allocation grid (that is `CBA-13` in GPU Window 2).
- **Acceptance Criteria (DoD):**
  - [ ] Smoke test trajectory and token breakdown saved in `runs/smoke_test_run_01/`.
  - [ ] 50-task baseline pilot completes for both `Qwen3-1.7B` and `Qwen3-4B`; floor pass rate ($\ge 15\%$) verified and logged in `notes/pilot_floor_results.md`.
- **Output Artifacts:** `runs/smoke_test_run_01/`, `runs/pilot_50_baseline/`, `notes/pilot_floor_results.md`.

---

### [CBA-12] Hand-Label 30–50 Pilot Transcripts & Calibrate Compliance Checkers ($\ge 90\%$ Gate)
- **Status:** ⏳ **AFTER CBA-5**
- **Environment:** 🟢 Local CPU / Browser (Zero GPU Cost)
- **Proposal Milestone:** Milestone 4 (Calibration against manually inspected sample transcripts)
- **Description:**  
  Manually inspect 30–50 trajectories from `runs/pilot_50_baseline/`, record ground-truth human labels for Rules `R1`–`R6` in `notes/hand_labelled_compliance.csv`, run `src/compliance_checkers.py`, and verify $\ge 90\%$ agreement between the automated deterministic checkers and human labels.
- **In Scope:**
  - Manual annotation of 30–50 pilot transcripts across `R1`–`R6`.
  - Computing per-rule percentage agreement and Cohen's $\kappa$, refining any regex/state heuristic in `src/compliance_checkers.py` until agreement $\ge 90\%$.
- **Not in Scope:**
  - Labelling hundreds of sweep transcripts manually (the calibrated engine runs automatically on all subsequent sweeps).
- **Acceptance Criteria (DoD):**
  - [ ] `notes/hand_labelled_compliance.csv` committed with human annotations.
  - [ ] Automated checker agreement reaches $\ge 90\%$ on validated rules; calibration table saved in `notes/checker_calibration_report.md`.
- **Output Artifacts:** `notes/hand_labelled_compliance.csv`, `notes/checker_calibration_report.md`.

---

## 🔵 BLOCK 3: Cloud GPU Window 2 — Core Allocation & Serving Cost Sweep (~12–16 GPU Hours)

---

### [CBA-13] Primary Allocation Sweep (`2k/4k/8k` $\times$ $S:H$ Ratios $\times$ `1.7B/4B`) + Latency & Prefix-Cache Logging
- **Status:** ⏳ **WEEK 7**
- **Environment:** 🔵 Cloud GPU Window 2
- **Proposal Milestone:** Milestone 3 (Primary allocation experiments) & Milestone 5 (Serving cost profiling)
- **Description:**  
  Execute the primary controlled experiment grid on the 50 held-out evaluation tasks across `Qwen3-1.7B` and `Qwen3-4B`, 3 total budgets (`2k`, `4k`, `8k`), and 4 $S:H$ allocation splits ($75:25$, $50:50$, $25:75$, and unconstrained baseline) with injected distractor tool schemas. Log verified task success, silent policy violations (`src/compliance_checkers.py`), TTFT, end-to-end latency, and vLLM `/metrics` prefix-cache hit rates.
- **In Scope:**
  - Running the core grid (2 models $\times$ 3 budgets $\times$ 3 constrained ratios + baseline $\times$ 50 eval tasks, with 3 seeds on key comparison points).
  - Running distractor control probes (varying distractor tool count $N \in \{0, 10, 25\}$ at fixed schema quota vs. varying schema quota at fixed $N$).
  - Logging TTFT, generation latency, and vLLM `gpu_prefix_cache_hit_rate` for every run.
- **Not in Scope:**
  - Fine-tuning models (that is Block 4).
- **Acceptance Criteria (DoD):**
  - [ ] Complete trajectory logs, task success scores, compliance reports, and latency/cache metrics saved under `runs/core_sweep_rq1/`.
  - [ ] Summary CSV `notes/rq1_allocation_results.csv` generated for plotting trade-off curves.
- **Output Artifacts:** `runs/core_sweep_rq1/`, `notes/rq1_allocation_results.csv`.

---

### [CBA-14] Secondary Policy-Retrieval Experiment (RQ1b) & Prefix-Cache ON vs OFF Probe
- **Status:** ⏳ **WEEK 7**
- **Environment:** 🔵 Cloud GPU Window 2
- **Proposal Milestone:** Milestone 4 (Secondary policy retrieval) & Milestone 5 (PagedAttention prefix cache reuse)
- **Description:**  
  1. **RQ1b (Pinned vs. Retrieved Policy):** At `2k` and `4k` budgets (using the best $S:H$ split from `CBA-13`), compare full pinned `policy.md` against top-$k$ retrieved policy paragraphs to measure how much task success and silent policy violations degrade when policy rules are retrieved rather than pinned.  
  2. **H4 Cache Probe:** Compare sliding-window history trimming vs. history summarization with vLLM `--enable-prefix-caching` ON vs. OFF to prove how prompt mutation degrades prefix-cache hit rate and inflates real TTFT.
- **In Scope:**
  - 1 compact RQ1b sweep (2 budgets $\times$ pinned vs. top-$k$ policy on 50 eval tasks).
  - 1 compact H4 serving probe (20 tasks $\times$ cache ON/OFF $\times$ trim vs. summarize).
- **Not in Scope:**
  - Building an external vector database or RAG corpus outside `policy.md`.
- **Acceptance Criteria (DoD):**
  - [ ] RQ1b compliance vs. success table saved in `notes/rq1b_policy_retrieval_results.csv`.
  - [ ] H4 prefix-cache hit-rate and TTFT table saved in `notes/rq4_serving_cache_results.csv`.
- **Output Artifacts:** `notes/rq1b_policy_retrieval_results.csv`, `notes/rq4_serving_cache_results.csv`.

---

## 🔵 BLOCK 4: Cloud GPU Window 3 — LoRA SFT Pipeline, Ablations & Generalisation (~8–12 GPU Hours)

---

### [CBA-15] Construct Verifier-Filtered SFT Trajectory Dataset (From Train Split Only)
- **Status:** ⏳ **WEEK 8**
- **Environment:** 🔵 Cloud GPU Window 3 (or Teacher API + Local Filter)
- **Proposal Milestone:** Milestone 6 (Trajectory dataset construction & filter pipeline)
- **Description:**  
  Generate multi-turn execution trajectories strictly on the **~64 Training Tasks** (`train_task_ids` from `configs/retail_task_split.json`), filter to keep only trajectories that pass **both** $\tau^3$-bench deterministic grading ($\text{Reward} = 1.0$) **and** our 6 policy compliance checkers ($\text{PolicyCompliant} = 1$), and format each assistant tool-calling turn under budget-constrained contexts (`2k` and `4k` with distractor schemas) as a supervised training example.
- **In Scope:**
  - Generating and verifier-filtering training trajectories strictly on `train_task_ids` (zero overlap with `eval_task_ids`).
  - Formatting assistant-turn completion datasets (`data/sft_train_100pct.jsonl` and a 50% subsampled split `data/sft_train_50pct.jsonl`) with prompt/policy/schema token budgets applied.
- **Not in Scope:**
  - Including any task from `eval_task_ids` in SFT data generation (strict data leakage prohibition).
- **Acceptance Criteria (DoD):**
  - [ ] Automated assertion verifies `set(sft_task_ids).isdisjoint(set(eval_task_ids))`.
  - [ ] 100% and 50% verifier-filtered training JSONL files and dataset card saved in `notes/sft_data_card.md`.
- **Output Artifacts:** `src/build_sft_dataset.py`, `notes/sft_data_card.md`.

---

### [CBA-16] LoRA SFT Training & $2 \times 2$ Ablation Grid (Data Size $\times$ Rank on `1.7B` + `4B`)
- **Status:** ⏳ **WEEK 8–9**
- **Environment:** 🔵 Cloud GPU Window 3 (24 GB GPU using PyTorch + PEFT + TRL `SFTTrainer`)
- **Proposal Milestone:** Milestone 6 (LoRA supervised fine-tuning runs with rank and data ablations)
- **Description:**  
  Train LoRA adapters with assistant-turn loss masking across a compact $2 \times 2$ ablation grid on `Qwen/Qwen3-1.7B`: Dataset Size $\in \{50\%, 100\%\} \times$ LoRA Rank $r \in \{8, 32\}$ ($\alpha_{\text{LoRA}} = 2r$, target modules `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, `bf16`, gradient checkpointing). Train 1 LoRA adapter on `Qwen/Qwen3-4B` using the best rank configuration.
- **In Scope:**
  - 4 fast LoRA SFT runs on `Qwen3-1.7B` (`50%_r8`, `50%_r32`, `100%_r8`, `100%_r32`) + 1 LoRA SFT run on `Qwen3-4B` (`100%_r_best`).
  - Logging training/validation loss curves, wall-clock training time, and peak GPU VRAM usage.
- **Not in Scope:**
  - Full-parameter fine-tuning of 4B (exceeds 24 GB VRAM envelope) or RL/GRPO training (explicitly deferred to post-MCA research phase).
- **Acceptance Criteria (DoD):**
  - [ ] All 5 LoRA adapter checkpoints trained without OOM, loss curves logged, and adapter weights saved.
- **Output Artifacts:** `src/train_lora_sft.py`, `notes/sft_training_ablation_logs.json`.

---

### [CBA-17] Evaluate SFT Budget Recovery, Unseen Tool Catalogues & Language Retention
- **Status:** ⏳ **WEEK 10**
- **Environment:** 🔵 Cloud GPU Window 3 (vLLM serving LoRA adapters via `--enable-lora`)
- **Proposal Milestone:** Milestone 7 (Evaluation on budget recovery, held-out generalisation, unseen tool catalogues, and language capability retention)
- **Description:**  
  Evaluate the trained LoRA adapters on the **50 Held-Out Evaluation Tasks** across three dimensions promised in the proposal:
  1. **Budget Recovery:** Compare LoRA-tuned models at `2k` and `4k` budgets against untrained base models at `4k` and `8k` budgets (both task success and silent policy violation rate).
  2. **Unseen Tool Catalogue Generalisation:** Evaluate LoRA models when injected with 15 novel distractor tool schemas never seen during SFT training.
  3. **Language Capability Retention (Forgetting Check):** Evaluate base vs. LoRA models on a fixed 100-item general instruction/reasoning benchmark subset (e.g., GSM8K-100 / MMLU-100) to confirm no catastrophic forgetting.
- **In Scope:**
  - Running the 3 evaluation suites via vLLM `--enable-lora` and logging results into `notes/sft_evaluation_results.csv`.
- **Not in Scope:**
  - Retraining adapters on held-out eval tasks.
- **Acceptance Criteria (DoD):**
  - [ ] Budget recovery curves, unseen tool catalogue accuracy, and language retention scores saved in `notes/sft_evaluation_results.csv`.
- **Output Artifacts:** `notes/sft_evaluation_results.csv`.

---

## 🟢 BLOCK 5: Statistical Analysis, Failure Taxonomy & Final Dissertation (Local CPU)

---

### [CBA-18] Paired Statistical Tests, 10k Bootstrap CIs, Failure Taxonomy & Final Dissertation Package
- **Status:** ⏳ **WEEK 11–13 (Hard Experiment Freeze: Sun 3 Jan 2027 · Final Submission: Thu 15 Jan 2027)**
- **Environment:** 🟢 Local CPU / Browser
- **Proposal Milestone:** Milestone 8 & Milestone 9
- **Description:**  
  Run non-parametric 10,000-iteration paired bootstrap confidence intervals and paired significance tests across all experimental tables (`RQ1`, `RQ1b`, `RQ2`, `RQ4`), categorize failed trajectories into the 5-class **Failure Taxonomy** (1. Wrong Tool Selected, 2. Hallucinated/Invalid Argument, 3. Forgotten Multi-Turn User Constraint, 4. Silent Policy Breach, 5. Malformed Tool Call / Context Overflow), generate all publication-quality Matplotlib figures, tag the reproducible Git repository release `v1.0.0`, and compile the final MCA Dissertation PDF and Viva slides.
- **In Scope:**
  - `scripts/analyze_and_plot.py` producing all bootstrap CIs, $p$-values, and Matplotlib PDF/PNG figures.
  - Failure taxonomy classification script + manual verification on failed episodes.
  - Complete MCA Final Dissertation Report, Certificate Page, Reproducibility README, and Viva presentation deck.
- **Not in Scope:**
  - Running any new GPU experiments after Sunday, 3 January 2027.
- **Acceptance Criteria (DoD):**
  - [ ] All claims reported with 95% paired bootstrap confidence intervals.
  - [ ] Reproducible GitHub repository tagged `v1.0.0` with clean `pytest` suite and one-command plot generation.
  - [ ] Final Dissertation Report submitted by **Thursday, 15 January 2027**.
- **Output Artifacts:** `scripts/analyze_and_plot.py`, `figures/`, `Final_Dissertation_Report_Surya_B_21CSA391A.pdf`.
