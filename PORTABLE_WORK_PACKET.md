# 📦 Context Budget Agent — Self-Contained Portable Work Packet

**Email this file to yourself (`PORTABLE_WORK_PACKET.md`).**  
It contains everything you need to progress on your MCA Minor Project (**21CSA391A**) from **any device** (browser, borrowed laptop, tablet, or phone) while away from your primary laptop.

---

## 1. Quick Identity & Project Snapshot (From Signed Proposal PDF)

- **Title:** *Token Exchange Rates: How Small Model Agents Should Spend a Fixed Context Budget*
- **Student:** Surya B · **Roll No.:** `AA.SC.P2MCA25014003` · **Course:** Amrita Online MCA (`21CSA391A`)
- **GitHub Repo:** `https://github.com/SuryaB98/context-budget-agent`
- **Pinned Benchmark:** `sierra-research/tau2-bench` at release tag **`v1.0.1`** (`retail` domain, 114 tasks; deterministic DB + substring communication grading; BFCL multi-turn as backup)
- **Candidate Models:** `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` (served via vLLM with `--enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching --max-model-len 8192`)
- **Fixed Token Budgets:** **2k (`2048`), 4k (`4096`), and 8k (`8192`)** total BPE tokens across controlled Schema ($S$) vs. History ($H$) ratios ($75:25$, $50:50$, $25:75$, and unconstrained baseline), with Policy ($P$) pinned in RQ1 and top-$k$ retrieved in RQ1b.

---

## 2. How to Work From Any Device Right Now (Zero Setup Required)

### Mode 1: You Only Have a Web Browser (Any PC / Chromebook / iPad)
1. **Open GitHub Web IDE (Instant, Free):**
   - Go to `https://github.dev/SuryaB98/context-budget-agent` (or press `.` on your GitHub repo page).
   - You get full VS Code in your browser to edit Markdown, Python scripts, and report chapters and commit directly to `main`.
2. **Open GitHub Codespaces (Full Linux Terminal + Python 3.12 in Browser):**
   - On `https://github.com/SuryaB98/context-budget-agent`, click **Code → Codespaces → Create codespace on main**.
   - We added `.devcontainer/devcontainer.json` to your repo, so it automatically installs Python 3.12, `uv`, `transformers`, `tokenizers`, `pytest`, `scipy`, and `matplotlib`.
   - Inside the Codespace terminal, run:
     ```bash
     # 1. Run authoritative Qwen3-1.7B token accounting (CBA-6) on CPU:
     python scripts/count_tokens.py --out notes/demo_token_breakdown.json

     # 2. Clone tau2-bench v1.0.1 and audit all 114 retail tasks (Issue #384) + freeze Train/Eval split:
     git clone --branch v1.0.1 --depth 1 https://github.com/sierra-research/tau2-bench.git /tmp/tau2-bench
     python scripts/audit_retail_tasks.py \
       --tasks-json /tmp/tau2-bench/data/tau2/domains/retail/tasks.json \
       --out configs/retail_task_split.json

     # 3. Profile exact token counts of the real tau3-bench v1.0.1 retail policy:
     python scripts/count_tokens.py \
       --policy-file /tmp/tau2-bench/data/tau2/domains/retail/policy.md \
       --out notes/retail_policy_token_profile.json
     ```

### Mode 2: Free Browser GPU (Google Colab / Kaggle) for PyTorch & LoRA Practice
1. Open [Google Colab](https://colab.research.google.com/) (select **Runtime → Change runtime type → T4 GPU**).
2. Run in the first cell:
   ```python
   !git clone https://github.com/SuryaB98/context-budget-agent.git
   %cd context-budget-agent
   !pip install -q transformers tokenizers datasets peft trl accelerate bitsandbytes
   !python scripts/count_tokens.py
   ```
3. You can test your hand-written PyTorch training loop and TRL `SFTTrainer` LoRA adapter training right inside Colab's free T4 GPU.

### Mode 3: Phone / Tablet / Offline Reading & Writing (No Terminal Needed)
If you only have a phone, tablet, or Google Docs/Word while away from your laptop, complete the **3 Offline Tasks** in Section 3 below.

---

## 3. Three High-Value Tasks You Can Complete While Away From Your Laptop

### Task A: Finalize Your Interim Report Draft (Copy-Ready Structure Below)
Your Interim Report requires: **1. Problem Statement, 2. Background & Related Work, 3. Benchmark Selection & Pinning, and 4. High-Level Design (Modules & Algorithms to be Implemented).**  
Use the ready-to-paste draft in **Section 4 of this document** to assemble your Interim Report in Google Docs / Word (`Times New Roman`, 10–11 pt).

### Task B: Select & Refine the 6 Deterministic Retail Policy Rules (Milestone 4)
In $\tau^3$-bench Retail, the policy requires the customer service agent to obey strict operational rules that the mock database tools **do not enforce automatically** (meaning an agent can pass the DB check while silently breaking a rule).  
Review and refine the formal specification of our **6 Deterministic Policy Compliance Checkers** below so we can code them immediately:

| Rule ID | Retail Policy Rule Name | Trigger Condition in Trajectory Log | Deterministic Violation Check (Violation vs. Omission) |
| :--- | :--- | :--- | :--- |
| **`R1_AUTH_BEFORE_LOOKUP`** | Mandatory User Authentication | Agent calls `get_order_details`, `get_user_details`, or any `modify_*`/`cancel_*`/`return_*`/`exchange_*` tool. | **Omission Violation** if no prior turn successfully executed `find_user_id_by_email` or `find_user_id_by_name_zip` matching the target user. |
| **`R2_EXPLICIT_CONFIRMATION`** | Explicit Customer Confirmation Before Mutation | Agent calls any state-mutating tool: `cancel_pending_order`, `modify_pending_order_*`, `return_delivered_order_items`, `exchange_delivered_order_items`, `modify_user_address`. | **Omission Violation** if the assistant did not list the action details (items/order ID/payment method) and receive an affirmative user turn (`yes` / `confirm` / `go ahead` / `sure`) between proposing the change and invoking the mutating tool. |
| **`R3_SINGLE_ORDER_MODIFICATION`** | At Most One Modification/Exchange Per Order | Agent calls `modify_pending_order_items` or `exchange_delivered_order_items` more than once on the same `order_id` within an episode. | **Action Violation** if `count(mutating_calls_on_same_order_id) > 1`. |
| **`R4_PENDING_VS_DELIVERED_STATUS`** | Order Status Precondition Compliance | Agent attempts `cancel_pending_order` / `modify_pending_order_*` on a non-`pending` order, or `return_delivered_*` / `exchange_delivered_*` on a non-`delivered` order. | **Action Violation** if prior `get_order_details` output showed an incompatible `status` field. |
| **`R5_PAYMENT_METHOD_VALIDITY`** | Registered Payment / Gift Card Constraint | Agent passes a `payment_method_id` to a modification/return/exchange tool. | **Action Violation** if `payment_method_id` was not present in the authenticated user's `get_user_details` profile or original order payment methods. |
| **`R6_PREMATURE_HUMAN_TRANSFER`** | Transfer to Human Only When Out-of-Scope | Agent calls `transfer_to_human_agents`. | **Action Violation** if the task's ground-truth evaluation criteria do not include `transfer_to_human_agents` (i.e., the task was solvable via standard retail tools). |

### Task C: Deep Paper Reading Checklist (`CBA-7`)
While away from your laptop, read the PDFs/HTML of these 6 papers (links below) and jot down 3 bullet points per paper (1 key number, 1 limitation, 1 sentence comparing it to our $2\text{k}/4\text{k}/8\text{k}$ $S$-vs-$H$ study):
1. **Tool-Schema Compression:** `https://arxiv.org/abs/2605.26165`
2. **Control Under Compression:** `https://arxiv.org/html/2608.01056`
3. **Governance Decay:** `https://arxiv.org/html/2606.22528v2`
4. **PA-Tool:** `https://arxiv.org/pdf/2510.07248`
5. **ToolRL:** `https://arxiv.org/abs/2504.13958`
6. **Debunk the Myth of SFT Generalization:** `https://arxiv.org/html/2510.00237v1`

---

## 4. Copy-Ready Interim Report Blueprint (Sections 1–4)

You can copy-paste and format the sections below directly into your Interim Report document while away from your laptop:

### Section 1: Problem Statement & Objectives
Deploying small open-weight language models ($\le 4\text{B}$ parameters, such as `Qwen3-1.7B` and `Qwen3-4B`) as autonomous, multi-turn tool-calling agents is fundamentally constrained by fixed context window budgets. During interactive multi-turn execution, three distinct context components compete for the serialized input budget $B \in \{2048, 4096, 8192\}$ tokens:
1. **Tool Schemas ($S$):** Structured JSON function specifications for available APIs and distractor tools.
2. **Policy Context ($P$):** Domain-specific operational rules and governance constraints.
3. **Interaction History ($H$):** Cumulative user turns, assistant responses, and intermediate tool execution outputs.

When $S + P + H + R_{\text{out}} > B$ (where $R_{\text{out}}$ is the reserved generation budget), practitioners face an unquantified trade-off: preserving $S$ while aggressively pruning $H$ causes the agent to forget multi-turn user constraints and prior tool outputs, whereas preserving $H$ while compressing or pruning $S$ induces tool-selection errors and argument hallucinations. Furthermore, existing benchmarks evaluate only terminal database state ($\text{DB\_Reward} \times \text{Communicate\_Reward}$), overlooking **silent policy violations**—episodes where an agent achieves the target database state while bypassing mandatory policy rules (such as user authentication or explicit pre-mutation confirmation).

**Core Objectives:**
1. **Context Allocation Trade-off (RQ1 & RQ1b):** Quantify how varying the $S:H$ allocation ratio ($75:25$, $50:50$, $25:75$, and unconstrained baseline) across **2k, 4k, and 8k** BPE token budgets affects verified task success and silent policy compliance on $\tau^3$-bench Retail (`v1.0.1`), with policy $P$ held constant in primary runs and retrieved (top-$k$ chunks) in secondary experiments.
2. **Post-Training Budget Recovery (RQ2):** Evaluate whether parameter-efficient fine-tuning (LoRA SFT across dataset-size and rank ablations) enables a budget-constrained small model (e.g., at 2k or 4k tokens) to recover the accuracy of an untrained model at a larger budget (4k or 8k), while testing generalization to unseen tool catalogues and language capability retention.
3. **Real-World Serving Cost Profiling (RQ4):** Measure how context manipulation strategies affect vLLM **PagedAttention prefix-cache hit rates**, **Time-to-First-Token (TTFT)**, end-to-end latency, and throughput on a single 24 GB GPU.

---

### Section 2: Related Work & Research Gap
*(See `notes/paper_notes.md` in the repository for the full 6-paper comparison matrix and formal 3-part Research Gap statement covering `arXiv:2605.26165`, `arXiv:2608.01056`, `arXiv:2606.22528`, `arXiv:2510.07248`, `arXiv:2504.13958`, and `arXiv:2510.00237`.)*

---

### Section 3: Benchmark Selection & Immutable Pinning
*(See `notes/benchmark_pinning.md` in the repository.)*
- **Selected Testbed:** $\tau^3$-bench Retail (`sierra-research/tau2-bench`, pinned to release tag **`v1.0.1`**), containing 114 multi-turn customer service tasks, 15 retail database tools, and a comprehensive written operational policy (`policy.md`).
- **Rationale for `v1.0.1`:** Incorporates 26 critical retail grading bug fixes from `v1.0.0` while retaining 100% deterministic database state verification and substring communication scoring (avoiding the non-deterministic LLM-judge grading introduced in `v2.0.0`).
- **Quality Filter (Issue #384) & Data Hygiene:** Implemented in `scripts/audit_retail_tasks.py`. All 114 tasks are audited for non-empty `reward_basis` and evaluation assertions; valid tasks are partitioned via a seeded deterministic split (`seed=42`) into **50 Held-Out Evaluation Tasks** and **~64 Training Tasks** (used exclusively for SFT trajectory generation to guarantee zero train/test leakage).

---

### Section 4: High-Level Design (HLD) & Core Algorithms

#### 4.1 System Architecture Modules
1. **Module 1 — Native BPE Token Accounting Engine (`scripts/count_tokens.py`):**
   Uses each model's official Hugging Face tokenizer (`Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`) and `apply_chat_template(..., tools=tools, tokenize=True)` to compute exact serialized token counts for $P$, $S$, $H$, and chat-template framing markers before every generation turn.
2. **Module 2 — Per-Component Context Budget Controller (`src/budget_controller.py`):**
   Given total window budget $B \in \{2048, 4096, 8192\}$, output reserve $R_{\text{out}} = 512$, pinned policy tokens $T(P)$, and target split ratio $\alpha \in \{0.75, 0.50, 0.25\}$:
   $$B_{\text{rem}} = B - R_{\text{out}} - T(P)$$
   $$B_S = \lfloor \alpha \cdot B_{\text{rem}} \rfloor, \quad B_H = B_{\text{rem}} - B_S$$
   - **Schema Allocator ($S \le B_S$):** Ranks candidate tool schemas (base retail tools + $N$ injected distractor tools) by lexical/embedding similarity to the active user query and packs top-$k$ schemas until $T(S) \le B_S$ (or applies compact schema description stripping at extreme 2k pressure).
   - **History Allocator ($H \le B_H$):** Preserves the initial user goal turn and greedily retains the most recent dialogue and tool-output turns $(m_{t}, m_{t-1}, \dots)$ in valid `(assistant_tool_call, tool_response)` pairs until $T(H) \le B_H$ (with structured history summarization as the secondary H4 cache-breaking condition).
3. **Module 3 — Deterministic Policy Compliance Engine (`src/compliance_checkers.py`):**
   Executes offline over saved trajectory logs $\tau = \{(r_1, a_1, o_1), \dots, (r_T, a_T, o_T)\}$ to evaluate Rules `R1`–`R6`, outputting per-episode `task_success` ($\in \{0, 1\}$), `policy_compliant` ($\in \{0, 1\}$), and the **Silent Policy Violation Indicator**:
   $$\text{SilentViolation}(\tau) = \mathbb{I}\big[\text{TaskSuccess}(\tau) = 1 \;\wedge\; \text{PolicyCompliant}(\tau) = 0\big]$$
4. **Module 4 — Verifier-Filtered LoRA SFT Pipeline (`src/sft_trainer.py`):**
   Constructs budget-constrained single-step tool-calling decision instances solely from the **Train** task split, filters trajectories requiring $\text{TaskSuccess}(\tau) = 1 \wedge \text{PolicyCompliant}(\tau) = 1$, and trains LoRA adapters (`peft` + `trl.SFTTrainer`) with assistant-turn loss masking across the $2 \times 2$ ablation grid ($\text{DataSize} \in \{50\%, 100\%\} \times \text{Rank } r \in \{8, 32\}$).
5. **Module 5 — vLLM Serving & Prefix-Cache Profiler (`src/serving_profiler.py`):**
   Records client-side **Time-to-First-Token (TTFT)**, inter-token latency, and scrapes vLLM's Prometheus `/metrics` endpoint (`vllm:gpu_prefix_cache_hit_rate`) before and after each episode to test Hypothesis H4 (whether prompt rewriting breaks PagedAttention KV-cache reuse).

---

## 5. Re-Baselined 8-Week Execution Schedule (22 Nov 2026 – 17 Jan 2027)

| Week | Dates | Engineering & Experiments (Proposal Milestones) | Writing & Documentation | Exit / Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Wk 5/6 (Now)** | **22–29 Nov** | **1. Local CPU Sprint:** Audit 114 retail tasks (`scripts/audit_retail_tasks.py`); run `scripts/count_tokens.py` on retail policy & schemas (`CBA-6`); build budget controller & 6 deterministic compliance checkers (`R1`–`R6`).<br>**2. GPU Window 1 (4–6 hrs):** Provision 24GB GPU (`CBA-3`), launch vLLM (`CBA-4`), run 1-task smoke test (`CBA-5`) + 50-task Floor Pilot on 1.7B & 4B. Hand-label 30–50 transcripts to calibrate checkers. | **Submit Interim Report** (Problem, Background, Related Work `CBA-7`, Benchmark Pinning, HLD & Algorithms). | ✅ Interim Report submitted.<br>✅ `count_tokens.py` & task split committed.<br>✅ 50-task pilot & checker calibration done. |
| **Wk 7** | **30 Nov – 6 Dec** | **GPU Window 2 — Core Sweep (Milestones 3, 4, 5):**<br>• Primary allocation grid: `Qwen3-1.7B` & `4B` $\times$ `{2k, 4k, 8k}` $\times$ 4 $S/H$ splits ($75:25$, $50:50$, $25:75$, baseline).<br>• Log TTFT, generation latency, and vLLM prefix-cache hit rates.<br>• Secondary policy-retrieval experiment (pinned vs top-$k$ policy chunks) and cache-ON vs cache-OFF probe.<br>• Offline compliance checkers over all trajectories for **silent policy violation rates**. | Draft **Chapter 3 (Methodology)** and **Chapter 4.1–4.3 (Allocation, Compliance & Latency Results)**. | ✅ All RQ1, RQ1b, and RQ4 (serving cost) raw logs saved. |
| **Wk 8** | **7–13 Dec** | **SFT Data Pipeline & First LoRA Training (Milestone 6 — Part 1):**<br>• Generate budget-constrained trajectories strictly on the **~64 Train tasks** (zero eval leakage).<br>• Filter trajectories with deterministic verifier + compliance checkers.<br>• Train baseline LoRA adapter on `Qwen3-1.7B` using TRL `SFTTrainer` with assistant-turn loss masking. | Draft **Chapter 3.5 (SFT Pipeline & Verifier Filtering)** + dataset card. | ✅ Verifier-filtered SFT dataset frozen.<br>✅ First `Qwen3-1.7B` LoRA adapter trained. |
| **Wk 9** | **14–20 Dec** | **LoRA Ablations & 4B Scaling (Milestone 6 — Part 2):**<br>• Run $2 \times 2$ ablation grid on `Qwen3-1.7B`: Dataset Size (`50%`, `100%`) $\times$ LoRA Rank (`r=8`, `r=32`).<br>• Train LoRA adapter on `Qwen3-4B` using the winning rank/hyperparameters. | Add ablation training curves and compute table to **Chapter 4.4**. | ✅ All LoRA adapters (`1.7B` ablations + `4B`) trained and saved. |
| **Wk 10** | **21–27 Dec** | **Budget Recovery, Unseen Tools & Forgetting Eval (Milestone 7):**<br>• Evaluate fine-tuned `1.7B` and `4B` adapters on the **50 Held-Out Eval tasks** at `2k` and `4k` budgets vs base models at `4k` and `8k`.<br>• **Unseen Tool Catalogue Generalisation:** Evaluate adapters with novel distractor tool schemas not present in training.<br>• **Language Capability Retention:** Run 100-item general capability check (base vs LoRA) to verify no catastrophic forgetting. | Draft **Chapter 4.4–4.5 (Budget Recovery, Generalisation & Retention Results)**. | ✅ All experimental evaluations complete. |
| **Wk 11** | **28 Dec – 3 Jan** | **Statistical Analysis & Failure Taxonomy (Milestone 8):**<br>• Paired statistical tests and 10,000-resample bootstrap CIs across all comparative claims.<br>• Categorize failed trajectories into the **Failure Taxonomy** (Wrong Tool, Hallucinated Argument, Lost History Constraint, Silent Policy Breach, Malformed Call).<br>• Generate all final publication-quality Matplotlib figures. | Draft **Chapter 5 (Failure Taxonomy, Statistical Analysis & Discussion)**. | ✅ **Hard Experiment Freeze (Sun 3 Jan 2027).** |
| **Wk 12** | **4–10 Jan** | **Full Dissertation Assembly & Reproducibility Audit (Milestone 9 — Part 1):**<br>• Compile complete dissertation report + clean and document Git repository (`README.md`, frozen configs, one-click evaluation & plotting scripts). | Send complete final report draft to Guide/Supervisor for review. | ✅ Full Dissertation Draft complete. |
| **Wk 13** | **11–17 Jan** | **Final Polish, Submission & Viva Prep (Milestone 9 — Part 2):**<br>• Incorporate supervisor feedback, finalize Certificate & Declaration pages, tag `v1.0.0` release, build Viva slides + demo. | **Submit Final Dissertation Report by Thu 15 Jan 2027.** | 🎉 **Final Report Submitted & Viva Ready.** |

---

## 6. Checklist When You Return to This Laptop (Or Log Into a Cloud GPU)

- [ ] Run `git pull origin main` to pull any edits you made in `github.dev`, Codespaces, or Colab.
- [ ] Run the Local CPU Audit (`scripts/audit_retail_tasks.py` and `scripts/count_tokens.py` on the `tau2-bench` `v1.0.1` retail files).
- [ ] Submit the Interim Report PDF/DOCX.
- [ ] Launch **GPU Window 1** (`bash setup/run_vllm.sh Qwen/Qwen3-1.7B 8000` $\to$ `python scripts/test_tool_call.py` $\to$ 1-task smoke run & 50-task floor pilot).
