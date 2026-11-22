# [CBA-8] Pragmatic Execution Plan: Audit $\tau^3$-bench `v1.0.1` Retail, Fix Token Accounting & Freeze Leak-Free Split

**Ticket:** `[CBA-8]` (Block 1, Step #1 — Zero GPU Required)  
**Environment:** Local CPU (Windows PowerShell / WSL2) or GitHub Codespaces  
**Estimated Time:** 60–90 minutes  
**Pre-Flight Audit Date:** 2026-11-22  
**Target Benchmark Pin:** `sierra-research/tau2-bench` tag `v1.0.1` (commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`)

---

## 1. Executive Summary & Pre-Flight Audit Findings (Why Naive CBA-8 Would Fail)

Running CBA-8 against the actual `sierra-research/tau2-bench@v1.0.1` repository and `Qwen/Qwen3-1.7B` / `Qwen/Qwen3-4B` tokenizers uncovered **six critical empirical facts and code defects**. Executing the current `scripts/count_tokens.py` and `scripts/audit_retail_tasks.py` without fixing these first would silently produce wrong token measurements, a broken `2k` experiment configuration, vacuous `100%` passes, and train/eval data leakage.

### Finding 1: Silent `transformers 5.x` Bug in `scripts/count_tokens.py`
- In `transformers 5.24.0`, `tokenizer.apply_chat_template(messages, tools=tools, tokenize=True)` returns a `BatchEncoding` dictionary (`{"input_ids": ..., "attention_mask": ...}`) rather than a flat `list[int]`.
- Because `scripts/count_tokens.py` evaluates `len(token_ids)` directly, it returns **`2`** (the number of dictionary keys) for every single call, causing `p_templated`, `s_templated`, and every per-tool token delta to evaluate to **`0`**.
- **Additional omission:** `scripts/count_tokens.py` currently counts raw `policy.md` without `tau2`'s actual `AGENT_INSTRUCTION` + `<policy>...</policy>` wrapper (`+85` tokens) and uses 2 mock tools rather than the real **16** retail tools.

### Finding 2: Real Qwen3 BPE Token Footprint Breaks Naive `2k` Total Budget Math
Measuring the real `tau2-bench@v1.0.1` retail prompt and OpenAI tool schemas with both `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` tokenizers (which produce **identical** token counts on the retail prompt and tools) reveals:

| Prompt Component | Real Measurement (`v1.0.1` Retail + Qwen3 Tokenizer) | Notes |
| :--- | :--- | :--- |
| `policy.md` raw size | `6,699` chars · `1,158` words · **`1,407` raw BPE tokens** | Retail domain policy |
| Templated Policy ($P$) | **`1,492` tokens** | `AGENT_INSTRUCTION` + `<policy>...</policy>` + `policy.md` |
| Templated Tools ($S_{\text{all}}$) | **`3,382` tokens** | **16** retail tools (`7` WRITE, `7` READ, `2` GENERIC; range `161`–`477` tokens/tool; earlier docs said 15, missing `get_item_details`) |
| Turn 1 Baseline ($P + S_{\text{all}} + \text{framing}$) | **`4,883` tokens** | Before any conversation history or user task prompt |
| Oracle History ($H_{\text{oracle}}$ across 114 tasks) | `min = 66`, `p50 = 1,777`, `mean = 2,305`, `p90 = 5,036`, `max = 7,016` | Expected gold tool-call + output replay + initial user goal |
| Minimal Needed Tools ($S_{\text{min}}$ across 114 tasks) | `min = 0`, `p50 = 821`, `mean = 794`, `p90 = 1,127`, `max = 1,681` | Only the distinct tools actually called by oracle trajectory |

- **What happens under a literal Total Context Window ($B_{\text{total}} \in \{2048, 4096, 8192\}$ with `output_reserve = 512`):**
  - **`2048` (`max_input = 1,536`):** Pinning $P = 1,492$ leaves only **`44` tokens** for $S + H$ combined. It is **mathematically impossible** to include even the single cheapest retail tool (`list_all_product_types` at `161` tokens), and `P + S_all` overflows by **`-3,347` tokens** (`left_for_H = -3347`).
  - **`4096` (`max_input = 3,584`):** Leaves **`2,092` tokens** for $S + H$ after $P$, overflowing `P + S_all` by **`-1,299` tokens** (`left_for_H = -1299`) and forcing aggressive schema pruning from Turn 1.
  - **`8192` (`max_input = 7,680`):** Leaves **`2,797` tokens** for $H$ after `P + S_all + framing`.
- **What happens under an Allocatable $(S + H)$ Budget ($B_{\text{alloc}} \in \{2048, 4096, 8192\}$, with $P = 1,492$ pinned separately):**
  - `S_min + H_oracle` fits in **`42 / 114` (36.8%)** tasks at `2k`, **`82 / 114` (71.9%)** tasks at `4k`, and **`113 / 114` (99.1%)** tasks at `8k`.

### Finding 3: Three Factual Errors in `notes/benchmark_pinning.md` About `v1.0.1`
Direct inspection of `tau2-bench@v1.0.1` source code (`src/tau2/evaluator/evaluator.py`, `evaluator_nl_assertions.py`, `config.py`, `cli.py`, `RELEASE_NOTES.md`) shows:
1. **No `--communicate-scorer` CLI flag & `COMMUNICATE` is unused:** `COMMUNICATE` appears in `reward_basis` for **`0 / 114`** retail tasks (`reward_basis` is `['DB', 'NL_ASSERTION']` on **`112 / 114`** tasks and `['DB']` on **`2 / 114`** tasks — IDs `33` and `34`).
2. **LLM Judge Active on 40 / 114 Tasks:** **`40 / 114`** tasks (`29` in official `train`, `11` in official `test`) define non-empty `nl_assertions`, which invokes `NLAssertionsEvaluator` using `gpt-4.1-2025-04-14` (`temperature = 0.0`). The other `74` tasks short-circuit `NL_ASSERTION` to `1.0` with no LLM call. Meanwhile, `36` tasks have `communicate_info` (`33` overlap with `nl_assertions`, while `3` tasks — `30, 31, 32` — have `communicate_info = ['746342064230']` and `nl_assertions = None`, leaving them silently ungraded under default `EvaluationType.ALL`).
3. **Exact Commit & Release Note Correction:** Tag `v1.0.1` resolves to commit **`fc0055dc4e0a316c3f83133267fbd6faaa770992`** (annotated tag `b711c1ead46f55111bf765cf44d5da8bacc2d28c`), and `tasks.json` SHA-256 is **`f44563d8f36362e2dc58282bb02cfb85010d22f27d9439808758d3e9b2bc16d2`**. (Also note that `v1.0.1` specifically bundles `banking_knowledge` grading fixes on top of earlier retail task fixes.)

### Finding 4: True Vacuous-Pass Defect (`scripts/audit_retail_tasks.py` Misses It)
- In `v1.0.1` retail `tasks.json`, `0` tasks have `evaluation_criteria is None` and `0` tasks have an empty `reward_basis`. As a result, the current `scripts/audit_retail_tasks.py` filters **`0` tasks**.
- **The actual vacuous-pass defect:** **`6` tasks (`10, 12, 25, 50, 57, 65`)** have **zero expected `WRITE` tool calls AND zero graded `nl_assertions`**.
  - Specifically, transfer tasks `10, 12, 50` rely solely on `transfer_to_human_agents` (`ToolType.GENERIC`), which returns `"Transfer successful"` without mutating `db.json`—so `gold_db_hash == initial_db_hash`. Read-only/refusal tasks `25, 57, 65` likewise perform no DB mutations and have `nl_assertions = None`.
  - Under `EvaluationType.ALL`, an agent that does **nothing at all** scores **`1.0` (100% false pass)** on all `6` tasks!
- Conversely, Task `24` has `actions = []` (empty actions) because the user cancels their mind on the grill, **but is NOT vacuous** because it has a graded `nl_assertion` (`'Agent should tell the user the materials of the two t-shirts are polyester and cotton.'`) and `communicate_info = ['polyester', 'cotton']`.
- Excluding the `6` vacuous tasks leaves **`108` strictly non-vacuous retail tasks** (`70` in official `train`, `38` in official `test`).

### Finding 5: Severe Persona / Entity Leakage in Official `split_tasks.json` & Naive Random Splits
- `tau2-bench@v1.0.1` ships `data/tau2/domains/retail/split_tasks.json` (`train = 74`, `test = 40`, `base = 114`, `overlap = 0`).
- However, all `114` retail tasks are built from only **`63` distinct customer personas** (`known_info` strings), with **`31` multi-task persona groups covering `82` tasks**.
- **`20` of those `31` groups cross Sierra's official `train`/`test` boundary** (e.g., Noah Patel: Task `33` in `test` vs. `34` in `train`; Mei Kovacs: `6, 7, 8` in `train` vs. `9` in `test`; Mia Garcia: `10, 11` in `train` vs. `12` in `test`; Isabella Johansson: `25, 28, 29` in `train` vs. `26, 27` in `test`; Olivia Lopez: `30, 31` in `train` vs. `32` in `test`; Daiki Sanchez: `36, 38` in `test` vs. `37` in `train`; Fatima Johnson: `15, 16` in `train` vs. `17` in `test`; Mei Davis: `18` in `test` vs. `19` in `train`).
- A naive task-level random split (like the current `audit_retail_tasks.py`) or ungrouped use of `split_tasks.json` would leak customer names, order histories, and near-identical task templates from LoRA SFT `train` into `eval`. CBA-8 must enforce a **persona-grouped stratified split** over the `108` non-vacuous tasks with **zero persona overlap**.

### Finding 6: Offline `litellm` Process-Exit Hang Pitfall
- Importing any `tau2` module initializes `litellm`, which spawns background threads attempting to download `model_prices_and_context_window.json` from GitHub. In offline/sandboxed shells, scripts that import `tau2` hang or emit network warnings unless `LITELLM_LOCAL_MODEL_COST_MAP=True` is set.
- **Pragmatic architecture fix:** Extract `configs/retail_tools_openai.json` and `configs/retail_system_prompt.txt` once via a standalone extractor script, so `audit_retail_tasks.py`, `count_tokens.py`, `budget_controller.py`, and all `pytest` suites run offline in `<2` seconds with **zero** `tau2`/`litellm` imports.

---

## 2. Three Design Decisions to Lock During CBA-8

| Decision ID | Design Question | Options Considered | Recommended Choice | Concrete Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | How to interpret the **`2k / 4k / 8k`** budgets (`MCA_Project_Proposal_Amrita_Revised_Written_Final.pdf`) given `P = 1,492` and `S_all = 3,382` | **(A)** Literal total window `B_total` (only `44` tokens left at `2k`!)<br>**(B)** **Allocatable $(S + H)$ budget** $B_{\text{alloc}} \in \{2048, 4096, 8192\}$ with `P = 1,492` pinned separately and `R_out = 512` | **Option (B): Allocatable $(S + H)$ Budget** (`2k / 4k / 8k` for $S + H$), setting `vLLM --max-model-len 12288` | Literal `2k` total context makes tool-calling impossible (`44 < 161` min tool). Under $B_{\text{alloc}}$, `S_min + H_oracle` fits `36.8%` (`2k`), `71.9%` (`4k`), and `99.1%` (`8k`) of tasks—creating a clean difficulty gradient! Record the exact literal-window infeasibility proof in `notes/retail_token_profile.json` and retain `RQ1b` (policy retrieval) to test compressing `P` itself. |
| **D2** | How to construct the frozen Train / Eval task split over the `108` non-vacuous tasks | **(A)** Official `split_tasks.json` (`70` train / `38` test after vacuous filter)<br>**(B)** Naive random `50/58` split<br>**(C)** **Persona-grouped stratified split** (`seed = 42`, ~50 Eval / ~58 Train) | **Option (C): Persona-Grouped Stratified Split** (`0` shared `known_info` personas) | Both (A) and (B) suffer from severe persona leakage (`20/31` multi-task customer groups cross the official train/test boundary). Stratifying grouped clusters by `has_nl_assertions` and `H_oracle` quartile prevents SFT leakage while keeping eval size near `50`. Tag every task with its `official_split` metadata for comparability. |
| **D3** | How to satisfy **Hard Gate G3** (100% deterministic grading) when `40/114` tasks rely on `gpt-4.1` `nl_assertions` | **(A)** Drop all `40` `nl_assertions` tasks<br>**(B)** Rely purely on `gpt-4.1` judge<br>**(C)** **Dual-metric reporting:** Deterministic `DB + COMMUNICATE` Substring Score + Cached `DB × NL_ASSERTION` Score | **Option (C): Dual-Metric Reporting** + Amend `notes/benchmark_pinning.md` | Dropping `40` tasks cuts the benchmark by 35% (including read-only Task `24`). Reporting **both** strict deterministic `DB + COMMUNICATE` substring accuracy (including tasks `30, 31, 32`) **and** official `tau2` `DB × NL_ASSERTION` (`gpt-4.1-2025-04-14`, `temperature=0.0` with cached judge verdicts) preserves benchmark comparability while strictly satisfying Gate G3. |

---

## 3. Exact Code & Documentation Changes Required to Execute CBA-8

When executing CBA-8 in the next session, apply these exact file changes:

1. **Create `scripts/extract_retail_artifacts.py` (New — Run Once with `LITELLM_LOCAL_MODEL_COST_MAP=True`):**
   - Imports `tau2.registry.registry` and `tau2.agent.llm_agent` (`SYSTEM_PROMPT`, `AGENT_INSTRUCTION`).
   - Exports:
     - `configs/retail_tools_openai.json`: OpenAI JSON schemas + `ToolType` metadata for all **16** retail tools (`7` `WRITE`, `7` `READ`, `2` `GENERIC`).
     - `configs/retail_system_prompt.txt`: exact formatted system prompt (`AGENT_INSTRUCTION` + `<policy>` + `policy.md`, `1,492` Qwen3 tokens).
     - `configs/retail_policy.md`: local frozen copy of `data/tau2/domains/retail/policy.md` for offline runs.
2. **Fix & Upgrade `scripts/count_tokens.py`:**
   - **Fix `transformers 5.x` bug:** Change `count_chat_tokens()` to call `tokenizer.apply_chat_template(messages, tools=tools if tools else None, tokenize=False, add_generation_prompt=True)` followed by `len(tokenizer.encode(rendered, add_special_tokens=False))`. Assert return value `> 10` on any non-trivial prompt to guard against `BatchEncoding` length bugs.
   - Add `--system-prompt-file` support so `P` includes `AGENT_INSTRUCTION` + `<policy>...</policy>` (`1,492` tokens).
   - Default `--tools-file` to `configs/retail_tools_openai.json` (all 16 tools, `3,382` templated tokens).
   - Add `--oracle-replay` mode (consuming `--tasks-json` and `tau2` env or precomputed oracle trajectories) to compute per-task $H_{\text{oracle}}$, $S_{\text{min}}$, and feasibility percentages across **both** $B_{\text{total}}$ and $B_{\text{alloc}} \in \{2048, 4096, 8192\}$.
3. **Upgrade `scripts/audit_retail_tasks.py`:**
   - Implement `is_vacuous_pass(task, write_tools)` checking whether a task has zero expected `WRITE` tool calls **and** zero graded `nl_assertions` (`"NL_ASSERTION"` in `reward_basis` with non-empty `nl_assertions`). Verify it catches exactly the **6 vacuous tasks** (`10, 12, 25, 50, 57, 65`) while retaining Task `24`.
   - Load `split_tasks.json` and `oracle_profile.json` (if present) to annotate every task with: `persona_group` (normalized `known_info`), `official_split` (`train`/`test`), `has_write`, `has_nl_assertions`, `has_communicate_info`, `is_ci_only`, and `H_oracle`.
   - Replace naive task shuffle with a **seeded (`seed=42`) persona-grouped stratified split** targeting `eval_size ≈ 50` (achieving `50 ± 2` depending on intact persona cluster sizes) over the `108` non-vacuous tasks, balanced across `has_nl_assertions` and `H_oracle` quartiles.
   - Enforce built-in assertions: `eval ∩ train == ∅`, `eval_personas ∩ train_personas == ∅`, `vacuous_ids ∩ (eval ∪ train) == ∅`.
4. **Update `notes/benchmark_pinning.md`:**
   - Record exact commit SHA `fc0055dc4e0a316c3f83133267fbd6faaa770992` and `tasks.json` SHA-256 `f44563d8f36362e2dc58282bb02cfb85010d22f27d9439808758d3e9b2bc16d2`.
   - Remove the nonexistent `--communicate-scorer substring` flag; document that `0/114` retail tasks use `COMMUNICATE` in `reward_basis`, `40/114` use LLM-judged `nl_assertions` (`gpt-4.1-2025-04-14`, `temperature=0.0`), and `3` tasks (`30, 31, 32`) are `ci_only`.
   - Document Decisions **D1**, **D2**, and **D3**, the `6` vacuous-pass tasks, and the `16` retail tools.
5. **Update `setup/run_vllm.sh`:**
   - Change `--max-model-len 8192` to `--max-model-len "${4:-12288}"` (and update echo banner) so `P (1,492) + B_alloc (8,192) + R_out (512) = 10,196 <= 12,288` fits cleanly inside Decision D1.
6. **Create `tests/test_cba8_artifacts.py` (`pytest` verification):**
   - Test `count_chat_tokens()` returns `1492` for `P` and `3382` for all 16 tools (not `2` or `0`).
   - Test `configs/retail_task_split.json` has `108` total valid tasks, `6` excluded vacuous tasks (`10, 12, 25, 50, 57, 65`), `0` train/eval task overlap, `0` persona group overlap, and valid `tasks.json` SHA-256.

---

## 4. Step-by-Step Execution Runbook (Copy-Pasteable, ~60–90 min)

Run these steps sequentially when executing CBA-8 (commands shown for **Windows PowerShell** on your laptop; equivalent Linux/Codespaces syntax noted in comments).

### Step 1: Initialize Local Python 3.12 Environment & Clone `tau2-bench@v1.0.1`
```powershell
# From repo root: d:\Surya\Masters\Amirtha_MCA_AI\Major Project\context-budget-agent\context-budget-agent
git clone --depth 1 --branch v1.0.1 https://github.com/sierra-research/tau2-bench.git .cache\tau2-bench
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -e .\.cache\tau2-bench transformers tokenizers jinja2 pytest
```
*(Ensure `.cache/` is in `.gitignore` or store `tau2-bench` outside the repo in `/tmp/tau2-bench` on Linux/Codespaces.)*

### Step 2: Extract Static Retail Tools, System Prompt & Policy (Once)
```powershell
$env:LITELLM_LOCAL_MODEL_COST_MAP = "True"
$env:HF_HUB_OFFLINE = "0"  # Only needed on first tokenizer download
.venv\Scripts\python.exe scripts\extract_retail_artifacts.py `
  --out-tools configs\retail_tools_openai.json `
  --out-system-prompt configs\retail_system_prompt.txt `
  --out-policy configs\retail_policy.md
```
**Expected verification:** `configs/retail_tools_openai.json` contains **16** tools; `configs/retail_system_prompt.txt` contains `AGENT_INSTRUCTION` + `<policy>` + `policy.md`.

### Step 3: Run Upgraded Retail Task Audit & Freeze Persona-Grouped Split
```powershell
.venv\Scripts\python.exe scripts\audit_retail_tasks.py `
  --tasks-json .cache\tau2-bench\data\tau2\domains\retail\tasks.json `
  --split-json .cache\tau2-bench\data\tau2\domains\retail\split_tasks.json `
  --eval-size 50 `
  --seed 42 `
  --out configs\retail_task_split.json
```
**Expected verification:**
- `total_raw_tasks`: **`114`**
- `excluded_vacuous_pass`: **`6`** (`['10', '12', '25', '50', '57', '65']`)
- `valid_tasks`: **`108`** (Task `24` retained)
- `persona_overlap_count`: **`0`**

### Step 4: Run Fixed Qwen3 Token Accounting & Oracle Profile
```powershell
.venv\Scripts\python.exe scripts\count_tokens.py `
  --model Qwen/Qwen3-1.7B `
  --system-prompt-file configs\retail_system_prompt.txt `
  --tools-file configs\retail_tools_openai.json `
  --tasks-json .cache\tau2-bench\data\tau2\domains\retail\tasks.json `
  --out notes\retail_token_profile.json
```
**Expected verification:**
- `P_system_prompt_templated`: **`1,492`**
- `S_all_16_tools_templated`: **`3,382`**
- `P_plus_S_plus_framing_first_turn`: **`4,883`**
- `H_oracle` median (`p50`): **`1,777`**, mean: **`2,305`**
- `B_alloc (S + H)` fit rates: **`42/114` (`36.8%`)** at `2k`, **`82/114` (`71.9%`)** at `4k`, **`113/114` (`99.1%`)** at `8k`.

### Step 5: Amend Docs/Scripts & Run Automated `pytest` Suite
Update `notes/benchmark_pinning.md` and `setup/run_vllm.sh`, write `tests/test_cba8_artifacts.py`, and run:
```powershell
.venv\Scripts\pytest.exe tests\test_cba8_artifacts.py -v
```
**Expected verification:** `100%` of assertions pass.

### Step 6: Commit & Push CBA-8 Deliverables
```powershell
git add configs/retail_tools_openai.json configs/retail_system_prompt.txt configs/retail_policy.md configs/retail_task_split.json notes/retail_token_profile.json notes/benchmark_pinning.md scripts/extract_retail_artifacts.py scripts/count_tokens.py scripts/audit_retail_tasks.py setup/run_vllm.sh tests/test_cba8_artifacts.py
git commit -m "feat(cba-8): fix Qwen3 token accounting, audit vacuous tasks, and freeze leak-free retail split"
git push origin main
```

---

## 5. Complete Reference Tables (Verified Against `v1.0.1`)

### 5.1 All 16 Retail Tool Schemas & Exact Qwen3 Templated Token Costs
*(Measured as the marginal BPE token increase over `system_prompt + minimal_user` in `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`.)*

| # | Tool Name | `ToolType` | Templated Tokens | Role in Retail Workflow |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `list_all_product_types` | `READ` | `161` | Catalog product discovery |
| 2 | `get_user_details` | `READ` | `174` | Profile, default address, order list, payment methods |
| 3 | `find_user_id_by_email` | `READ` | `180` | Authentication via email |
| 4 | `calculate` | `GENERIC` | `181` | Deterministic arithmetic helper |
| 5 | `get_product_details` | `READ` | `184` | Product variants, options, prices, availability |
| 6 | `get_item_details` | `READ` | `184` | Single item details (often missed in 15-tool summaries) |
| 7 | `get_order_details` | `READ` | `186` | Order status, items, fulfillment tracking, payments |
| 8 | `transfer_to_human_agents` | `GENERIC` | `204` | Out-of-scope escalation (non-mutating DB) |
| 9 | `modify_pending_order_payment` | `WRITE` | `290` | Switch payment method on pending order |
| 10 | `find_user_id_by_name_zip` | `READ` | `299` | Authentication via full name + zip code |
| 11 | `cancel_pending_order` | `WRITE` | `334` | Cancel pending order with valid reason |
| 12 | `return_delivered_order_items` | `WRITE` | `388` | Return items from delivered order |
| 13 | `modify_user_address` | `WRITE` | `407` | Update user's default profile address |
| 14 | `modify_pending_order_address` | `WRITE` | `423` | Update shipping address on pending order |
| 15 | `modify_pending_order_items` | `WRITE` | `465` | Swap variant items on pending order |
| 16 | `exchange_delivered_order_items` | `WRITE` | `477` | Exchange variant items on delivered order |
| **Total** | **All 16 Tools (`S_all`)** | `7W / 7R / 2G` | **`3,382`** | *(Combined template serialization is smaller than the standalone sum `4,537` because shared `<tools>` wrapper overhead is only paid once)* |

### 5.2 Edge-Case Task Registry (Audit Disposition)

| Category | Task IDs (Official Split) | Count | Underlying Cause | Disposition in CBA-8 |
| :--- | :--- | :---: | :--- | :--- |
| **Vacuous-Pass Transfer Tasks** | `10` (train), `12` (test), `50` (train) | `3` | Expected action is `transfer_to_human_agents` (`ToolType.GENERIC`); DB is unchanged and `nl_assertions` is `None`. Empty agent trajectory scores `1.0`. | **EXCLUDE** (`vacuous_pass`) |
| **Vacuous-Pass Read/Refusal Tasks** | `25` (train), `57` (train), `65` (test) | `3` | Zero expected `WRITE` actions (Task `57` has `actions=[]`) and `nl_assertions` is `None`. Empty agent trajectory scores `1.0`. | **EXCLUDE** (`vacuous_pass`) |
| **Empty-Action Non-Vacuous Task** | `24` (train) | `1` | `actions = []` (user declines cancellation), **but** has `communicate_info = ['polyester', 'cotton']` and graded `nl_assertion`. | **KEEP** (Valid non-vacuous test of communication) |
| **`communicate_info`-Only Tasks (`ci_only`)** | `30` (train), `31` (train), `32` (test) | `3` | Have `communicate_info = ['746342064230']` and `nl_assertions = None`. Default `EvaluationType.ALL` silently skips checking `'746342064230'`. | **KEEP** (Graded via our `DB + COMMUNICATE` metric in Decision D3) |
| **Pure `DB`-Only `reward_basis` Tasks** | `33` (test), `34` (train) | `2` | `reward_basis = ['DB']` (no `NL_ASSERTION`); same Noah Patel persona crossing official split! | **KEEP** (Assigned to the same split per persona grouping D2) |
| **Transfer Task with Graded Assertion** | `26` (test) | `1` | Contains `transfer_to_human_agents` **and** non-empty `nl_assertions`, so it is NOT vacuous. | **KEEP** |

- **Note on Expected-Action Replay Errors (`15` tasks):** When replaying `evaluation_criteria.actions` inside `RetailTools`, **`15` tasks** (`2, 3, 4, 35, 37, 38, 39, 46, 47, 54, 55, 64, 67, 68, 105`) emit tool error strings (e.g., Task `2`/`3` looking up invalid `product_id: '6086499569'`, Task `35` querying mistyped email `aarav.santos8321@example.com` before `8320`, Task `46` querying `#9502126`/`#9502127` without `W` before `#W9502127`). Inspection confirms these are **benign exploratory lookup misses** embedded in the gold traces and do **not** invalidate DB end-state grading.

---

## 6. CBA-8 Definition of Done & Handoff to CBA-9 / CBA-10

### CBA-8 Verification Checklist (DoD)
- [ ] `scripts/extract_retail_artifacts.py` exports all **16** real retail tools (`configs/retail_tools_openai.json`) and system prompt (`configs/retail_system_prompt.txt`).
- [ ] `scripts/count_tokens.py` uses `tokenize=False` + `tokenizer.encode(..., add_special_tokens=False)` (immune to `transformers 5.x` `BatchEncoding`), reporting `P = 1,492` and `S_all = 3,382`.
- [ ] `notes/retail_token_profile.json` records the full `Qwen3-1.7B` / `Qwen3-4B` token breakdown and `2k / 4k / 8k` fit rates under both $B_{\text{total}}$ and $B_{\text{alloc}}$.
- [ ] `configs/retail_task_split.json` locks `108` valid tasks, excludes the `6` vacuous-pass tasks (`10, 12, 25, 50, 57, 65`), enforces `0` persona-group overlap between `train` and `eval`, and records `tasks.json` SHA-256.
- [ ] `notes/benchmark_pinning.md` and `setup/run_vllm.sh` (`--max-model-len 12288`) updated to match Decisions D1–D3.
- [ ] `pytest tests/test_cba8_artifacts.py` passes 100% and changes are committed/pushed to `main`.

### Concrete Handoff Notes for Downstream Tickets
- **For `[CBA-9]` (`src/budget_controller.py`):**
  - Use **Decision D1 ($B_{\text{alloc}} \in \{2048, 4096, 8192\}$)** for $S + H$ with `P = 1,492` pinned separately.
  - Design schema packing around **16** retail tools (`3,382` tokens uncompressed; min single tool `161`, max `477`, oracle median `S_min = 821`) plus $N$ distractor schemas, and history truncation around `H_oracle` (`p50 = 1,777`, `p90 = 5,036`).
  - Keep `find_user_id_by_email` (`180`) and `find_user_id_by_name_zip` (`299`) or query-matched tools prioritized so authentication turns do not fail at `2k` (`S_budget` as low as `512`–`1,024` tokens).
- **For `[CBA-10]` (`src/compliance_checkers.py`):**
  - Inspection of `data/tau2/domains/retail/policy.md` (lines `10`–`136`) confirms additional concrete deterministic rules beyond `R1`–`R6` to encode cleanly:
    - **Single-Tool-Per-Turn Rule (line `20`):** Assistant turn must not emit `> 1` parallel tool call, and must not mix a non-empty user message with `tool_calls` in the same turn.
    - **Authentication Even With Provided ID (line `10`):** `find_user_id_by_email` or `find_user_id_by_name_zip` is mandatory even if the user states their `user_id` directly.
    - **Cancel Reason Enum (line `90`):** `cancel_pending_order` reason argument must strictly be `'no longer needed'` or `'ordered by mistake'`.
    - **Refund Destination Constraint (line `124`):** `return_delivered_order_items` `payment_method_id` must match either the order's original payment method or an existing gift card in the user's profile.
    - **Transfer Sequence Constraint (line `24`):** After calling `transfer_to_human_agents`, the assistant must send `'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'`.
