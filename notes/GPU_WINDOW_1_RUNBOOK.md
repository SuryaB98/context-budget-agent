# GPU Window 1 Execution Runbook (`CBA-3` → `CBA-4` → `CBA-5`)

**Environment:** RunPod Community Cloud + 40 GB Persistent Network Volume (`/workspace`)  
**Estimated GPU Time:** ~4–6 GPU hours (1× A100 80 GB for 4-model floor screen, or 1× RTX 4090 / 3090 24 GB for `1.7B` + `4B` pilot)  
**Pre-flight Gate:** All `40/40` offline unit tests (`pytest tests/ -v`) passed in commit `9b33279`.

---

## 0. Pre-Flight Checklist (Before Starting the GPU Clock)

1. **OpenAI API Key:** Confirm `OPENAI_API_KEY` has a **$10 hard spend limit** (used by `tau2`'s `gpt-4o-mini` user simulator and the disk-cached `gpt-4.1-2025-04-14` NL judge).
2. **RunPod Prepaid Cap:** Load only your prepaid budget (~`$25`) so spend is strictly capped.
3. **Create a 40 GB Persistent Network Volume (once, reused across GPU Windows 1–3):**
   - RunPod Console → **Storage → + New Network Volume**
   - Name: `cba-workspace` · Size: **40 GB** (~`$2.80/mo`) · Mount path: `/workspace`
   - Choose a datacenter with **RTX 4090 / 3090 24 GB** and **A100 80 GB** availability.
4. **Launch 1 Pod attached to `cba-workspace` (`/workspace`):**
   - **GPU choice:**
     - **1× A100 80 GB** if running the 45-min 4-model floor screen (`Qwen3-1.7B / 4B / 8B / 14B`) + the `1.7B/4B` 50-task pilot in one batch.
     - **1× RTX 4090 (or 3090) 24 GB** if running `Qwen3-1.7B` and `Qwen3-4B` only.
   - **Container Image:** `vllm/vllm-openai:v0.8.5` (override Start Command to `sleep infinity` for interactive Web Terminal / SSH), or a RunPod PyTorch CUDA 12.4 template (`pip install vllm==0.8.5`).
   - Open Web Terminal / SSH and start `tmux new -s cba` so session drops do not interrupt runs.

---

## 1. `[CBA-3]` First-Boot Persistent Setup & Hardware Validation (~5 min)

```bash
# 1. Verify GPU VRAM, driver, and CUDA version (record output in notes/vm_environment.md)
nvidia-smi

# 2. Persist HF cache, Antigravity memory, and raw runs on /workspace
mkdir -p /workspace/.gemini /workspace/hf_cache /workspace/runs
ln -sfn /workspace/.gemini ~/.gemini
export HF_HOME=/workspace/hf_cache
export LITELLM_LOCAL_MODEL_COST_MAP=True
export OPENAI_API_KEY="sk-YOUR-KEY-HERE"

# 3. Clone project repo and pinned tau2-bench@v1.0.1 onto /workspace
cd /workspace
git clone https://github.com/SuryaB98/context-budget-agent.git
git clone --branch v1.0.1 --depth 1 https://github.com/sierra-research/tau2-bench.git

# 4. Install dependencies into the pod's Python 3.12 environment
cd /workspace/context-budget-agent
pip install -e /workspace/tau2-bench transformers tokenizers rank-bm25 jinja2 openai pytest scipy numpy pandas matplotlib

# 5. Verify all 40 offline unit tests pass on the pod
pytest tests/ -v
```

**CBA-3 DoD:** Record GPU model, VRAM (`>= 24 GB`), driver/CUDA version, region, and `$/hr` rate in `notes/vm_environment.md`.

---

## 2. `[CBA-4]` Launch vLLM Natively & Verify Hermes Tool Calling (~3 min)

> **Operational Rule:** RunPod pods already run inside Docker. Always launch `setup/run_vllm_runpod.sh` (`vllm serve` directly) — never nested `docker run` (`setup/run_vllm.sh`).

**In `tmux` window 1:**
```bash
cd /workspace/context-budget-agent
bash setup/run_vllm_runpod.sh Qwen/Qwen3-1.7B 8000 12288 on
```
Wait for `Uvicorn running on http://0.0.0.0:8000`.

**In `tmux` window 2 (`Ctrl+B` then `C`):**
```bash
cd /workspace/context-budget-agent
export HF_HOME=/workspace/hf_cache
export LITELLM_LOCAL_MODEL_COST_MAP=True
export OPENAI_API_KEY="sk-YOUR-KEY-HERE"

python scripts/test_tool_call.py \
  --model Qwen/Qwen3-1.7B \
  --out notes/vllm_test_response.json
```

**CBA-4 DoD:** Exit code `0`, structured `tool_calls` JSON parsed (`get_order_details`), and `notes/vllm_test_response.json` saved.

---

## 3. `[CBA-5]` Part A — 1-Task End-to-End Smoke Test (~1 min)

Run task `0` end-to-end before launching 50 tasks:

```bash
python -m src.eval_runner \
  --run-id smoke_test_run_01 \
  --out-root /workspace/runs \
  --model Qwen/Qwen3-1.7B \
  --task-ids 0 \
  --grid baseline_pilot \
  --nl-judge-cache /workspace/runs/nl_judge_cache.json
```

Verify `/workspace/runs/smoke_test_run_01/summary.json`:
- `all_turns_within_budget_rate == 1.0`
- `mean_ttft_ms > 0`
- `mean_prefix_cache_hit_rate_delta` is non-null
- `episodes.jsonl` and `results.csv` contain full trajectory, dual rewards, and `R1`–`R8` compliance fields.

---

## 4. `[CBA-5]` Part B — 4-Model Floor Screen & 50-Task Baseline Pilot (`>= 15%` Gate G5)

### 4a. Optional 45-Min 4-Model Floor Screen (A100 80 GB)
Before running all 50 tasks, optionally run a fast 10-task deterministic screen (`--max-tasks 10 --skip-nl-judge`) across `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, and `Qwen/Qwen3-14B` to check the `>= 15%` floor gate across model scales:

```bash
python -m src.eval_runner \
  --run-id floor_screen_10_<MODEL_TAG> \
  --out-root /workspace/runs \
  --model <MODEL_ID> \
  --split eval \
  --max-tasks 10 \
  --grid baseline_pilot \
  --skip-nl-judge
```

### 4b. Full 50-Task Baseline Pilot on `Qwen/Qwen3-1.7B`
```bash
python -m src.eval_runner \
  --run-id pilot_50_qwen3_1.7b \
  --out-root /workspace/runs \
  --model Qwen/Qwen3-1.7B \
  --split eval \
  --grid baseline_pilot \
  --nl-judge-cache /workspace/runs/nl_judge_cache.json
```

### 4c. Switch vLLM to `Qwen/Qwen3-4B` & Run 50-Task Baseline Pilot
In `tmux` window 1 (`Ctrl+B` then `0`), press `Ctrl+C` to stop `1.7B`, then start `4B`:
```bash
bash setup/run_vllm_runpod.sh Qwen/Qwen3-4B 8000 12288 on
```
In `tmux` window 2 (`Ctrl+B` then `1`):
```bash
python scripts/test_tool_call.py \
  --model Qwen/Qwen3-4B \
  --out notes/vllm_test_response_4b.json

python -m src.eval_runner \
  --run-id pilot_50_qwen3_4b \
  --out-root /workspace/runs \
  --model Qwen/Qwen3-4B \
  --split eval \
  --grid baseline_pilot \
  --nl-judge-cache /workspace/runs/nl_judge_cache.json
```

> **Crash Recovery:** `--resume` is enabled by default. Re-running the exact command skips completed `(condition_id, task_id, seed)` episodes found in `episodes.jsonl`.  
> **Qwen3 Thinking Toggle:** If `1.7B` exhausts `output_reserve = 512` inside `<think>...</think>` blocks and drops below `15%`, re-test 10 tasks with `--enable-thinking false` (`--run-id pilot_10_1.7b_nothink`) and compare pass rates.

---

## 5. Save Summaries, Push to `origin/main` & Terminate Pod Immediately

```bash
cd /workspace/context-budget-agent

# Review both summary.json files (verify passes_15pct_floor_gate: true)
cat /workspace/runs/pilot_50_qwen3_1.7b/summary.json
cat /workspace/runs/pilot_50_qwen3_4b/summary.json

# Record results in notes/pilot_floor_results.md and update GEMINI.md Section 5,
# then commit ONLY compact notes/ (never commit /workspace/runs/)
git add notes/vm_environment.md notes/vllm_test_response.json notes/pilot_floor_results.md GEMINI.md
git commit -m "chore(gpu-w1): record CBA-3/4/5 hardware specs, vLLM check, and 50-task floor pilot"
git push origin main
```

**Immediately After Push:**
1. Download `/workspace/runs/pilot_50_qwen3_1.7b/episodes.jsonl` and `/workspace/runs/pilot_50_qwen3_4b/episodes.jsonl` to your local machine (or Codespace) for **`CBA-12`** (`$0` CPU hand-labeling of 30–50 transcripts to calibrate `src/compliance_checkers.py` to `>= 90%`).
2. **TERMINATE the RunPod GPU pod immediately** in the RunPod Console. Keep the **40 GB `cba-workspace` Network Volume** intact so cached Hugging Face models, `nl_judge_cache.json`, and raw `/workspace/runs/` logs are preserved for **GPU Window 2**.
