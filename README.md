# Context Budget Agent

Empirical investigation into how small tool-using language model agents perform when a fixed input token budget is allocated across **tool schemas** versus **interaction history** (with policy context held constant), and whether supervised fine-tuning (SFT) recovers lost task accuracy.

---

## 📂 Repository Layout

```text
context-budget-agent/
├── configs/       # Model serving configurations and benchmark parameters
├── notes/         # Literature reviews, benchmark decision logs, and environment specs
├── runs/          # Experiment outputs, trajectories, evaluation scores (git-ignored)
├── scripts/       # Token counting, API verification, and operational utilities
├── setup/         # Environment installation scripts and container definitions
├── .gitignore     # Excludes virtualenvs, ML weights, runs, and local artifacts
├── LICENSE        # MIT License
└── README.md      # Repository architecture and execution guide
```

---

## 🛠️ Prerequisites & Stack

- **Operating System:** Linux (Ubuntu 22.04 LTS recommended for cloud GPU VM)
- **Python:** 3.12+ managed via [`uv`](https://github.com/astral-sh/uv)
- **GPU Serving:** NVIDIA GPU (≥ 24 GB VRAM) with Docker and NVIDIA Container Toolkit
- **Serving Engine:** [vLLM](https://github.com/vllm-project/vllm) (`vllm/vllm-openai`) with Hermes tool-calling parser
- **Foundation Model:** `Qwen/Qwen3-1.7B`
- **Benchmark:** [τ²-bench (Tau2-Bench)](https://github.com/sierra-research/tau2-bench) retail domain

---

## 🚀 Week 1 Smoke Test Workflow

The goal of Week 1 is to verify end-to-end plumbing: executing one retail task through `Qwen/Qwen3-1.7B` served via vLLM, recording the trajectory and token count.

### 1. Launch vLLM Serving Container
```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  vllm/vllm-openai:<PINNED_TAG> \
  --model Qwen/Qwen3-1.7B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192
```

### 2. Verify Tool-Calling Endpoint
Verify that tool calling produces parsed JSON arguments:
```bash
python scripts/test_tool_call.py
```

### 3. Install τ²-bench & Run Retail Task
```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench && git checkout <PINNED_REVISION>
uv sync
uv run tau2 run --domain retail --task-id <TASK_ID> --agent-endpoint http://localhost:8000/v1
```

### 4. Authoritative Token Counting
Run Qwen's chat template tokenizer to record the prompt breakdown:
```bash
python scripts/count_tokens.py --trajectory-file runs/smoke_test_run_01/trajectory.json
```

---

## 📝 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
