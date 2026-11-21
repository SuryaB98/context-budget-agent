# Literature Review & Research Gap Qualification ([CBA-7])

**Project:** Token Exchange Rates: How Small Model Agents Should Spend a Fixed Context Budget  
**Course:** Amrita Online MCA (Minor Project 21CSA391A)  
**Author:** Surya B (`AA.SC.P2MCA25014003`)  
**Updated:** 2026-11-22  

---

## 1. Comparative Synthesis Matrix of Core Prior Art

| Paper & Venue / ID | Primary Problem Studied | Models & Budgets Tested | Context Components Varied | Primary Metrics | What It Leaves Open (Our Gap) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Tool-Schema Compression**<br>([arXiv:2605.26165](https://arxiv.org/abs/2605.26165)) | Compressing verbose MCP/API tool schemas to make room for retrieved RAG evidence. | 14 local models (1.5B–32B); 8k / 16k / 32k context windows; 28 tools. | Tool Schemas ($S$) vs. Retrieved RAG Evidence ($E$). | Single-turn / RAG tool selection accuracy (+20.5 pp at 8k after schema compression; no gain at 32k). | Does **not** study multi-turn dialogue history ($H$) vs. tool schemas ($S$) under pinned operational policy ($P$), nor silent policy violations or SFT budget recovery. |
| **2. Control Under Compression**<br>([arXiv:2608.01056](https://arxiv.org/html/2608.01056)) | Impact of compressing agent control contexts (system prompts, tool specs, protocols) on reliability. | 15,500+ execution runs across open and frontier models. | Control Context ($S + P$) compression ratio (100% down to <30%). | Agent task reliability and control adherence (diverges sharply below ~50% retention). | Treats control context ($S + P$) as a single block rather than trading off $S$ against cumulative multi-turn history $H$ with $P$ held constant. |
| **3. Governance Decay**<br>([arXiv:2606.22528](https://arxiv.org/html/2606.22528v2)) | How context length growth and compression degrade adherence to operational governance and safety policies. | Multi-turn agent benchmarks under growing context load. | Policy/Governance Instructions ($P$) under history growth ($H$). | Policy violation rate across conversation turns. | Motivates our **Silent Policy Violation** metric (H1b) and secondary policy-retrieval experiment (RQ1b), but does not quantify the $S$-vs-$H$ exchange rate at 2k/4k/8k budgets. |
| **4. PA-Tool (ACL 2026)**<br>([arXiv:2510.07248](https://arxiv.org/pdf/2510.07248)) | Adapting and rewriting tool schemas to fit model preferences vs. adapting the model itself. | Open-weight tool-calling models. | Schema representation ($S$) vs. model weights. | Function-calling accuracy across adapted schemas. | Argues for schema engineering; we directly compare **context engineering ($S/H$ allocation)** against **parameter-efficient post-training (LoRA SFT)** as competing remedies for budget starvation. |
| **5. ToolRL**<br>([arXiv:2504.13958](https://arxiv.org/abs/2504.13958)) | Reward design and RL (GRPO) vs. SFT for tool-using language models. | Small/medium open models (1.5B–7B). | Unconstrained standard prompts (no budget dimension). | Tool selection and argument accuracy. | Does not evaluate training under tight token budgets (2k/4k/8k) or measure whether post-training recovers performance lost to context pruning. |
| **6. Debunk the Myth of SFT Generalization**<br>([arXiv:2510.00237](https://arxiv.org/html/2510.00237v1)) | Whether SFT generalizes to unseen tasks/tools when given sufficient prompt diversity and compute. | Open-weight LLMs across reasoning and tool benchmarks. | Training data diversity & SFT configurations. | In-distribution vs. out-of-distribution generalization. | Motivates our Milestone 6 & 7 SFT ablations (dataset size $\times$ LoRA rank) and evaluation on **held-out budgets** and **unseen tool catalogues**. |

---

## 2. Deep Notes on Foundational Papers (Ready for Interim Report Chapter 2)

### 2.1 Tool-Schema Compression (`arXiv:2605.26165`)
- **Core Premise:** Modern Model Context Protocol (MCP) servers expose dozens of tools whose JSON schemas easily consume 4,000–6,000 tokens before a single user turn occurs.
- **Empirical Finding:** Across 14 local models (1.5B to 32B) and 28 tools, raw schemas crowd out retrieved RAG passages at an 8k budget. Compressing tool schemas yields a **+20.5 percentage point** jump in accuracy at 8k, whereas at 32k the difference disappears (a sharp threshold effect).
- **Distinction from Our Work:** `2605.26165` studies **Schemas ($S$) vs. Retrieved Evidence ($E$)** in RAG-heavy settings. In interactive transactional agents (such as $\tau^3$-bench Retail), facts are not pre-retrieved from a static corpus; they arrive dynamically inside **Interaction History ($H$)** as the output of sequential tool calls (`get_order_details`, `find_user_id_by_email`) and user turns, governed by a strict **Policy ($P$)**.

### 2.2 Control Under Compression (`arXiv:2608.01056`) & Governance Decay (`arXiv:2606.22528`)
- **Core Premise:** When developers compress prompts to save latency or fit small windows, they often summarize or prune control instructions and policy constraints.
- **Empirical Finding:** `2608.01056` shows across 15.5k runs that agent reliability degrades non-linearly once control context drops below ~50% of its uncompressed length. `2606.22528` further demonstrates *Governance Decay*: agents continue completing user requests while silently dropping mandatory verification or confirmation steps.
- **Distinction from Our Work:** Standard benchmarks like $\tau^3$-bench grade only the final database state ($\text{DB\_Reward} \times \text{Communicate\_Reward}$). If an agent cancels an order without asking the customer for explicit confirmation first—or modifies an account before authenticating the user's email/zip—the final DB state still matches, producing a **Silent Policy Violation**. Our work builds deterministic trajectory rule checkers to separate *task completion* from *silent policy violations* across controlled $S:H$ budget ratios.

### 2.3 Post-Training vs. Context Engineering (`PA-Tool`, `ToolRL`, `2510.00237`)
- **Core Premise:** When a small model (1.7B–4B) fails under a tight context budget (e.g., 2k or 4k tokens), practitioners can either engineer the prompt allocation ($S:H$ ratio, schema pruning, history trimming) or fine-tune the model via LoRA on budget-constrained trajectories.
- **Distinction from Our Work:** Existing tool-learning papers (`ToolRL`, `PA-Tool`) train and evaluate at generous, fixed context lengths. None quantify the **Token Exchange Rate**—how many kilobytes/tokens of inference context budget a LoRA adapter recovers (e.g., whether a LoRA-tuned `Qwen3-1.7B` or `4B` at a **2k/4k** budget matches the untrained base model at **4k/8k**), nor how prompt restructuring affects vLLM **PagedAttention prefix-cache hit rates** and **TTFT**.

---

## 3. Formal Research Gap Statement (Copy-Ready for Interim & Final Reports)

> **Research Gap Formulation:**  
> While recent studies have independently examined tool-schema compression against retrieved RAG passages (*arXiv:2605.26165*) and the degradation of agent reliability under control-context compression (*arXiv:2608.01056*; *arXiv:2606.22528*), three critical questions remain unaddressed for small open-weight language agents ($\le 4\text{B}$ parameters):  
> 1. **The Schema-vs-History Exchange Rate in Multi-Turn Policy-Governed Environments:** In interactive tool-use domains where operational rules ($P$) must remain fixed, no prior benchmark study systematically maps the performance trade-off between API tool-schema capacity ($S$) and multi-turn interaction history ($H$) across strict **2k, 4k, and 8k** BPE token budgets.  
> 2. **Silent Policy Violations vs. End-State Task Success:** Existing agent leaderboards evaluate terminal database mutations while overlooking intermediate procedural breaches (such as unauthenticated account lookups or unconfirmed state changes) that escalate as context budgets tighten.  
> 3. **Post-Training Budget Recovery and Serving Cost Reality:** It remains unknown how much context budget parameter-efficient fine-tuning (LoRA SFT across varying dataset sizes and ranks) can recover on held-out tasks and unseen tool catalogues, or when token savings from history rewriting are negated in practice by reduced **vLLM PagedAttention prefix-cache reuse** and elevated **Time-to-First-Token (TTFT)**.
