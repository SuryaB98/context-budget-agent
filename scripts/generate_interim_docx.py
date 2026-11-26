#!/usr/bin/env python3
"""
Generate a formatted university-ready Interim Report DOCX (`notes/Interim_Report_21CSA699A_Surya_B.docx`)
and a 300-DPI System Architecture figure (`notes/figures/system_architecture.png`) matching
`interim report temp.md` and the verified CBA-7/8/9/10/eval_runner artifacts.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_PATH = REPO_ROOT / "notes" / "figures" / "system_architecture.png"
DOCX_PATH = REPO_ROOT / "notes" / "Interim_Report_21CSA699A_Surya_B.docx"


def build_architecture_figure(out_path: Path) -> None:
    """Render a clean 300-DPI system architecture diagram for Section 4.2."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 56)
    ax.axis("off")

    def draw_box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str,
        fc: str,
        ec: str,
        title_color: str = "#0F172A",
    ) -> None:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.4",
            linewidth=1.5,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=title_color,
            family="DejaVu Sans",
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.3,
            color="#334155",
            family="DejaVu Sans",
            linespacing=1.25,
        )

    def draw_arrow(
        p1: tuple[float, float],
        p2: tuple[float, float],
        *,
        dashed: bool = False,
        rad: float = 0.0,
        label: str | None = None,
        label_pos: tuple[float, float] | None = None,
    ) -> None:
        arr = FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>,head_length=6,head_width=3.5",
            linewidth=1.4,
            linestyle="--" if dashed else "-",
            color="#475569",
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arr)
        if label and label_pos:
            ax.text(
                label_pos[0],
                label_pos[1],
                label,
                ha="center",
                va="center",
                fontsize=7.2,
                fontstyle="italic",
                color="#334155",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            )

    # Title
    ax.text(
        50,
        53.6,
        "Context Budget Agent — End-to-End Architecture & Experimental Pipeline",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#0F172A",
    )

    # Row 1: Inputs -> Controller -> vLLM -> Eval Runner
    draw_box(
        2, 34, 20, 14,
        "tau2-bench@v1.0.1 Retail",
        "108 Valid Tasks (50 Eval / 58 Train)\n16 Native Retail Tools (3,382 tok)\nPinned Policy P (1,492 tok)",
        fc="#EFF6FF", ec="#2563EB",
    )
    draw_box(
        2, 18.5, 19.5, 12,
        "Frozen Distractor Pools",
        "23 In-Dist (Airline + Telecom)\n13 Unseen (Banking Knowledge)\nZero Eval-Split Leakage",
        fc="#F0FDF4", ec="#16A34A",
    )
    draw_box(
        27, 25, 22, 18,
        "ContextBudgetController",
        "Exact Qwen3 BPE Accounting\nB_alloc in {2k, 4k, 8k}, R_out=512\nBM25 Schema & Policy Selection\nAtomic History Trimming / H4 Sum",
        fc="#FFF7ED", ec="#EA580C",
    )
    draw_box(
        54, 25, 21, 18,
        "vLLM v0.8.5 Server",
        "Qwen3-1.7B / Qwen3-4B (bf16)\nHermes Tool-Call Parser\nPagedAttention Prefix Cache\n--max-model-len 12288 | LoRA",
        fc="#FAF5FF", ec="#7C3AED",
    )
    draw_box(
        79, 25, 19, 18,
        "Streaming eval_runner",
        "Client TTFT, ITL & Throughput\nPrometheus /metrics Delta\nCrash-Safe JSONL + --resume\nresults.csv & summary.json",
        fc="#ECFEFF", ec="#0891B2",
    )

    # Row 2: Dual Grader, Compliance Engine, LoRA SFT
    draw_box(
        51, 4, 21, 13,
        "Decision D3 Dual Grader",
        "1. Strict DB + COMMUNICATE\n2. Cached DB x NL_ASSERTION\n(gpt-4.1-2025-04-14, temp=0)",
        fc="#FEFCE8", ec="#CA8A04",
    )
    draw_box(
        76, 4, 22, 13,
        "RetailComplianceEngine",
        "8 Deterministic Rules (R1-R8)\nTier A Silent Policy Violations\nTier B Attempted / Hallucination",
        fc="#FEF2F2", ec="#DC2626",
    )
    draw_box(
        24, 4, 22.5, 13,
        "Verifier-Filtered LoRA SFT",
        "58 Train Tasks Only (0 Overlap)\nFilter: Success=1 & Compliant=1\n2x2 Grid (50/100% x r=8/32)",
        fc="#EEF2FF", ec="#4F46E5",
    )

    # Arrows
    draw_arrow((22.6, 40), (26.4, 37))
    draw_arrow((22.1, 24.5), (26.4, 30))
    draw_arrow((49.6, 34), (53.4, 34))
    draw_arrow((75.6, 34), (78.4, 34))

    # Down arrows from eval_runner to Dual Grader & Compliance Engine
    draw_arrow((88.5, 24.4), (87.0, 17.6))
    draw_arrow((83.0, 24.4), (65.0, 17.6))

    # Grader + Compliance -> LoRA SFT (Compliance routes cleanly underneath Dual Grader)
    draw_arrow((50.4, 10.5), (47.1, 10.5))
    draw_arrow((87.0, 3.4), (35.2, 3.4), rad=-0.16)

    # Feedback arrow LoRA SFT -> vLLM Server (Milestone 7)
    draw_arrow(
        (35.2, 17.6),
        (63.5, 24.4),
        dashed=True,
        rad=-0.12,
        label="Milestone 7 Eval (--enable-lora)",
        label_pos=(49.0, 21.8),
    )

    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _set_cell_shading(cell, fill_hex: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def _set_table_borders(table, color_hex: str = "CBD5E1") -> None:
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
        f'<w:left w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
        f'<w:insideV w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
        f"</w:tblBorders>"
    )
    tblPr.append(borders)


def _set_cell_margins(cell, top: int = 80, bottom: int = 80, left: int = 110, right: int = 110) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def _add_page_number_footer(section) -> None:
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run1 = p.add_run("Major Project 21CSA699A — Interim Report  |  Surya B (AA.SC.P2MCA25014003)  |  Page ")
    run1.font.name = "Times New Roman"
    run1.font.size = Pt(9)
    run1.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    fldSimple = OxmlElement("w:fldSimple")
    fldSimple.set(qn("w:instr"), "PAGE")
    p._p.append(fldSimple)


def _add_rich_runs(paragraph, text: str, font_size: float = 11.0, base_color: RGBColor | None = None) -> None:
    """Render inline **bold**, *italic*, and `code` markdown spans cleanly into Word runs."""
    token_re = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    pos = 0
    for m in token_re.finditer(text):
        if m.start() > pos:
            r = paragraph.add_run(text[pos : m.start()])
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size)
            if base_color:
                r.font.color.rgb = base_color
        tok = m.group(0)
        if tok.startswith("**") and tok.endswith("**"):
            r = paragraph.add_run(tok[2:-2])
            r.bold = True
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size)
        elif tok.startswith("*") and tok.endswith("*"):
            r = paragraph.add_run(tok[1:-1])
            r.italic = True
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size)
        elif tok.startswith("`") and tok.endswith("`"):
            r = paragraph.add_run(tok[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(max(8.5, font_size - 1.0))
        if base_color:
            r.font.color.rgb = base_color
        pos = m.end()
    if pos < len(text):
        r = paragraph.add_run(text[pos:])
        r.font.name = "Times New Roman"
        r.font.size = Pt(font_size)
        if base_color:
            r.font.color.rgb = base_color


def _add_body(doc: Document, text: str, space_after: float = 6.0, justify: bool = True) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if justify else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    _add_rich_runs(p, text, font_size=11.0)


def _add_bullet(doc: Document, text: str, space_after: float = 4.0) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    _add_rich_runs(p, text, font_size=11.0)


def _add_equation(doc: Document, eq_text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(eq_text)
    r.font.name = "Cambria Math"
    r.font.size = Pt(11)
    r.italic = True


def _add_h1(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)


def _add_h2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)


def _add_table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float] | None = None,
    font_size: float = 9.0,
) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    _set_table_borders(table)

    # Header row
    hdr_row = table.rows[0]
    trPr = hdr_row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))
    for j, htxt in enumerate(headers):
        cell = hdr_row.cells[j]
        if col_widths:
            cell.width = Inches(col_widths[j])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _set_cell_shading(cell, "1E3A8A")
        _set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05
        r = p.add_run(htxt)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(font_size)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data rows
    for i, row_data in enumerate(rows):
        row = table.rows[i + 1]
        bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            if col_widths:
                cell.width = Inches(col_widths[j])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
            _set_cell_shading(cell, bg)
            _set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            _add_rich_runs(p, val, font_size=font_size)

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(4)


def build_interim_docx(fig_path: Path, out_path: Path) -> None:
    doc = Document()

    for sec in doc.sections:
        sec.top_margin = Inches(0.95)
        sec.bottom_margin = Inches(0.95)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)
        _add_page_number_footer(sec)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)

    # ---------------- Header / Cover Block (Matching interim report temp.md) ----------------
    p_course = doc.add_paragraph()
    p_course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_course.paragraph_format.space_after = Pt(2)
    r = p_course.add_run("Major Project 21CSA699A")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r = p_sub.add_run("INTERIM REPORT")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(meta_table, color_hex="94A3B8")
    meta_rows = [
        ("Project Title", "Token Exchange Rates: How Small Model Agents Should Spend a Fixed Context Budget"),
        ("Student Name", "Surya B"),
        ("Roll No.", "AA.SC.P2MCA25014003"),
        ("Program", "Master of Computer Applications (Amrita Online MCA)"),
        ("Repository", "https://github.com/SuryaB98/context-budget-agent"),
    ]
    for idx, (k, v) in enumerate(meta_rows):
        c0, c1 = meta_table.rows[idx].cells
        c0.width, c1.width = Inches(1.6), Inches(4.9)
        _set_cell_shading(c0, "EFF6FF")
        _set_cell_shading(c1, "FFFFFF")
        _set_cell_margins(c0, 70, 70, 110, 110)
        _set_cell_margins(c1, 70, 70, 110, 110)
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.name = "Times New Roman"
        r0.font.size = Pt(10.5)
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(v)
        r1.bold = idx == 0
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(10.5)

    # ---------------- Abstract ----------------
    _add_h1(doc, "Abstract")
    _add_body(
        doc,
        "Deploying small open-weight language models (≤ 4B parameters, such as `Qwen3-1.7B` and `Qwen3-4B`) "
        "as autonomous multi-turn tool-calling agents requires operating within tight context-window budgets "
        "where three components compete for input tokens: API tool schemas (*S*), operational policy rules (*P*), "
        "and cumulative interaction history (*H*). When total prompt length exceeds the budget, preserving schemas "
        "at the expense of history erases earlier user constraints and tool outputs, whereas preserving history "
        "by pruning schemas induces tool-selection errors and argument hallucinations. Existing benchmarks score "
        "only terminal database state and therefore overlook *silent policy violations*—episodes that achieve the "
        "target end-state while bypassing mandatory procedures such as user authentication or pre-mutation confirmation.",
    )
    _add_body(
        doc,
        "This project investigates the empirical *token exchange rate* across allocatable (*S* + *H*) budgets of "
        "`2k`, `4k`, and `8k` BPE tokens on `tau2-bench@v1.0.1` Retail. We design and implement a deterministic "
        "per-component budget controller, an 8-rule offline policy compliance engine, a streaming vLLM evaluation "
        "and PagedAttention prefix-cache profiler, and a leak-free verifier-filtered LoRA supervised fine-tuning (SFT) "
        "pipeline. Pre-flight auditing uncovered 6 vacuous-pass tasks and 20/31 leaky customer-persona groups in the "
        "upstream split, yielding a frozen 50-eval / 58-train split with zero persona overlap and an offline-verified "
        "harness (40/40 passing unit tests) ready for cloud GPU execution.",
    )

    # ---------------- 1. Objectives ----------------
    _add_h1(doc, "1. Objectives")
    _add_bullet(
        doc,
        "**Quantify the Schema-vs.-History Allocation Trade-off (RQ1):** Measure how varying the tool-schema (*S*) "
        "to interaction-history (*H*) token allocation ratio (75:25, 50:50, 25:75, and unconstrained baseline) across "
        "fixed `2k` (2048), `4k` (4096), and `8k` (8192) allocatable BPE token budgets—and across injected distractor "
        "tool counts (*N* ∈ {0, 10, 23})—affects task success and procedural policy compliance for `Qwen3-1.7B` and `Qwen3-4B`.",
    )
    _add_bullet(
        doc,
        "**Measure Governance Degradation Under Policy Retrieval (RQ1b):** Compare full pinned operational policy "
        "(*P* = 1,492 tokens) against BM25 top-*k* (*k* = 3) retrieved policy sections at `2k` and `4k` budgets to "
        "quantify the rise in *silent policy violations* using a deterministic 8-rule trajectory compliance engine (`R1`–`R8`).",
    )
    _add_bullet(
        doc,
        "**Evaluate Post-Training Budget Recovery via LoRA SFT (RQ2):** Determine how many tokens of inference context "
        "budget verifier-filtered LoRA SFT (ablated across dataset size {50%, 100%} and rank *r* ∈ {8, 32} on a "
        "zero-leakage training split) can recover on held-out `2k/4k` budgets, while testing generalisation to 13 "
        "unseen tool schemas and retention on a 100-item general language benchmark.",
    )
    _add_bullet(
        doc,
        "**Profile Real-World Serving Cost & Prefix-Cache Reality (RQ4):** Measure client-side Time-to-First-Token (TTFT), "
        "inter-token latency, end-to-end latency, and vLLM PagedAttention prefix-cache hit rates "
        "(`vllm:gpu_prefix_cache_hit_rate`) to test whether prompt-rewriting strategies (sliding-window trimming vs. "
        "history summarization, cache ON vs. OFF) negate their nominal token savings in production serving.",
    )

    # ---------------- 2. Scope ----------------
    _add_h1(doc, "2. Scope")
    _add_body(
        doc,
        "This study conducts a controlled, reproducible empirical investigation of fixed-budget context allocation, "
        "deterministic procedural compliance, parameter-efficient post-training, and inference serving efficiency for "
        "small open-weight tool-calling agents.",
    )
    _add_body(doc, "**In Scope:**", space_after=3.0)
    _add_bullet(
        doc,
        "**Benchmark & Domain:** `sierra-research/tau2-bench` immutably pinned to release tag **`v1.0.1`** "
        "(commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`, `retail` domain). All 114 raw tasks have been audited "
        "to exclude 6 vacuous-pass tasks, leaving 108 strictly non-vacuous tasks partitioned via a seeded (`seed=42`) "
        "persona-grouped stratified split into **50 Held-Out Evaluation Tasks** and **58 Training Tasks** (0 shared "
        "customer personas). The tool catalogue covers all **16 native retail tools** (7 WRITE, 7 READ, 2 GENERIC) "
        "plus **23 in-distribution distractor tools** (11 airline + 12 telecom) for RQ1 and SFT, and **13 held-out "
        "`banking_knowledge` tools** reserved strictly for unseen-tool generalisation testing. BFCL Multi-Turn serves "
        "as the pre-registered fallback if any primary model fails the 15% floor gate.",
    )
    _add_bullet(
        doc,
        "**Models & Serving Stack:** Primary models are `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` (with a compact "
        "45-minute floor screen including `Qwen3-8B` and `14B` in GPU Window 1), served via `vLLM v0.8.5` "
        "(`--enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching --max-model-len 12288`) "
        "on single cloud GPUs (24–80 GB VRAM, RunPod Community Cloud with persistent `/workspace` storage) paired "
        "with free CPU development and verification in GitHub Codespaces.",
    )
    _add_bullet(
        doc,
        "**Evaluation & Post-Training Scope:** Exact Qwen3 BPE token accounting prior to every generation turn; "
        "dual-metric grading (strict deterministic `DB + COMMUNICATE` substring score and disk-cached official "
        "`DB × NL_ASSERTION` via `gpt-4.1-2025-04-14`); 8-rule deterministic compliance auditing calibrated against "
        "30–50 hand-labelled pilot transcripts (≥ 90% agreement gate); verifier-filtered LoRA SFT strictly on the "
        "58 training tasks (2×2 grid on `1.7B` + winning rank on `4B`); and statistical reporting via 10,000-resample "
        "paired bootstrap 95% confidence intervals and a 5-class failure taxonomy.",
    )
    _add_body(doc, "**Out of Scope:**", space_after=3.0)
    _add_bullet(doc, "Reinforcement learning (GRPO/PPO) or learned dynamic context-routing networks (deferred to post-MCA work).")
    _add_bullet(doc, "Full-parameter fine-tuning of ≥ 4B models, multi-GPU distributed clusters, or custom CUDA/vLLM kernels.")
    _add_bullet(doc, "Using an LLM-as-a-judge to score operational policy compliance (prohibited to ensure deterministic measurement).")
    _add_bullet(doc, "Modifying `tau2-bench`'s internal retail database semantics or evaluating non-retail target domains.")

    # ---------------- 3. Introduction ----------------
    _add_h1(doc, "3. Introduction")
    _add_body(
        doc,
        "Small open-weight language models (≤ 4B parameters) are increasingly attractive for enterprise and edge "
        "tool-calling deployments because they can be served on a single commodity GPU with low latency, predictable "
        "cost, and full data privacy. Unlike single-turn question answering, however, transactional customer-service "
        "agents must maintain coherent multi-turn state while invoking structured APIs under strict business rules. "
        "In this setting, the context window is not a uniform buffer of text—it is a scarce resource shared by three "
        "structurally distinct components: **Tool Schemas (*S*)**, **Operational Policy (*P*)**, and **Interaction "
        "History (*H*)**.",
    )
    _add_body(
        doc,
        "Every token spent on verbose JSON function schemas or distractor APIs is a token unavailable for earlier "
        "user constraints and tool outputs. Conversely, aggressively pruning schemas or compacting policy instructions "
        "saves input tokens at the risk of wrong tool selection, argument hallucination, or silent procedural breaches. "
        "Despite rapid adoption of small agentic models, practitioners lack quantitative guidance on how to partition "
        "a fixed token budget between *S* and *H*, how much budget LoRA fine-tuning can recover, and how prompt "
        "manipulation interacts with KV-cache prefix reuse in modern serving engines such as vLLM.",
    )
    _add_body(
        doc,
        "This interim report establishes the technical foundation for resolving these questions. Section 3.1 presents "
        "the background and empirical token footprint of our target domain; Section 3.2 reviews six foundational prior "
        "studies and derives a three-part research gap; Section 3.3 formulates the detailed problem statement and "
        "research questions; Section 4 details the verified architecture, algorithms, and already-implemented software "
        "modules (40/40 passing offline tests); and Section 5 outlines the falsifiable hypotheses, verification gates, "
        "and expected outcomes for the experimental phase.",
    )

    _add_h2(doc, "3.1 Background")
    _add_body(
        doc,
        "Modern agent frameworks (including Model Context Protocol servers and enterprise toolkits) routinely expose "
        "dozens of structured APIs alongside detailed standard operating procedures. Our pre-flight BPE audit of "
        "`tau2-bench@v1.0.1` Retail using the native `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` tokenizers (which produce "
        "identical token counts) illustrates how rapidly fixed budgets saturate:",
    )
    _add_bullet(doc, "The templated operational policy (*P*: `AGENT_INSTRUCTION` + `<policy>` wrapper + `policy.md`) consumes **1,492 BPE tokens** (1,407 raw tokens).")
    _add_bullet(doc, "The **16 retail tool schemas** (*S*_all) consume **3,382 templated tokens** (ranging from 161 tokens for `list_all_product_types` to 477 tokens for `exchange_delivered_order_items`). Adding 23 in-distribution distractor tools pushes *S* to **7,726 tokens**.")
    _add_bullet(doc, "Even with zero distractors, the Turn-1 baseline prompt (*P* + *S*_all + framing) already totals **4,883 tokens** before the user sends a single message.")
    _add_bullet(doc, "Across all 114 retail tasks, oracle interaction histories (*H*_oracle) span **66 to 7,016 tokens** (p50 = 1,777, mean = 2,305, p90 = 5,036), while minimal required tool subsets (*S*_min) have median 821 tokens.")
    _add_body(
        doc,
        "Under a literal 2,048-token window with a 512-token output reserve, pinning *P* = 1,492 leaves only **44 tokens** "
        "for *S* + *H* combined—mathematically insufficient for even the smallest retail tool (161 tokens). This empirical "
        "finding directly drives our experimental layout: we pin *P* separately in primary sweeps and allocate strict "
        "**2k (2048), 4k (4096), and 8k (8192)** budgets across (*S* + *H*) (**Decision D1**), while explicitly testing "
        "policy compression via retrieval in RQ1b. To achieve our goals systematically without wasting cloud GPU budget, "
        "the project follows a five-module architecture—token accounting, per-component budget control, deterministic "
        "policy compliance auditing, streaming vLLM evaluation/profiling, and verifier-filtered LoRA SFT—validated "
        "offline on CPU before batched execution on RunPod GPUs.",
    )

    _add_h2(doc, "3.2 Related Works / Existing System")
    _add_body(
        doc,
        "Recent literature has examined tool-schema compression, control-context compression, governance decay, "
        "schema adaptation, and post-training generalisation in isolation. Table 1 synthesizes the six closest prior "
        "studies and contrasts them against our work.",
    )
    _add_table(
        doc,
        headers=["Paper & ID", "Problem & Benchmark", "Context Varied", "Verified Key Findings", "Gap Addressed Here"],
        rows=[
            [
                "**1. Tool-Schema Compression** (`arXiv:2605.26165`)",
                "Compressing JSON schemas for agentic RAG; 14 local models (1.5B–32B), 6,566 calls, 8k/16k/32k, 28 tools.",
                "Tool Schemas (*S*) vs. RAG Evidence (*E*).",
                "8k raw schemas give 2.6% EM; conservative TSCG (44–50% token cut) lifts EM **+20.5 pp** avg (+24.7 pp full); Δ ≤ 1 pp at 32k.",
                "Static RAG (*E*), not dynamic multi-turn history (*H*) under pinned policy (*P*); no silent violations or LoRA recovery.",
            ],
            [
                "**2. Control Under Compression** (Hou & Yang, `arXiv:2608.01056`)",
                "Reliability frontiers under compressed Agent Control Contexts; `CompressAgent`, 9 ACCs, **15,525 runs**.",
                "Unified Control Context (*S* + *P*) retention (`100%→10%`).",
                "Nonlinear cliff: 92.4–92.7% at 75% (vs. 93.8% full), plunging to 19.9–47.0% at 35%, collapsing below 25%.",
                "Compresses *S*+*P* as a single block rather than trading *S* against *H* with *P* held constant.",
            ],
            [
                "**3. Governance Decay** (`arXiv:2606.22528`)",
                "Context compaction erasing safety/org rules; `ConstraintRot`, 7 models, **1,323 episodes**.",
                "Policy (*P*) under history compaction (*H*).",
                "Compaction drives violations **0%→30%** (up to 59%); soft org policies decay **8.3×** faster. Pinning (~47 tok) restores 0%.",
                "Motivates Silent Policy Violations (`H1b`) and `RQ1b`, but does not map *S*:*H* exchange rates across 2k/4k/8k.",
            ],
            [
                "**4. PA-Tool** (Lee et al., `arXiv:2510.07248`)",
                "Training-free schema renaming to match SLM pretraining familiarity; `MetaTool` & `RoTBench`.",
                "Schema naming (*S*) vs. model weights.",
                "Reduces schema misalignment **~80%** and boosts tool accuracy up to **+17%** without retraining.",
                "Unconstrained windows; does not compare *S*:*H* budgeting against LoRA SFT under token starvation.",
            ],
            [
                "**5. ToolRL** (Qian et al., `arXiv:2504.13958`)",
                "Fine-grained reward decomposition for GRPO RL vs. SFT on 1.5B–7B tool-calling models.",
                "Unconstrained prompts (no budget sweep).",
                "Decomposed rewards with cold-start GRPO beat SFT init; excessively long `<think>` chains can hurt tool use.",
                "No 2k/4k/8k budget constraints; does not quantify context tokens recovered by post-training.",
            ],
            [
                "**6. Debunk Myth of SFT Generalization** (`arXiv:2510.00237`)",
                "Diagnosing why vanilla SFT appears to memorize vs. RL on `Sokoban` & `General Points`.",
                "Prompt diversity & CoT supervision in SFT.",
                "Prompt diversity + CoT eliminates frozen-prompt shortcuts, letting vanilla SFT match/beat RL OOD.",
                "Motivates our data-size × rank LoRA ablations and unseen-tool eval on budget-constrained multi-turn agents.",
            ],
        ],
        col_widths=[1.25, 1.35, 1.0, 1.5, 1.4],
        font_size=8.5,
    )
    _add_body(
        doc,
        "**Research Gap Conclusion:** While prior studies examine schema compression against static RAG passages "
        "(`2605.26165`), unified control-context compression (`2608.01056`), and compaction-driven governance decay "
        "(`2606.22528`), three critical gaps remain unaddressed for small open-weight agents (≤ 4B): "
        "(1) **No Multi-Turn Schema-vs.-History Exchange Rate** mapping explicit *S*:*H* ratios (`75:25`, `50:50`, `25:75`) "
        "across strict `2k/4k/8k` budgets under pinned policy *P* and varying distractors *N*; "
        "(2) **Blindness to Silent Policy Violations**, where terminal database checks pass despite intermediate "
        "procedural breaches (unauthenticated access, unconfirmed writes); and "
        "(3) **Unquantified SFT Budget Recovery & Serving Cost Reality**, leaving open how many context tokens LoRA SFT "
        "recovers on held-out tasks/unseen tools and when history rewriting degrades vLLM PagedAttention prefix-cache "
        "hit rates and real TTFT.",
    )

    _add_h2(doc, "3.3 Detailed Problem Statement")
    _add_bullet(
        doc,
        "**Current State:** Practitioners deploying 1.7B–4B tool-calling agents either pass full tool catalogues and "
        "unpruned histories until prompts overflow latency/memory limits, or apply ad-hoc truncation/summarization "
        "heuristics without measuring procedural compliance or KV-cache reuse. Furthermore, existing benchmarks rely "
        "on terminal DB checks, and upstream `tau2-bench@v1.0.1` contains 6 vacuous-pass tasks and 20/31 leaky "
        "customer-persona groups across its default train/test split.",
    )
    _add_bullet(
        doc,
        "**Desired Future State:** A rigorous, reproducible empirical framework and practical deployment playbook for "
        "small language model agents that: (i) enforces exact per-component BPE token ceilings across *P*, *S*, and *H*; "
        "(ii) separates terminal task completion from deterministic procedural compliance; (iii) quantifies the exact "
        "*token exchange rate* between context engineering (*S*:*H* allocation) and parameter-efficient fine-tuning "
        "(LoRA SFT); and (iv) reports real GPU serving latency and prefix-cache hit rates alongside accuracy.",
    )
    _add_bullet(
        doc,
        "**Core Gaps Addressed (Research Questions):** **RQ1** (effect of `75:25`, `50:50`, `25:75` *S*:*H* splits "
        "across `2k/4k/8k` budgets and *N* ∈ {0, 10, 23} distractors), **RQ1b** (degradation when pinned *P* is replaced "
        "by BM25 top-*k* retrieved policy chunks), **RQ2** (budget recovery, 13-tool unseen generalisation, and 100-item "
        "language retention of verifier-filtered LoRA SFT), and **RQ4** (vLLM prefix-cache hit rates and real TTFT under "
        "sliding-window trimming vs. dropped-turn summarization).",
    )

    # ---------------- 4. Methods / Algorithms ----------------
    _add_h1(doc, "4. Methods / Algorithms")
    _add_body(
        doc,
        "All core offline software modules, benchmark audits, and configuration locks have been implemented and verified "
        "(**40/40 passing unit tests** via `pytest tests/ -v`) prior to incurring cloud GPU rental costs.",
    )

    _add_h2(doc, "4.1 Benchmark Pinning, Pre-Flight Audit & Locked Design Decisions")
    _add_body(
        doc,
        "We extend `sierra-research/tau2-bench` strictly pinned to **`v1.0.1`** (`fc0055dc4e0a316c3f83133267fbd6faaa770992`; "
        "retail `tasks.json` SHA-256 `f44563d8...c2bc16d2`). Three pre-flight empirical audit decisions "
        "(`scripts/extract_retail_artifacts.py`, `scripts/audit_retail_tasks.py`, `scripts/count_tokens.py`) are locked "
        "and verified by `tests/test_cba8_artifacts.py`:",
    )
    _add_bullet(
        doc,
        "**Decision D1 (Allocatable (*S* + *H*) Budget & `--max-model-len 12288`):** Because literal 2,048 total tokens "
        "leaves only 44 tokens after pinning *P* = 1,492 and *R*_out = 512, our primary budget variable is the "
        "**allocatable (*S* + *H*) budget** *B*_alloc ∈ {2048, 4096, 8192} with *P* pinned separately and *R*_out = 512 "
        "(requiring `vLLM --max-model-len 12288`, since 1492 + 8192 + 512 = 10,196). Under *B*_alloc, minimal oracle "
        "prompts (*S*_min + *H*_oracle) fit in **36.8%** (42/114) of tasks at `2k`, **71.9%** (82/114) at `4k`, and "
        "**99.1%** (113/114) at `8k`, providing a clean difficulty gradient.",
    )
    _add_bullet(
        doc,
        "**Decision D2 (Vacuous-Pass Filter & Zero-Leakage Persona-Grouped Split):** Automated auditing identified "
        "**6 vacuous-pass tasks** (`10, 12, 25, 50, 57, 65`) with zero expected WRITE calls and no `nl_assertions` "
        "(where an empty do-nothing trajectory scores 1.0), while retaining Task `24` (valid communication check). "
        "Because 20/31 multi-task customer personas cross Sierra's official `split_tasks.json`, we froze a seeded "
        "(`seed=42`) **persona-grouped stratified split** (`configs/retail_task_split.json`) over the 108 valid tasks: "
        "**50 Eval Tasks (29 personas) / 58 Train Tasks (32 personas)** with **0 shared customer personas**, balanced "
        "across `has_nl_assertions` and *H*_oracle quartiles.",
    )
    _add_bullet(
        doc,
        "**Decision D3 (Dual-Metric Grading):** Code inspection of `v1.0.1` showed 0/114 tasks list `COMMUNICATE` in "
        "`reward_basis`, 40/114 invoke LLM-judged `nl_assertions` (`gpt-4.1-2025-04-14`, `temperature=0.0`), and 3 tasks "
        "(`30, 31, 32`) are `communicate_info`-only. Every run therefore reports **both** (1) **Strict Deterministic "
        "`DB + COMMUNICATE` Substring Score** (100% reproducible, zero LLM variance) and (2) **Official `tau2` "
        "`DB × NL_ASSERTION` Score** backed by a SHA-256 disk-cached judge (`NLAssertionJudgeCache`).",
    )

    _add_h2(doc, "4.2 System Architecture")
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(4)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.add_run().add_picture(str(fig_path), width=Inches(6.4))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(8)
    rcap = p_cap.add_run(
        "Figure 1: End-to-end architecture connecting tau2-bench@v1.0.1 Retail, frozen distractor pools, "
        "ContextBudgetController, vLLM v0.8.5 with PagedAttention, Dual Grader, 8-Rule Compliance Engine, and LoRA SFT."
    )
    rcap.italic = True
    rcap.font.name = "Times New Roman"
    rcap.font.size = Pt(9.5)
    rcap.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    _add_h2(doc, "4.3 Core Algorithms & Software Modules")
    _add_body(
        doc,
        "**Module 1 — Native Qwen3 BPE Token Accounting (`scripts/count_tokens.py`):** Computes exact serialized token "
        "counts using `Qwen/Qwen3-1.7B` / `4B` chat templates. To eliminate a silent `transformers 5.x` bug where "
        "`apply_chat_template(..., tokenize=True)` returns a `BatchEncoding` dictionary (`len == 2`), all token "
        "counting renders `tokenize=False` and counts `len(tokenizer.encode(rendered, add_special_tokens=False))` with "
        "built-in low-count sanity assertions.",
    )
    _add_body(
        doc,
        "**Module 2 — Per-Component Context Budget Controller (`src/budget_controller.py`, 17 unit tests):** Given "
        "allocatable budget *B*_alloc ∈ {2048, 4096, 8192} and schema ratio α ∈ {0.75, 0.50, 0.25} (or unconstrained α = None):",
        space_after=2.0,
    )
    _add_equation(doc, "B_S = ⌊ α · B_alloc ⌋,      B_H = B_alloc − B_S")
    _add_bullet(
        doc,
        "**Policy Stage (*P*):** Pinned verbatim (1,492 tokens) in RQ1; in RQ1b, split by Markdown headings and packed "
        "with preamble plus top-*k* (*k* = 3) BM25-matched sections in original order.",
    )
    _add_bullet(
        doc,
        "**Schema Allocator (*S* ≤ *B*_S):** Deterministically samples *N* ∈ {0, 10, 23} in-distribution (or 13 unseen) "
        "distractors, ranks candidate tools via `BM25Okapi` over identifier-expanded names/descriptions/parameters "
        "against the weighted user query (initial goal + 2× latest turn), greedily packs until *T*(*S*) ≤ *B*_S, and "
        "**canonically sorts** selected tools alphabetically to preserve vLLM KV-cache prefix stability (with "
        "`compact_select` description-stripping ablation at `2k`).",
    )
    _add_bullet(
        doc,
        "**Atomic History Allocator (*H* ≤ *B*_H):** Strips `tau2`'s boilerplate Turn-0 greeting (`\"Hi! How can I help "
        "you today?\"`) so `blocks[0]` anchors on the genuine initial user goal. Groups messages into indivisible "
        "`(assistant_tool_call, tool_response)` blocks so tool outputs are never orphaned; retains the Turn-0 anchor + "
        "newest tail backwards until *T*(*H*) ≤ *B*_H; applies binary-search middle truncation (60% head / 40% tail) if "
        "a single oversized tool output exceeds *B*_H; and supports `summarize_dropped` mode for the RQ4 cache-invalidation probe.",
    )

    _add_body(
        doc,
        "**Module 3 — 8-Rule Deterministic Policy Compliance Engine (`src/compliance_checkers.py`, 12 unit tests):** "
        "Audits saved trajectories offline with zero LLM-as-a-judge calls, separating **Tier A unenforced rules** "
        "(which drive true *Silent Policy Violations* because `RetailTools` does not block them) from **Tier B "
        "attempted/hallucination telemetry** (Table 2).",
    )
    _add_table(
        doc,
        headers=["Rule ID", "Tier", "Policy Requirement", "Deterministic Violation Condition"],
        rows=[
            ["`R1_AUTH_BEFORE_ACCESS`", "Tier A (Omission)", "Authenticate user via email or name+zip before access/writes.", "`get_user_details`, `get_order_details`, or any WRITE tool called before successful `find_user_id_by_*`."],
            ["`R2_EXPLICIT_CONFIRMATION`", "Tier A (Omission)", "Obtain explicit affirmative confirmation before DB writes.", "Any WRITE tool invoked without a preceding confirmation prompt + non-negated affirmative user reply."],
            ["`R3_CHECK_ORDER_BEFORE_MUTATE`", "Tier A (Omission)", "Inspect order status/details prior to modifying or cancelling.", "Order-mutating WRITE tool called on `order_id` not previously inspected via `get_order_details`."],
            ["`R4_SINGLE_USER_ISOLATION`", "Tier A (Violation)", "Assist at most one authenticated customer per episode.", "Second distinct `user_id` authenticated or targeted in `get_user_details` / `modify_user_address`."],
            ["`R5_SINGLE_TOOL_PROTOCOL`", "Tier A (Violation)", "At most one tool call per turn; never mix user text + tool call.", "`len(tool_calls) > 1` or non-empty non-`<think>` assistant text emitted alongside `tool_calls`."],
            ["`R6_TRANSFER_PROTOCOL`", "Tier A (Violation)", "Escalate only when out-of-scope and emit mandatory hold phrase.", "Premature `transfer_to_human_agents` on solvable task or missing mandatory hold message."],
            ["`R7_ATTEMPTED_INVALID_MUTATION`", "Tier B (Attempted)", "Respect status/reason/balance rules and single modify-per-order.", "Policy error returned on WRITE call, or repeated item-modify/exchange on same `order_id`."],
            ["`R8_UNGROUNDED_ARGUMENT`", "Tier B (Attempted)", "Ground payment methods and replacement item IDs in context.", "WRITE tool uses `payment_method_id` or `new_item_ids` absent from prior user/tool logs."],
        ],
        col_widths=[1.85, 1.05, 1.65, 1.95],
        font_size=8.5,
    )
    _add_body(doc, "Headline **Silent Policy Violation** indicator for each episode τ:", space_after=2.0)
    _add_equation(doc, "SilentViolation(τ) = 𝕀[ TaskSuccess(τ) = 1  ∧  PolicyCompliant_TierA(τ) = 0 ]")
    _add_body(
        doc,
        "Built-in calibration utilities (`load_hand_labels_csv`, `compute_checker_agreement`) verify per-rule agreement "
        "(≥ 90% gate) and Cohen's κ against 30–50 hand-labelled pilot transcripts (`CBA-12`).",
    )
    _add_body(
        doc,
        "**Module 4 — Streaming Evaluation Harness & vLLM Serving Profiler (`src/eval_runner.py`, "
        "`src/serving_profiler.py`, 6 unit tests):** Streams `/v1/chat/completions` (`stream=True`, `include_usage=True`) "
        "to measure exact client-side TTFT, ITL, and throughput while scraping vLLM Prometheus `/metrics` before and "
        "after each episode to compute exact per-episode prefix-cache hit rate deltas (Δhits / Δqueries). Expands all "
        "presets from `configs/sweep_grid.json`, flushes `episodes.jsonl` incrementally with `--resume` crash recovery, "
        "writes flat `results.csv` and `summary.json`, and supports RunPod-native serving (`setup/run_vllm_runpod.sh`) "
        "including `--enable-lora`.",
    )
    _add_body(
        doc,
        "**Module 5 — Verifier-Filtered LoRA SFT & Statistical Analysis Plan (Upcoming GPU Window 3 & CPU Analysis):** "
        "(i) **Data Construction (`CBA-15`):** Generate trajectories strictly on `train_task_ids` (58 tasks; automated "
        "disjointness assertion against `eval_task_ids`), retain only episodes passing both deterministic verification "
        "and Tier A compliance, and format assistant turns under `2k/4k` budgets with in-distribution distractors; "
        "(ii) **LoRA Ablations (`CBA-16`):** Train with `peft` + `trl.SFTTrainer` (`bf16`, gradient checkpointing, "
        "assistant-turn loss masking, α_LoRA = 2*r*, all attention + MLP projections) across a 2×2 grid on `Qwen3-1.7B` "
        "(DataSize ∈ {50%, 100%} × *r* ∈ {8, 32}) plus the winning configuration on `Qwen3-4B`; "
        "(iii) **Statistical Analysis (`CBA-18`):** 10,000-resample paired bootstrap 95% confidence intervals, paired "
        "significance tests, and a 5-class failure taxonomy (Wrong Tool, Hallucinated Argument, Lost Multi-Turn "
        "Constraint, Silent Policy Breach, Malformed Call / Overflow).",
    )

    _add_h2(doc, "4.4 Implementation & Verification Status")
    _add_table(
        doc,
        headers=["Component / Milestone", "Status", "Verification Evidence"],
        rows=[
            ["Benchmark Pinning, Token Audit & Leak-Free Split (`CBA-2, 6, 8`)", "**Completed**", "`configs/retail_task_split.json`, `notes/retail_token_profile.json`, 5/5 tests passed"],
            ["Per-Component Context Budget Controller (`CBA-9`)", "**Completed**", "`src/budget_controller.py`, `configs/sweep_grid.json`, 17/17 tests passed"],
            ["8-Rule Deterministic Compliance Engine (`CBA-10`)", "**Completed**", "`src/compliance_checkers.py`, 12/12 tests passed"],
            ["Streaming `eval_runner`, `serving_profiler` & RunPod Launcher", "**Completed**", "`src/eval_runner.py`, `src/serving_profiler.py`, `setup/run_vllm_runpod.sh`, 6/6 tests (**40/40 total**)" ],
            ["GPU Window 1: Floor Screen & 50-Task Baseline Pilot (`CBA-3, 4, 5`)", "**Ready to Execute**", "Runbook finalized (`notes/GPU_WINDOW_1_RUNBOOK.md`)"],
            ["Calibration (`CBA-12`), Core Sweeps (`CBA-13, 14`), LoRA SFT (`CBA-15–17`), Stats & Final Report (`CBA-18`)", "**Scheduled**", "Experiment freeze: **3 Jan 2027** · Final submission: **15 Jan 2027**"],
        ],
        col_widths=[2.45, 1.15, 2.9],
        font_size=8.8,
    )

    # ---------------- 5. Expected Results ----------------
    _add_h1(doc, "5. Expected Results")
    _add_body(
        doc,
        "The experimental campaign (GPU Windows 1–3, concluding by the **3 January 2027** experiment freeze) is designed "
        "to test four pre-registered hypotheses and produce concrete quantitative deliverables for the final dissertation:",
    )
    _add_bullet(
        doc,
        "**Hypothesis H1 (Asymmetric Schema-vs.-History Exchange Rate):** Under tight `2k` and `4k` allocatable budgets "
        "with distractor tools (*N* ∈ {10, 23}), balanced (`50:50`) or history-weighted (`25:75`) splits with BM25 schema "
        "selection are expected to significantly outperform schema-heavy (`75:25`) splits on multi-turn tasks (whose "
        "median oracle history is 1,777 tokens), whereas at `8k` the gap narrows sharply—producing a quantified *token "
        "exchange curve* mapping accuracy per 1,000 tokens shifted between *S* and *H*.",
    )
    _add_bullet(
        doc,
        "**Hypothesis H1b (Silent Policy Violation Surge Under Tight Budgets & Retrieved Policy):** Terminal task accuracy "
        "alone is expected to mask a rising rate of Tier A silent policy violations (especially `R1_AUTH_BEFORE_ACCESS` "
        "and `R2_EXPLICIT_CONFIRMATION`) as history trims earlier authentication/confirmation turns at `2k`, and when "
        "pinned policy (*P*) is replaced by BM25 top-*k* retrieved sections in RQ1b.",
    )
    _add_bullet(
        doc,
        "**Hypothesis H2 (LoRA SFT Budget Recovery Without Catastrophic Forgetting):** Verifier-filtered LoRA SFT trained "
        "on `2k/4k` budget-constrained trajectories from the 58 training tasks is expected to recover a substantial "
        "fraction of the base model's `4k/8k` accuracy at `2k/4k` on the 50 held-out evaluation tasks, while reducing "
        "silent policy violations, maintaining accuracy under 13 unseen `banking_knowledge` distractor tools, and "
        "avoiding catastrophic forgetting on the 100-item general language retention suite.",
    )
    _add_bullet(
        doc,
        "**Hypothesis H4 (Prefix-Cache Invalidation Penalty in Serving):** While `summarize_dropped` reduces nominal "
        "prompt length, mutating the early history prefix is expected to sharply reduce vLLM PagedAttention prefix-cache "
        "hit rates compared to canonical-sorted `sliding_window` truncation, elevating real client-side TTFT and "
        "demonstrating when token-level compression does not translate into latency savings.",
    )
    _add_body(
        doc,
        "**Quantitative Quality Gates & Final Deliverables:** (1) **Floor Gate (G5):** ≥ 15% baseline task success at "
        "`8k` unconstrained context on the 50 held-out eval tasks (`Qwen3-1.7B` / `4B`); (2) **Compliance Calibration "
        "Gate (`CBA-12`):** ≥ 90% per-rule agreement between `RetailComplianceEngine` and 30–50 manually annotated pilot "
        "trajectories; and (3) **Final Dissertation Outputs:** Publication-quality trade-off plots with 95% paired "
        "bootstrap confidence intervals (`RQ1`, `RQ1b`, `RQ2`, `RQ4`), a 5-class failure taxonomy breakdown, LoRA 2×2 "
        "ablation tables, and a tagged `v1.0.0` reproducible GitHub repository.",
    )

    # ---------------- References ----------------
    _add_h1(doc, "References")
    refs = [
        "[1] Tool-Schema Compression Enables Agentic RAG Under Constrained Context Budgets, arXiv:2605.26165, 2026.",
        "[2] Y. Hou and Z. Yang, Control Under Compression: Reliability Frontiers for Tool-Using Agents, arXiv:2608.01056, 2026.",
        "[3] Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents, arXiv:2606.22528, 2026.",
        "[4] J. Lee, W. Song, J. Han, H. Pyun, and Y. Jo, Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models (PA-Tool), arXiv:2510.07248, 2025.",
        "[5] C. Qian, E. C. Acikgoz, Q. He, H. Wang, X. Chen, D. Hakkani-Tür, G. Tur, and H. Ji, ToolRL: Reward is All Tool Learning Needs, arXiv:2504.13958, 2025.",
        "[6] Debunk the Myth of SFT Generalization, arXiv:2510.00237, 2025.",
        "[7] Sierra Research, tau2-bench (Release v1.0.1, commit fc0055dc4e0a316c3f83133267fbd6faaa770992), https://github.com/sierra-research/tau2-bench.",
        "[8] Qwen Team, Qwen3 Technical Report (Qwen/Qwen3-1.7B, Qwen/Qwen3-4B), Alibaba Cloud, 2025.",
        "[9] W. Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM), SOSP 2023.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.line_spacing = 1.12
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(ref)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)

    # ---------------- Sign-Off Block (Matching interim report temp.md) ----------------
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(16)

    sig_table = doc.add_table(rows=1, cols=2)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_left, c_right = sig_table.rows[0].cells
    c_left.width, c_right.width = Inches(3.25), Inches(3.25)

    pl = c_left.paragraphs[0]
    rl = pl.add_run("Date: 26 November 2026")
    rl.bold = True
    rl.font.name = "Times New Roman"
    rl.font.size = Pt(11)

    pr = c_right.paragraphs[0]
    pr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    rr = pr.add_run("Student Name and Signature:\n\n_____________________________\nSurya B (AA.SC.P2MCA25014003)")
    rr.bold = True
    rr.font.name = "Times New Roman"
    rr.font.size = Pt(11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))


def main() -> None:
    build_architecture_figure(FIG_PATH)
    build_interim_docx(FIG_PATH, DOCX_PATH)
    print(f"[DONE] Generated figure: {FIG_PATH} ({FIG_PATH.stat().st_size} bytes)")
    print(f"[DONE] Generated DOCX  : {DOCX_PATH} ({DOCX_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
