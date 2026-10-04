"""Build the RAG-Forge technical report (PDF) with ReportLab.

All numbers are taken from the project's recorded runs, docs and commits (Phases 1-6).

Usage (macOS; uses the system Times New Roman / Arial / Courier New fonts):
    pip install reportlab
    cd docs/report && python build_paper.py
"""

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

OUT = "RAG-Forge_Technical_Report.pdf"

# ------------------------------------------------------------------ fonts
S = "/System/Library/Fonts/Supplemental/"
for name, file in [
    ("Serif", "Times New Roman.ttf"),
    ("Serif-Bold", "Times New Roman Bold.ttf"),
    ("Serif-Italic", "Times New Roman Italic.ttf"),
    ("Serif-BoldItalic", "Times New Roman Bold Italic.ttf"),
    ("Sans", "Arial.ttf"),
    ("Sans-Bold", "Arial Bold.ttf"),
    ("Sans-Italic", "Arial Italic.ttf"),
    ("Mono", "Courier New.ttf"),
    ("Mono-Bold", "Courier New Bold.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, S + file))
pdfmetrics.registerFontFamily(
    "Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic", boldItalic="Serif-BoldItalic"
)
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic")
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold")

# ------------------------------------------------------------------ palette (dataviz reference)
INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
GRID = colors.HexColor("#e1e0d9")
SURFACE = colors.HexColor("#fcfcfb")
BLUE = colors.HexColor("#2a78d6")
ORANGE = colors.HexColor("#eb6834")
AQUA = colors.HexColor("#1baf7a")
PANEL = colors.HexColor("#f3f2ee")

# ------------------------------------------------------------------ styles
body = ParagraphStyle(
    "body",
    fontName="Serif",
    fontSize=10.5,
    leading=14.2,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
    textColor=INK,
)
body_left = ParagraphStyle("body_left", parent=body, alignment=TA_LEFT)
title = ParagraphStyle(
    "title",
    fontName="Sans-Bold",
    fontSize=20,
    leading=25,
    alignment=TA_CENTER,
    spaceAfter=6,
    textColor=INK,
)
subtitle = ParagraphStyle(
    "subtitle",
    fontName="Sans",
    fontSize=11.5,
    leading=15,
    alignment=TA_CENTER,
    textColor=INK2,
    spaceAfter=4,
)
author_style = ParagraphStyle(
    "author",
    fontName="Sans-Bold",
    fontSize=11,
    leading=14,
    alignment=TA_CENTER,
    textColor=INK,
    spaceBefore=6,
    spaceAfter=2,
)
meta = ParagraphStyle(
    "meta",
    fontName="Sans",
    fontSize=9,
    leading=12,
    alignment=TA_CENTER,
    textColor=MUTED,
    spaceAfter=14,
)
h1 = ParagraphStyle(
    "h1",
    fontName="Sans-Bold",
    fontSize=13.5,
    leading=17,
    spaceBefore=14,
    spaceAfter=6,
    textColor=INK,
    keepWithNext=1,
)
h2 = ParagraphStyle(
    "h2",
    fontName="Sans-Bold",
    fontSize=11,
    leading=14,
    spaceBefore=9,
    spaceAfter=4,
    textColor=INK,
    keepWithNext=1,
)
h3 = ParagraphStyle(
    "h3",
    fontName="Sans-Bold",
    fontSize=10,
    leading=13,
    spaceBefore=6,
    spaceAfter=2,
    textColor=INK2,
    keepWithNext=1,
)
abstract_style = ParagraphStyle("abstract", parent=body, fontSize=10, leading=13.5)
caption = ParagraphStyle(
    "caption",
    fontName="Sans",
    fontSize=8.5,
    leading=11,
    textColor=INK2,
    alignment=TA_LEFT,
    spaceBefore=3,
    spaceAfter=10,
)
table_caption = ParagraphStyle("table_caption", parent=caption, spaceBefore=0, spaceAfter=0)
cell = ParagraphStyle("cell", fontName="Sans", fontSize=8.2, leading=10.2, textColor=INK)
cell_b = ParagraphStyle("cell_b", parent=cell, fontName="Sans-Bold")
cell_r = ParagraphStyle("cell_r", parent=cell, alignment=2)
code = ParagraphStyle(
    "code",
    fontName="Mono",
    fontSize=7.8,
    leading=9.6,
    textColor=INK,
    backColor=PANEL,
    borderPadding=5,
    spaceBefore=4,
    spaceAfter=9,
)
ref_style = ParagraphStyle(
    "ref",
    parent=body_left,
    fontSize=9,
    leading=11.5,
    leftIndent=16,
    firstLineIndent=-16,
    spaceAfter=3,
)
bullet_style = ParagraphStyle("bullet", parent=body, spaceAfter=2)

story = []
_sec = [0, 0]
_fig = [0]
_tab = [0]


def P(text, style=body):
    story.append(Paragraph(text, style))


def H1(text):
    _sec[0] += 1
    _sec[1] = 0
    # Start a new page if a heading would land with too little room for content after it.
    story.append(CondPageBreak(130))
    story.append(Paragraph(f"{_sec[0]}&nbsp;&nbsp;{text}", h1))


def H2(text):
    _sec[1] += 1
    story.append(CondPageBreak(120))
    story.append(Paragraph(f"{_sec[0]}.{_sec[1]}&nbsp;&nbsp;{text}", h2))


def H3(text):
    story.append(CondPageBreak(70))
    story.append(Paragraph(text, h3))


def bullets(items, style=bullet_style):
    story.append(
        ListFlowable(
            [ListItem(Paragraph(t, style), leftIndent=12, value="•") for t in items],
            bulletType="bullet",
            start="•",
            leftIndent=12,
            bulletFontName="Serif",
            bulletFontSize=9,
            spaceAfter=6,
        )
    )


def numbered(items):
    story.append(
        ListFlowable(
            [ListItem(Paragraph(t, bullet_style), leftIndent=14) for t in items],
            bulletType="1",
            leftIndent=14,
            bulletFontName="Serif",
            bulletFontSize=10,
            spaceAfter=6,
        )
    )


def CODE(text):
    story.append(Preformatted(text, code))


def table(
    rows,
    widths,
    caption_text=None,
    align_right_from=None,
    header=True,
    bold_last=False,
    left_cols=(),
):
    """rows: list of lists of str (header row first).

    The caption is the table's first row (spanning all columns) and repeats with the header
    when the table splits across pages, so captions never detach and long tables never
    leave half-empty pages.
    """
    data = []
    offset = 0
    if caption_text:
        _tab[0] += 1
        cap = Paragraph(f"<b>Table {_tab[0]}.</b> {caption_text}", table_caption)
        data.append([cap] + [""] * (len(widths) - 1))
        offset = 1
    for r, row in enumerate(rows):
        out = []
        for c, v in enumerate(row):
            is_head = header and r == 0
            st = cell_b if is_head or (bold_last and r == len(rows) - 1) else cell
            if align_right_from is not None and c >= align_right_from and c not in left_cols:
                st = ParagraphStyle("xr", parent=st, alignment=2)
            out.append(Paragraph(str(v), st))
        data.append(out)
    t = Table(data, colWidths=widths, repeatRows=offset + (1 if header else 0), hAlign="LEFT")
    h = offset  # index of the header row
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, h), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, h), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, INK2),
    ]
    if caption_text:
        style += [
            ("SPAN", (0, 0), (-1, 0)),
            ("LEFTPADDING", (0, 0), (-1, 0), 0),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
        ]
    if header:
        style += [
            ("LINEABOVE", (0, h), (-1, h), 0.8, INK),
            ("LINEBELOW", (0, h), (-1, h), 0.5, INK2),
            ("BACKGROUND", (0, h), (-1, h), PANEL),
        ]
    for r in range(h + (1 if header else 0), len(data)):
        if (r - h) % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f9f9f7")))
    t.setStyle(TableStyle(style))
    # The gap is the table's own spaceBefore (not a Spacer flowable), so a preceding
    # heading's keepWithNext binds to the table itself and is never stranded.
    t.spaceBefore = 4
    # Short tables stay whole; splitting them strands a row or two under a repeated header.
    if len(rows) > 8:
        story.append(CondPageBreak(170))  # a split table starts with several rows, not one
    story.append(KeepTogether([t]) if len(rows) <= 8 else t)
    story.append(Spacer(1, 8))


def figure(drawing, caption_text):
    _fig[0] += 1
    story.append(
        KeepTogether(
            [Spacer(1, 4), drawing, Paragraph(f"<b>Figure {_fig[0]}.</b> {caption_text}", caption)]
        )
    )


# ------------------------------------------------------------------ drawings
def box(d, x, y, w, h, label, sub=None, fill=SURFACE, stroke=INK2, bold=True, size=8):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.8))
    ty = y + h / 2 + (3 if sub else -3)
    d.add(
        String(
            x + w / 2,
            ty,
            label,
            fontName="Sans-Bold" if bold else "Sans",
            fontSize=size,
            fillColor=INK,
            textAnchor="middle",
        )
    )
    if sub:
        for i, line in enumerate(sub.split("\n")):
            d.add(
                String(
                    x + w / 2,
                    ty - 10 - i * 9,
                    line,
                    fontName="Sans",
                    fontSize=6.3,
                    fillColor=INK2,
                    textAnchor="middle",
                )
            )


def arrow(d, x1, y1, x2, y2, color=INK2, dashed=False):
    d.add(
        Line(
            x1,
            y1,
            x2,
            y2,
            strokeColor=color,
            strokeWidth=0.9,
            strokeDashArray=[3, 2] if dashed else None,
        )
    )
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    s = 4.5
    p1 = (x2 - s * math.cos(ang - 0.4), y2 - s * math.sin(ang - 0.4))
    p2 = (x2 - s * math.cos(ang + 0.4), y2 - s * math.sin(ang + 0.4))
    d.add(Polygon([x2, y2, *p1, *p2], fillColor=color, strokeColor=color, strokeWidth=0.1))


def architecture_drawing():
    W, H = 470, 216
    d = Drawing(W, H)
    d.add(Rect(0, 0, W, H, fillColor=colors.white, strokeColor=None))
    # API layer
    d.add(Rect(8, 176, W - 16, 36, rx=5, ry=5, fillColor=PANEL, strokeColor=GRID))
    d.add(String(16, 200, "FastAPI", fontName="Sans-Bold", fontSize=8.5, fillColor=INK))
    for i, ep in enumerate(
        ["GET /health", "POST /documents/ingest", "GET /documents", "POST /query"]
    ):
        box(d, 62 + i * 100, 182, 96, 22, ep, bold=False, size=7.2)
    # core components
    box(d, 8, 100, 100, 50, "Ingestion", "loaders · Hugo resolver\nparent/child chunker")
    box(d, 124, 100, 100, 50, "Retrieval", "dense · BM25 · hybrid\nfusion (components)")
    box(d, 240, 100, 100, 50, "Generation", "LLMClient · structured\noutput · citations · cost")
    box(d, 356, 100, 106, 50, "Evaluation", "synthetic dataset · span\nlabels · metrics · runner")
    # storage and model servers
    box(
        d,
        60,
        18,
        150,
        58,
        "PostgreSQL 16 + pgvector 0.8.6",
        "documents · parent_chunks · chunks\n(HNSW, GIN on metadata) · experiments",
    )
    box(d, 250, 18, 100, 58, "Local models", "bge-small-en-v1.5\n(sentence-transformers)")
    box(
        d, 362, 18, 100, 58, "MLX server (host)", "Qwen2.5-3B-Instruct 4-bit\nOpenAI-compatible API"
    )
    for x in (58, 174, 290, 409):
        arrow(d, x, 181, x, 151)
    arrow(d, 58, 100, 110, 77)
    arrow(d, 174, 100, 150, 77)
    arrow(d, 409, 100, 170, 77, dashed=True)
    arrow(d, 174, 100, 290, 77)
    arrow(d, 58, 100, 280, 77, dashed=True)
    arrow(d, 290, 100, 410, 77)
    arrow(d, 224, 125, 240, 125)
    arrow(d, 356, 125, 341, 125, dashed=True)  # evaluation uses the LLM layer
    d.add(
        String(
            W - 8,
            4,
            "solid: request path · dashed: offline / evaluation path",
            fontName="Sans",
            fontSize=6.5,
            fillColor=MUTED,
            textAnchor="end",
        )
    )
    scale = 0.9
    d.scale(scale, scale)
    d.width, d.height = W * scale, H * scale
    return d


def pipeline_drawing():
    W, H = 470, 120
    d = Drawing(W, H)
    steps = [
        ("Sample", "seeded; chunk used\nonce; ≤ 3 per doc"),
        ("Number", "sentences via the\nchunker's rules"),
        ("Ask LLM", "JSON schema;\n1 repair retry"),
        ("Map IDs", "sentence IDs to\nexact spans"),
        ("Validate", "deterministic,\ntype-specific"),
        ("Label, split", "span → chunks;\ndev / test"),
    ]
    w, gap = 70, 10
    for i, (a, b) in enumerate(steps):
        x = 5 + i * (w + gap)
        box(d, x, 52, w, 50, a, b, size=7.6)
        if i < len(steps) - 1:
            arrow(d, x + w, 77, x + w + gap, 77)
    # reject loop
    d.add(
        Line(
            5 + 4 * (w + gap) + w / 2,
            52,
            5 + 4 * (w + gap) + w / 2,
            26,
            strokeColor=ORANGE,
            strokeWidth=0.9,
        )
    )
    d.add(Line(5 + 4 * (w + gap) + w / 2, 26, 5 + w / 2, 26, strokeColor=ORANGE, strokeWidth=0.9))
    arrow(d, 5 + w / 2, 26, 5 + w / 2, 51, color=ORANGE)
    d.add(
        String(
            5 + 2.5 * (w + gap),
            15,
            "rejected: reason counted per type, next attempt (up to 6× or 12× the quota)",
            fontName="Sans",
            fontSize=7,
            fillColor=INK2,
            textAnchor="middle",
        )
    )
    return d


def results_chart():
    """Evidence Recall@5 by question type for the three retrievers (test split)."""
    data = [  # type, n, dense, bm25, hybrid
        ("semantic", 31, 0.672, 0.710, 0.806),
        ("multi_hop", 26, 0.641, 0.689, 0.760),
        ("comparison", 13, 0.449, 0.368, 0.468),
        ("keyword_heavy", 31, 0.871, 0.935, 0.935),
        ("adversarial", 17, 0.824, 1.000, 1.000),
        ("simple_factual", 42, 0.821, 0.905, 0.881),
        ("numerical", 21, 0.571, 0.952, 0.809),
        ("ambiguous", 17, 0.471, 0.471, 0.471),
    ]
    series = (("Dense", BLUE), ("BM25", ORANGE), ("Hybrid", AQUA))
    W, H = 470, 300
    left, right, top, bottom = 118, 40, 30, 28
    plot_w = W - left - right
    row_h = (H - top - bottom) / len(data)
    bar_h = 6.5
    d = Drawing(W, H)
    d.add(Rect(0, 0, W, H, fillColor=SURFACE, strokeColor=None))
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + v * plot_w
        d.add(Line(x, bottom, x, H - top, strokeColor=GRID, strokeWidth=0.5))
        d.add(
            String(
                x,
                bottom - 12,
                f"{v:.2f}",
                fontName="Sans",
                fontSize=7,
                fillColor=MUTED,
                textAnchor="middle",
            )
        )
    d.add(
        String(
            left + plot_w / 2,
            4,
            "Evidence Recall@5 (fraction of required evidence spans in the top 5)",
            fontName="Sans",
            fontSize=7.2,
            fillColor=INK2,
            textAnchor="middle",
        )
    )
    lx = left
    for label, color in series:
        d.add(Rect(lx, H - 16, 9, 9, fillColor=color, strokeColor=None, rx=2, ry=2))
        d.add(String(lx + 13, H - 14.5, label, fontName="Sans", fontSize=7.5, fillColor=INK))
        lx += 70
    for i, (t, n, *vals) in enumerate(data):
        yc = H - top - (i + 0.5) * row_h
        d.add(
            String(
                left - 6,
                yc - 2.5,
                f"{t} (n={n})",
                fontName="Sans",
                fontSize=7.5,
                fillColor=INK,
                textAnchor="end",
            )
        )
        for j, (v, (_, color)) in enumerate(zip(vals, series, strict=True)):
            y = yc + bar_h * 0.5 + 1 - j * (bar_h + 1)
            d.add(
                Rect(
                    left,
                    y,
                    max(v * plot_w, 0.5),
                    bar_h,
                    fillColor=color,
                    strokeColor=SURFACE,
                    strokeWidth=1,
                    rx=2,
                    ry=2,
                )
            )
            d.add(
                String(
                    left + v * plot_w + 3,
                    y + 0.8,
                    f"{v:.2f}",
                    fontName="Sans",
                    fontSize=6.3,
                    fillColor=INK2,
                )
            )
    return d


# ------------------------------------------------------------------ page template
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(
            20 * mm,
            A4[1] - 12 * mm,
            "RAG-Forge: Evaluation-Driven Adaptive RAG · Technical Report (Phases 1–6)",
        )
    canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


doc = BaseDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=20 * mm,
    rightMargin=20 * mm,
    topMargin=18 * mm,
    bottomMargin=16 * mm,
    title="RAG-Forge: Building an Evaluation-Driven RAG System",
    author="Nabeel Ahmad",
    subject="Technical report, Phases 1-6",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=on_page)])
TW = doc.width  # text width

# ================================================================== CONTENT
P(
    "RAG-Forge: Building an Evaluation-Driven RAG System with Span-Anchored Synthetic "
    "Evaluation on a Fully Local Stack",
    title,
)
P(
    "Technical report covering Phases 1–6: infrastructure, ingestion, baseline dense RAG, "
    "evaluation dataset, BM25 and hybrid retrieval",
    subtitle,
)
P("Nabeel Ahmad", author_style)
P(
    "RAG-Forge project · October 2026 · work in progress · repository commits f2854e3 → 726e2a7",
    meta,
)

abstract = (
    "<b>Abstract.</b> Retrieval-augmented generation (RAG) systems are commonly judged by "
    "inspecting a handful of answers. We describe the first six phases of RAG-Forge, a "
    "system built so that every technique can be switched on or off and measured against a "
    "baseline. The system runs entirely on a laptop-class machine (Apple M1, 8 GB) using "
    "open models: bge-small-en-v1.5 embeddings, PostgreSQL with pgvector, and "
    "Qwen2.5-3B-Instruct served locally through MLX. We ingest a pinned snapshot of the "
    "Kubernetes documentation (551 documents, 3,855 chunks) with a structure-aware "
    "parent/child chunker whose design was driven by measurements on the real corpus; an "
    "audit against the embedder's tokenizer exposed, and a calibrated estimator removed, "
    "silent truncation of 4% of embedding inputs. For evaluation we generate 316 synthetic "
    "questions across nine types. Every item passes deterministic validators (only 17% of "
    "1,901 generations were kept), and relevance is anchored to exact character spans, so "
    "labels survive re-chunking. We then compare dense, BM25 and hybrid retrieval under a "
    "strict protocol: every configuration is chosen on a dev split, partly by rules fixed "
    "before tuning, and the held-out test split is evaluated once, with paired bootstrap "
    "tests. On test, BM25 beats dense retrieval (Evidence Recall@5 0.791 vs 0.701; paired "
    "difference +0.090, 95% CI [+0.034, +0.143]). Hybrid fusion reaches 0.802 and is "
    "significantly better than dense retrieval, but statistically indistinguishable from "
    "BM25 overall. Per question type, hybrid gains on semantic (+0.10), multi-hop (+0.07) "
    "and comparison (+0.10) questions and loses on numerical ones (−0.14). We trace much of "
    "BM25's advantage to lexical bias in passage-derived synthetic questions, and we "
    "document the failure modes found at each step, including those of the evaluation data."
)
story.append(
    Table(
        [[Paragraph(abstract, abstract_style)]],
        colWidths=[TW],
        style=[
            ("BACKGROUND", (0, 0), (-1, -1), PANEL),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ],
    )
)
story.append(Spacer(1, 8))

# ------------------------------------------------------------------ 1 Introduction
H1("Introduction")
P(
    "Most RAG prototypes implement a single path, <i>query → vector search → LLM → answer</i>, "
    "and judge quality by eye. That makes it impossible to answer the questions that matter "
    "when engineering such a system. Does hybrid retrieval beat dense retrieval on this "
    "corpus, and for which kinds of questions? How much does a cross-encoder reranker improve "
    "early precision, and at what latency? When corrective retrieval fires, does the extra "
    "LLM call pay for itself? RAG-Forge is organised around these questions. Each technique "
    "(BM25, hybrid fusion, reranking, parent-child retrieval, query rewriting, corrective RAG "
    "[2], a Self-RAG-inspired loop [3] and claim-level verification) is to be implemented "
    "behind a configuration switch and benchmarked against the same baseline, on the same "
    "held-out questions, with quality, latency and cost reported together."
)
P(
    "This report covers the first six of eighteen planned phases: the infrastructure, the "
    "ingestion pipeline, a baseline dense RAG pipeline, the evaluation dataset and retrieval "
    "benchmark, BM25, and hybrid retrieval. Reranking, query rewriting, corrective and "
    "Self-RAG-inspired behaviour and claim verification are <i>not</i> implemented yet, and "
    "no results are reported for them. We describe the architecture and every design "
    "choice, give particular attention to how each evaluation question type is generated "
    "and validated, and report the first controlled comparison between retrieval methods."
)
H3("Contributions of this phase")
bullets(
    [
        "<b>A span-anchored evaluation design.</b> Relevance labels are exact character spans in "
        "the stored document text. Chunk-level relevance is recomputed against whatever index is "
        "being evaluated, so chunking itself becomes a benchmarkable variable instead of "
        "something that silently invalidates the labels (§6.1).",
        "<b>A validated synthetic dataset built with a small local model.</b> 316 items across "
        "nine question types; 17% of generator outputs survive deterministic validation; three "
        "rounds of manual pilot review each changed the generator design (§6).",
        "<b>Evidence Recall@K</b> for questions that need several pieces of evidence, with "
        "bootstrap confidence intervals, paired comparisons and per-type breakdowns (§7).",
        "<b>A controlled comparison of dense, BM25 and hybrid retrieval</b>: dev-only tuning "
        "under selection rules fixed in advance, a single test evaluation, and an analysis "
        "showing how much of BM25's lead comes from the dataset's lexical bias (§8).",
        "<b>Measurement-driven ingestion.</b> Corpus-specific template resolution, and a chunker "
        "whose parameters were corrected by audits against the real tokenizer and the real "
        "corpus (§4).",
        "<b>A fully local, free and reproducible stack</b>, with a provider abstraction that "
        "allows the same experiments to be re-run with a stronger model later (§3, §5).",
    ]
)

# ------------------------------------------------------------------ 2 Background
H1("Background and related work")
P(
    "<b>RAG.</b> Retrieval-augmented generation conditions a language model on retrieved "
    "passages [1]. Retrieval can be lexical, typically BM25 [4], dense, using bi-encoder "
    "embeddings such as the BGE family [8] indexed with approximate nearest-neighbour "
    "structures such as HNSW [7], or hybrid, commonly fused by Reciprocal Rank Fusion [5]. "
    "Cross-encoders, and sequence-to-sequence rankers such as monoT5 [11], rerank a candidate "
    "list at higher cost."
)
P(
    "<b>Adaptive RAG.</b> Corrective RAG (CRAG) [2] adds a lightweight retrieval evaluator, a "
    "fine-tuned T5-large in the original work, that grades retrieved documents and triggers "
    "corrective actions when they are judged irrelevant. Self-RAG [3] fine-tunes a generator "
    "(Llama-2 7B/13B) to emit reflection tokens that decide whether to retrieve and whether "
    "the output is supported. RAG-Forge will implement <i>practical adaptations</i> of these "
    "ideas and does not claim to reproduce either paper. In particular, its Self-RAG-style "
    "decisions will be made by separate graders rather than by a model trained with "
    "reflection tokens."
)
P(
    "<b>Evaluation.</b> Retrieval is conventionally scored with Recall@K, Precision@K, MRR "
    "and NDCG [6]. RAG-specific frameworks such as RAGAS [9] and ARES [15] use LLMs to judge "
    "faithfulness and relevance, and ARES also generates synthetic queries for evaluation. "
    "LLM judges are known to have biases, including self-preference [14]. Small models "
    "trained for grounding verification, such as MiniCheck [10], offer an alternative to large "
    "judges for claim-level checks. Benchmarks built from passages are known to interact with "
    "retrieval method; BEIR [16], for example, found BM25 to be a strong zero-shot baseline. "
    "We return to this lexical-bias concern for our own synthetic questions in §6.7."
)

# ------------------------------------------------------------------ 3 Architecture
H1("System architecture")
P(
    "RAG-Forge is a single Python service rather than a set of microservices. The spec "
    "explicitly prioritised correctness, evaluation and reproducibility over deployment "
    "complexity. Figure 1 shows the components as of Phase 4."
)
figure(
    architecture_drawing(),
    "Component architecture after Phase 4. The API and the evaluation scripts share one "
    "code base. The LLM runs on the host because MLX needs the Apple GPU, which containers "
    "cannot access. The API container reaches it through <font face='Mono'>host.docker.internal</font>.",
)

H2("Technology choices")
table(
    [
        ["Concern", "Choice", "Reason"],
        [
            "API",
            "FastAPI, Pydantic v2",
            "Typed request and response models; dependency injection "
            "makes every component replaceable in tests",
        ],
        [
            "Storage",
            "PostgreSQL 16 + pgvector 0.8.6",
            "Vectors, metadata and document text in one "
            "transactional store; metadata filters are plain SQL; no second system to keep in sync",
        ],
        [
            "ANN index",
            "HNSW with operator class matched to the configured metric",
            "Standard, and representative of larger corpora even though ~4k vectors could be "
            "searched exactly",
        ],
        [
            "DB access",
            "Synchronous SQLAlchemy 2.0 + psycopg 3",
            "Expensive work (embedding, reranking) is CPU/GPU-bound, so async I/O adds complexity "
            "with little benefit at this scale",
        ],
        [
            "Embeddings",
            "BAAI/bge-small-en-v1.5 (384-d, 512-token limit)",
            "Runs locally beside the LLM within 8 GB of RAM",
        ],
        [
            "Lexical retrieval",
            "In-process Okapi BM25 with an analyzer for technical identifiers",
            "PostgreSQL ts_rank is not BM25; ~4k chunks index in memory in about 1 s",
        ],
        [
            "Fusion",
            "Weighted RRF or min-max linear fusion of dense and BM25 lists",
            "Rank-based RRF avoids reconciling score scales; linear chosen on dev (§8.1)",
        ],
        [
            "LLM",
            "Qwen2.5-3B-Instruct [12], 4-bit, served by mlx-lm 0.31.3",
            "Free, offline, uses the M1 GPU; its OpenAI-compatible server needs no custom client",
        ],
        [
            "Packaging",
            "Docker Compose (API + Postgres); host port 5434",
            "Ports 5432 and 5433 were occupied on the development machine",
        ],
    ],
    [0.14 * TW, 0.33 * TW, 0.53 * TW],
    "Core technology choices and their rationale.",
)

H2("Configuration as the experiment interface")
P(
    "Every tunable value lives in one validated settings object (pydantic-settings), read from "
    "flat environment variables (<font face='Mono'>TOP_K</font>, "
    "<font face='Mono'>RELEVANCE_THRESHOLD</font>, …) and grouped internally into database, "
    "embedding, chunking, retrieval, adaptive, LLM and observability sections. The design "
    "rules are:"
)
bullets(
    [
        "<b>No hard-coded thresholds.</b> Components receive settings explicitly, which is what "
        "allows the experiment runner to build a different configuration per experiment.",
        "<b>Validation at load time.</b> This covers ranges (thresholds in [0, 1]), hard caps on "
        "loop iterations (≤ 5, so a typo cannot create a runaway corrective loop) and "
        "cross-field constraints (<font face='Mono'>RERANK_TOP_K ≤ RERANK_CANDIDATES ≤ "
        "RETRIEVAL_CANDIDATES</font>; fusion weights not both zero).",
        "<b>Lazy secret checks.</b> API keys are validated only when a provider client is "
        "built, so tests and offline tools run without secrets.",
        "<b>Secret-free snapshots.</b> <font face='Mono'>Settings.snapshot()</font> produces a "
        "copy of the configuration with secrets removed, stored with every experiment report.",
    ]
)

H2("Data model")
P(
    "The schema is <font face='Mono'>documents → parent_chunks → chunks</font>, plus an "
    "<font face='Mono'>experiments</font> table. Three decisions shape everything downstream:"
)
numbered(
    [
        "<b>Chunks are offsets, not copies.</b> Each document stores its full normalised text "
        "(<font face='Mono'>documents.content</font>). Every parent and child chunk is exactly "
        "<font face='Mono'>content[start_char:end_char]</font>. Evaluation labels refer to these "
        "offsets.",
        "<b>Deterministic identifiers.</b> Document IDs are hashes of the normalised source "
        "path, so re-ingestion is an idempotent upsert and dataset references stay stable.",
        "<b>Fixed embedding dimension and model isolation.</b> The vector column has a fixed "
        "dimension, and every chunk records the embedding model that produced its vector. "
        "Retrieval only ranks vectors from the current model.",
    ]
)

H2("Engineering practice")
P(
    "The work was committed phase by phase, with each phase tested before its commit. At "
    "commit <font face='Mono'>726e2a7</font> the suite has 167 tests:"
)
bullets(
    [
        "unit tests for configuration, parsing, chunking invariants, the HTTP client's retry "
        "policy, structured output, citations and metric definitions",
        "integration tests against a dedicated PostgreSQL test database",
        "evaluation tests for dataset validation, span labelling and the evaluation runner",
    ]
)
P(
    "Metric functions are checked against hand-computed values. Integration tests run "
    "against real pgvector, not mocks."
)

# ------------------------------------------------------------------ 4 Corpus & ingestion
H1("Corpus and ingestion")
H2("Corpus")
P(
    "We use the English Kubernetes documentation from the <font face='Mono'>kubernetes/website</font> "
    "repository (CC BY 4.0), pinned to commit <font face='Mono'>5dd61e1</font> of 2026-09-23 "
    "(docs version v1.37). We take the <font face='Mono'>concepts/</font>, "
    "<font face='Mono'>tasks/</font> and <font face='Mono'>reference/glossary/</font> "
    "subtrees: 570 Markdown files, 551 of which contain prose (19 are empty section-index "
    "pages). A committed manifest records a SHA-256 for every file, so drift is detectable. "
    "The corpus was chosen because it supports factual, configuration, troubleshooting, "
    "comparison and multi-hop questions, and because its heavy cross-linking can be used to "
    "construct multi-hop pairs (§6.3)."
)

H2("Template resolution")
P(
    "The documentation is written for the Hugo site generator and is dense with template "
    "shortcodes. In <font face='Mono'>concepts/</font> and <font face='Mono'>tasks/</font> "
    "alone we counted 744 <font face='Mono'>glossary_tooltip</font>, 585 "
    "<font face='Mono'>note</font>, 354 <font face='Mono'>feature-state</font>, 277 "
    "<font face='Mono'>code_sample</font> and 187 <font face='Mono'>skew</font> shortcodes. "
    "Indexed raw, BM25 and the embedder would match template syntax instead of prose. A "
    "corpus-specific resolver approximates what a reader sees on kubernetes.io:"
)
bullets(
    [
        "tooltips become their text, or the glossary term's title",
        "glossary definitions are expanded from the glossary files, recursively, with a depth limit",
        "version shortcodes become the pinned version",
        "admonitions become an inline “Note:” label",
        "boilerplate includes are dropped",
    ]
)
P(
    "Seven rare shortcodes that the first pass did not handle were found by logging "
    "unknown names. Two of them (<font face='Mono'>link</font>, "
    "<font face='Mono'>relref</font>) were deleting visible text. After resolution, 0 of "
    "3,855 chunks contain template syntax. Each document stores its canonical kubernetes.io "
    "URL for citation."
)

H2("Structure-aware parent/child chunking")
P("The chunker only chooses boundaries; it never rewrites text. It works in three steps:")
numbered(
    [
        "The text is cut into <i>blocks</i>: paragraphs, with fenced code blocks kept whole and "
        "headings attached to the block that follows them.",
        "Oversized blocks are split into sentences, or lines for code, then into word windows, "
        "then into character slices. The size limits are therefore strict.",
        "<i>Parents</i> are consecutive heading sections merged up to "
        "<font face='Mono'>PARENT_CHUNK_SIZE</font>. <i>Children</i> are packed greedily inside "
        "each parent, with sentence-aligned overlap.",
    ]
)
P("Four invariants are tested on synthetic documents and on a sample of the real corpus:")
bullets(
    [
        "no non-whitespace character is lost",
        "every child lies inside its parent",
        "no chunk exceeds its limit",
        "the output is deterministic",
    ]
)
P(
    "Three parameters of this design were changed because measurements on the corpus showed "
    "a problem (Table 2)."
)
table(
    [
        ["Observation (measured on the corpus)", "Change", "Effect"],
        [
            "Parents were not allowed to span top-level (H2) sections. Kubernetes pages have many "
            "short H2 sections, so <b>50.1%</b> of children were identical to their parent, and "
            "parent-child retrieval would add no context",
            "Allow merging across H2 boundaries (never across PDF pages)",
            "Children identical to their parent: 50.1% → 8.0% (6.3% with final settings); median "
            "parent/child size ratio 1.0 → 3.8×; chunks under 30 tokens: 195 → 10",
        ],
        [
            "The source is hard-wrapped, and treating every newline as a sentence end produced "
            "chunks that began mid-sentence",
            "Split only after . ! ?, after a line-ending colon, or before list items",
            "Chunks starting in lowercase (a proxy for mid-sentence starts): 31 of 3,062 (1%), "
            "all of them legitimate on inspection",
        ],
        [
            "A word-count token estimate counted a 5,000-character base64 run as one token",
            "Cost long runs by length; slice oversized unsplittable runs",
            "Hard size limits; see the calibration in Table 3",
        ],
    ],
    [0.42 * TW, 0.26 * TW, 0.32 * TW],
    "Chunking changes driven by corpus measurements.",
)

H2("Calibrating the token estimate against the real tokenizer")
P(
    "Chunk sizes are configured in tokens, but the chunker deliberately uses a "
    "model-independent estimate, so that changing the embedding model does not move chunk "
    "boundaries. An audit script compares the estimate with bge-small's WordPiece tokenizer "
    "on every stored chunk and counts embedding inputs above the model's 512-token limit. The "
    "first estimator undercounted, and <b>123 of 3,062 embedding inputs (4%) were being "
    "silently truncated</b> by the embedder. The worst cases were hex container IDs in "
    "command output, wide Markdown tables and long CamelCase feature-gate names."
)
P("The calibrated rules are:")
bullets(
    [
        "word pieces cost one token per ~7 characters",
        "runs that mix letters and digits cost one per ~2 characters",
        "CamelCase identifiers are costed per component",
    ]
)
P("The child size was also reduced from 400 to 360 estimated tokens.")
table(
    [
        ["", "Naive estimator, 400-token children", "Calibrated estimator, 360-token children"],
        ["Estimate / actual, median", "0.874", "0.978"],
        ["Estimate / actual, 5th–95th percentile", "0.770 – 0.956", "0.862 – 1.086"],
        ["Worst underestimate", "0.43", "0.70"],
        ["Largest embedding input (real tokens)", "746", "491"],
        ["Embedding inputs truncated at 512 tokens", "123 of 3,062 (4.0%)", "0 of 3,855"],
    ],
    [0.40 * TW, 0.30 * TW, 0.30 * TW],
    "Token-estimate calibration against the bge-small "
    "tokenizer (whole corpus). Overestimates are rarer and harmless: only 4 chunks are "
    "overestimated by more than 1.5×. The extreme case (119×) is a base64 certificate, which "
    "WordPiece maps to a single [UNK] token.",
    align_right_from=1,
)
P(
    "The pipeline also flags any chunk whose embedding input still exceeds the model limit "
    "(<font face='Mono'>embedding_truncated</font> in chunk metadata), so truncation can no "
    "longer be silent."
)

H2("Resulting index")
table(
    [
        ["Category", "Documents", "Child chunks", "Avg child tokens"],
        ["concepts", "185", "2,099", "289"],
        ["tasks", "204", "1,593", "292"],
        ["reference (glossary)", "162", "163", "101"],
        ["Total", "551", "3,855", "–"],
    ],
    [0.34 * TW, 0.2 * TW, 0.22 * TW, 0.24 * TW],
    "Index statistics. There are 1,044 parents (median 1,126 tokens), and 0 offset mismatches "
    "between stored chunk text and content[start:end]. Embedding all 3,855 chunks took "
    "2.7 min on the M1 GPU (MPS) versus ~4 min on the CPU (43 vs 79 ms per chunk).",
    align_right_from=1,
    bold_last=True,
)
P(
    "Each chunk is embedded as <font face='Mono'>\"&lt;title&gt; &gt; &lt;section path&gt;\\n\\n&lt;text&gt;\"</font>. "
    "The stored text is unchanged; the prefix disambiguates chunks such as “Set the field "
    "to <font face='Mono'>true</font> to …”. Re-ingestion is idempotent. Each document "
    "stores a fingerprint of its content hash, the chunker version and configuration, the "
    "embedding model and the context flag, and only documents whose fingerprint changes are "
    "rebuilt."
)

# ------------------------------------------------------------------ 5 Baseline RAG
H1("Retrieval and generation pipelines")
H2("Dense retrieval")
P(
    "All retrievers implement one method, <font face='Mono'>retrieve(query, k, filters)</font>, "
    "and return a standard result (document ID, chunk ID, text, score, rank, offsets and "
    "metadata). The pgvector retriever handles three details that silently change results if "
    "ignored:"
)
bullets(
    [
        "<b><font face='Mono'>hnsw.ef_search</font> caps the number of results.</b> Its default "
        "of 40 would quietly truncate a top-50 candidate list and bias every later "
        "retrieve-then-rerank experiment. It is raised per query to max(40, 2k). Verified: "
        "k = 60 returns 60 results.",
        "<b>Filtered HNSW search can return fewer than k rows.</b> Filtered queries enable "
        "pgvector 0.8's iterative scans and re-sort the results.",
        "<b>Rows are restricted to the current embedding model</b>, so vectors from a "
        "previous model are never ranked against new query vectors.",
    ]
)
P(
    "Query latency is ~25 ms once the model is loaded. The first query after process start "
    "paid ~12 s of model loading, so the API preloads the embedder at startup."
)

H2("BM25")
P(
    "BM25 [4] is implemented in-process rather than with PostgreSQL full-text search, whose "
    "<font face='Mono'>ts_rank</font> lacks BM25's IDF and saturating term frequency; a "
    "\u201cBM25\u201d result built on it would be mislabelled. The inverted index uses "
    "Lucene-form IDF, ln(1 + (N − df + 0.5)/(df + 0.5)), which is never negative, and indexes "
    "the same <font face='Mono'>Title &gt; Section</font> input as dense retrieval. The index "
    "(3,855 chunks, 21,030 terms) builds in about 1.1 s. A cheap "
    "<font face='Mono'>count(*), max(created_at)</font> check before each query rebuilds it "
    "after re-ingestion, so it cannot go stale. Ties break by chunk ID, and queries take "
    "3.9 ms at p50."
)
P(
    "The analyzer lowercases and drops stopwords, and can additionally emit compound "
    "identifiers whole (<font face='Mono'>kube-apiserver</font>, "
    "<font face='Mono'>spec.replicas</font>), CamelCase components "
    "(<font face='Mono'>PodDisruptionBudget</font> → pod, disruption, budget) and light "
    "stems. Which options to enable was decided on the dev split (§8.1)."
)

H2("Hybrid fusion")
P(
    "The hybrid retriever takes 50 candidates from each retriever and fuses the two lists, "
    "by one of two methods:"
)
bullets(
    [
        "<b>weighted Reciprocal Rank Fusion [5]:</b> score = Σ<sub>r</sub> w<sub>r</sub> / "
        "(k + rank<sub>r</sub>); it uses ranks only, so the incompatible score scales (cosine "
        "similarity vs unbounded BM25) never interact",
        "<b>linear fusion:</b> score = Σ<sub>r</sub> w<sub>r</sub> · minmax<sub>r</sub>(score), "
        "with each list normalised over its own candidates",
    ]
)
P(
    "Weights, method and k come from configuration. Every fused result carries its rank and "
    "score in each component list, and the API returns them, so it is always visible why a "
    "chunk ranked where it did. Hybrid queries take 39 ms at p50. Like BM25, hybrid is "
    "registered as a strategy, so RAG with any of the three retrievers is the same pipeline "
    "with a different retrieval stage."
)

H2("Provider-neutral LLM layer")
P(
    "Pipelines depend on a single <font face='Mono'>complete(messages)</font> interface. One "
    "HTTP client covers every OpenAI-compatible server: local MLX, Ollama, vLLM, or hosted "
    "APIs. Its retry policy is explicit:"
)
bullets(
    [
        "timeouts, connection errors and 5xx responses are retried with jittered exponential "
        "backoff",
        "429 responses are retried, honouring <font face='Mono'>Retry-After</font>",
        "other 4xx responses fail immediately",
        "the number of attempts is bounded",
    ]
)
P(
    "Structured output does not rely on server-side JSON enforcement, which "
    "<font face='Mono'>mlx_lm.server</font> lacks:"
)
bullets(
    [
        "the schema is described in the prompt",
        "replies are parsed tolerantly (fences stripped; the first balanced object extracted)",
        "the validation error is fed back for one repair attempt",
        "otherwise a <font face='Mono'>StructuredOutputError</font> is raised; no default is "
        "substituted silently",
    ]
)
P(
    "Every call is recorded with its purpose, token counts and latency. Models without a "
    "configured price are reported as <i>unpriced</i> rather than counted as free, so a paid "
    "model with a missing price cannot make a strategy look cheap."
)

H2("Answer generation and citations")
P(
    "The baseline embeds the query, retrieves the top-k chunks, and makes exactly one LLM "
    "call over numbered sources. The model must cite inline (<font face='Mono'>[n]</font>) and "
    "use an exact abstention phrase when the sources lack the answer, so abstention is "
    "detected deterministically, without a judge. Empty retrieval abstains without calling "
    "the LLM. Citation numbers are mapped back to chunk, section, page, URL and character "
    "span; numbers that match no source are reported as <i>invalid citations</i>."
)
P(
    "With the 3B model, the first prompt (all rules in the system message) produced valid "
    "citations in only 1 of 8 answers on a sanity probe. Adding a format example raised this "
    "to 2 of 8. Restating the citation rule after the sources raised it to <b>8 of 8</b>. "
    "Small models tend to lose system-prompt instructions behind ~1.5k tokens of context. "
    "This is an 8-question probe used to choose a reasonable baseline prompt, not a benchmark "
    "result."
)
table(
    [
        ["Measurement (dev machine)", "Value"],
        [
            "Structured grading probe: valid JSON matching the schema",
            "6 / 6; ~1.1 s per call when warm",
        ],
        [
            "Grading judgment: a passage stating the answer (“graceful within 30 seconds”) "
            "marked contains_answer",
            "false on 3 / 3 repeats",
        ],
        ["Baseline answer generation, top-5 contexts (~1.6–1.8k prompt tokens)", "15–19 s"],
        [
            "End to end through Docker: retrieval / generation",
            "0.14 s / 19.1 s; 1,625 input and 17 output tokens; $0.00",
        ],
        ["Unanswerable probe (“What is the capital of France?”)", "abstained correctly"],
    ],
    [0.64 * TW, 0.36 * TW],
    "Observed behaviour of the local model. These are sanity checks, not benchmarks.",
)
P(
    "The grading failure in Table 5 is important for later phases. The 3B model formats "
    "output reliably but judges relevance poorly. Following CRAG's own design of a small "
    "dedicated evaluator [2], relevance grading will use a cross-encoder (or monoT5 [11]), "
    "and claim verification will use MiniCheck [10]. The LLM grader remains a switchable "
    "comparison."
)

# ------------------------------------------------------------------ 6 Dataset
H1("The evaluation dataset")
P(
    "The dataset, <font face='Mono'>kubernetes_v1</font>, contains 316 items and is "
    "<b>synthetic</b>. Questions, answers and evidence selections were produced by the local "
    "3B model and filtered by deterministic validators. It is <b>not human-annotated</b>, and "
    "no item is labelled as human-reviewed."
)

H2("Design principles")
H3("Span-anchored relevance")
P(
    "Each item's evidence is a list of exact spans <font face='Mono'>(document_id, "
    "start_char, end_char, quote)</font> into <font face='Mono'>documents.content</font>. At "
    "evaluation time a chunk is relevant to a span if it covers at least 50% of the span's "
    "characters, computed against the index being evaluated. The stored "
    "<font face='Mono'>relevant_chunk_ids</font> are only a convenience snapshot. As a "
    "result, the same dataset can evaluate different chunk sizes, overlaps or parent-child "
    "settings. A span that no current chunk covers is reported and counted as a miss."
)
H3("Held-out test split")
P(
    "Items are split 30% dev and 70% test, stratified by question type with a fixed seed "
    "(97 dev, 219 test). Thresholds and weights may be tuned only on dev; reported results "
    "use test. Tuning on the reported questions would overstate every tuned technique. With "
    "about 100 dev items, k-fold cross-validation over the dev split is available for "
    "threshold tuning."
)
H3("Provenance")
P("Every item records:")
bullets(
    [
        "the generator model",
        "the prompt version",
        "the list of validators it passed",
    ]
)
P(
    "The companion metadata file records the corpus commit, chunker fingerprint, seeds, "
    "per-type counts, every generation run, and per-type rejection counts for each validator."
)

H2("Generation procedure")
figure(
    pipeline_drawing(),
    "Per-item generation loop. Passages are shown to the model as numbered sentences, "
    "using the chunker's own sentence rules, and the model returns sentence numbers, not "
    "quotes. Evidence spans are then mapped deterministically, so they are always exact.",
)
P(
    "Sampling is seeded. Each chunk is used at most once, and each document yields at most "
    "three items. Chunks that are mostly code, or more than 5% HTML markup, are skipped. The "
    "items that pass are joined with dataset-wide checks:"
)
bullets(
    [
        "the question is well formed and self-contained: it may not say “the passage”, "
        "“the example” or “in this case”",
        "the question is not a near-duplicate of an earlier one (content-word Jaccard below 0.8)",
        "the answer is non-empty and <i>supported</i>: at least 50% of the stemmed content words "
        "it adds beyond the question must appear in the evidence",
        "the evidence maps to at least one indexed chunk",
    ]
)

H2("Question types")
P(
    "Table 6 summarises how each of the nine types is built and which type-specific "
    "validators it must pass. The notes below explain the less obvious ones."
)
table(
    [
        ["Type", "Construction", "Type-specific validation"],
        [
            "simple_factual",
            "One chunk → direct factual question",
            "No 8-gram copied from the evidence",
        ],
        [
            "semantic",
            "One chunk → question in different words",
            "At most 40% of the question's stemmed, non-generic content words appear in the evidence",
        ],
        [
            "keyword_heavy",
            "One chunk plus a required identifier (flag, dotted field, camelCase)",
            "The identifier appears in both the question and the evidence",
        ],
        [
            "numerical",
            "Chunk containing number + unit",
            "The answer's number appears in the evidence",
        ],
        [
            "multi_hop",
            "Chunk A, plus the best-matching chunk of a page that A links to",
            "Answer uses ≥ 2 words unique to each source; neither evidence is only a link; prose "
            "only",
        ],
        [
            "comparison",
            "Intro chunks of the two most similar concept pages in one section",
            "As multi-hop",
        ],
        [
            "ambiguous",
            "Specific question → vague rewrite (“What happens when it is deleted?”)",
            "The rewrite names neither the page title nor the capitalised subjects; "
            "the specific form is kept",
        ],
        [
            "adversarial",
            "LLM states a true numeric fact; code changes the number",
            "The fact is ≥ 70% supported by its evidence and shares its number; the true and false "
            "values are recorded",
        ],
        [
            "unanswerable",
            "(a) invented Kubernetes feature; (b) question about another technology",
            "(a) the invented term appears nowhere in the corpus (SQL); (b) the topic keyword appears "
            "nowhere in the corpus, and there are no Kubernetes terms",
        ],
    ],
    [0.15 * TW, 0.40 * TW, 0.45 * TW],
    "Construction and validation per question type.",
)
H3("Multi-hop pairs follow real links")
P(
    "Of the eligible concept and task chunks, 1,244 link to another document in the corpus. "
    "For a chunk A containing a link, we take the text around the link and use it as a dense "
    "query <i>restricted to the linked page</i> (a metadata filter on its URL). The best "
    "match is passage B. The pair is therefore connected by the documentation's own "
    "structure rather than chosen at random."
)
H3("Comparisons pair similar pages")
P(
    "A concept page's introduction is paired with the most similar other introduction in the "
    "same documentation section, for example ReplicationController with ReplicaSet."
)
H3("False premises are built by code")
P(
    "The model only states a true fact containing a number. Code then replaces that number "
    "with a different value that does not occur in the evidence. The question takes the "
    "form “I read that &lt;false statement&gt;. Why is that?”, and the ground truth "
    "is the verified true statement."
)
H3("Unanswerable items are verified against the corpus")
P(
    "A fake feature or an out-of-domain topic is accepted only if a SQL search of all "
    "document text finds no occurrence of its key term. Of 20 candidate technologies, only 9 "
    "were absent from the corpus; Nginx and Redis, for example, appear in Kubernetes examples."
)

H2("How pilot reviews changed the generator")
P(
    "Three pilots of three items per type were read item by item before the full run. Each "
    "review exposed failures that the validators of the time did not catch (Table 7). These "
    "reviews are the main reason the final validators exist."
)
table(
    [
        ["Pilot", "Finding", "Design change"],
        [
            "1",
            "<b>All 3 adversarial items were wrong.</b> The model wrote “false” premises "
            "that were true, or restated the passage",
            "False premises constructed by code from a verified true fact",
        ],
        [
            "1",
            "Comparisons of unrelated pages (metrics vs version emulation)",
            "Pair each page with its most similar intro in the same section",
        ],
        [
            "1",
            "A “multi-hop” item answerable from A alone; B's evidence was “See [link] for more”",
            "Uses-both-sources check; link-only evidence rejected",
        ],
        [
            "1",
            "Semantic questions copied the passage, but Jaccard stayed low because quotes "
            "were long",
            "Containment instead of Jaccard",
        ],
        [
            "2",
            "<b>A fabricated fact</b>: “the default Pod grace period is 30 seconds” was "
            "attached to evidence about CoreDNS caching for 30 seconds. The model had copied the "
            "prompt's example sentence",
            "Example removed; answer-support check added",
        ],
        [
            "2",
            "Multi-hop: 0 of 3 accepted; 14 of 18 attempts failed because quotes were not copied "
            "verbatim",
            "Numbered sentences and sentence-ID evidence instead of quotes",
        ],
        [
            "2",
            "Generic words (“Kubernetes cluster”) diluted the paraphrase ratio; "
            "“require” did not match “requires”",
            "Light stemming; generic words excluded; threshold 0.4",
        ],
        [
            "3",
            "Large YAML blocks, counted as one “sentence”, satisfied the uses-both check "
            "by word count",
            "Prose-only passages for two-source questions",
        ],
        ["3", "Evidence that was raw HTML table markup", "Skip chunks with > 5% HTML"],
        [
            "Full run",
            "At temperature 0, one prompt per topic gives one question: 81 "
            "near-duplicates for 9 topics",
            "Topic × angle prompts for out-of-domain questions",
        ],
        [
            "Full run",
            "A fake <font face='Mono'>--force-delete</font> flag sits next to the real "
            "<font face='Mono'>--force</font>. “Not documented; use --force” is a correct "
            "answer, but the label demanded a flat abstention",
            "Ground truth for nonexistent features changed to accept a correction",
        ],
    ],
    [0.09 * TW, 0.53 * TW, 0.38 * TW],
    "Failures found by reading generated items, and the resulting design changes.",
)

H2("Resulting dataset")
P(
    "The full run (seed 42) produced 275 items. A top-up run (seed 43, extend mode) marked "
    "all existing chunks and questions as used and added 41 items to the types that had "
    "fallen short."
)
table(
    [
        ["Type", "Items", "Test", "Dev", "Accepted / attempts", "Rate", "Most common rejection"],
        ["simple_factual", "60", "42", "18", "60 / 100", "60%", "not self-contained (15)"],
        ["semantic", "45", "31", "14", "45 / 423", "11%", "not paraphrased (304)"],
        ["keyword_heavy", "45", "31", "14", "45 / 251", "18%", "identifier missing (150)"],
        ["numerical", "30", "21", "9", "30 / 59", "51%", "number not in evidence (12)"],
        ["multi_hop", "37", "26", "11", "37 / 432", "9%", "does not use both sources (262)"],
        ["comparison", "19*", "13", "6", "19 / 336", "6%", "does not use both sources (158)"],
        ["ambiguous", "25", "17", "8", "25 / 79", "32%", "rewrite names the subject (25)"],
        ["adversarial", "25", "17", "8", "25 / 103", "24%", "stated fact had no number (40)"],
        ["unanswerable", "30", "21", "9", "30 / 118", "25%", "near-duplicate (81)"],
        ["Total", "316", "219", "97", "316 / 1,901", "17%", ""],
    ],
    [0.16 * TW, 0.08 * TW, 0.08 * TW, 0.07 * TW, 0.17 * TW, 0.08 * TW, 0.36 * TW],
    "Dataset composition and generation statistics (version c6009ef85e53). *Comparison "
    "fell short of its target of 30; the shortfall is reported rather than filled by relaxing "
    "validation. The items cover 208 of the 551 documents.",
    align_right_from=1,
    bold_last=True,
    left_cols=(6,),
)

H2("Examples")
P("One real item per type, taken verbatim from the dataset (answers abridged where marked with …):")
EX = [
    (
        "simple_factual",
        "What role is assigned by default to an extension apiserver to allow it "
        "to retrieve a configmap?",
        "The default role named "
        "`extension-apiserver-authentication-reader` in the `kube-system` namespace can be "
        "assigned.",
    ),
    (
        "semantic",
        "What are the common prefixes used for different subsystems in the sysctl interface?",
        "…'kernel.' for kernel parameters, 'net.' for networking parameters, and "
        "'vm.' for virtual memory…",
    ),
    (
        "keyword_heavy",
        "What operators are not available for podAffinity?",
        "The `Gt` and `Lt` operators are not available for podAffinity.",
    ),
    (
        "numerical",
        "How many seconds does the container need to be alive before the liveness "
        "probe starts failing?",
        "10",
    ),
    (
        "multi_hop",
        "What actions are recommended for cleaning up a ConfigMap after it has been "
        "changed, and how do these actions relate to the process of recreating existing Pods?",
        "Recreating existing Pods that maintain a mount point to the deleted ConfigMap, and "
        "removing the `kustomization.yaml` file…",
    ),
    (
        "comparison",
        "How do the methods for protecting secret data differ between Linux and Windows nodes?",
        "On Linux nodes, secret data is stored in memory-backed volumes using a "
        "tmpfs filesystem, while on Windows nodes, secret data is written out in clear text…",
    ),
    (
        "ambiguous",
        "What command do you use to see the rollout history of it?",
        "kubectl rollout history deployment/nginx-deployment",
    ),
    (
        "adversarial",
        "I read that the Kubernetes project recommends rotating the KEK at least "
        "every 180 days to avoid issues related to the number of writes performed with the DEK. "
        "Why is that?",
        "That premise is incorrect: … at least every 90 days … (true value 90, false value 180)",
    ),
    (
        "unanswerable",
        "How can I use the --auto-heal-pods flag to automatically heal pods "
        "during the restoration of an etcd cluster?",
        "The corpus does not document "
        "`--auto-heal-pods`. A correct response says so (and may point to a related documented "
        "feature)…",
    ),
]


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


table(
    [["Type", "Question", "Ground truth"]] + [[t, esc(q), esc(a)] for t, q, a in EX],
    [0.15 * TW, 0.47 * TW, 0.38 * TW],
    "Example items (dataset kubernetes_v1).",
)

H2("Quality audit and known weaknesses")
P(
    "After each run, a random sample was read in full during development by an automated "
    "LLM-based reviewer. This is a sanity check, not human annotation. Of 33 sampled items, "
    "25 were acceptable. That count includes one unanswerable item whose label was wrong and "
    "has since been corrected. The other 8 were flawed:"
)
bullets(
    [
        "3 multi-hop items with vague answers, or answers that only partly use the second source",
        "1 comparison with a fabricated contrast (“scheduler vs X.509 certificates”)",
        "1 numerical item that took “8” from “8 hours late” and dropped the unit",
        "1 ambiguous item whose answer misses the question",
        "2 simple items with garbled grammar",
    ]
)
P(
    "One further error found while preparing this report shows the limit of word-overlap "
    "support checks. An adversarial item attributes <font face='Mono'>progressDeadlineSeconds"
    "</font> to <font face='Mono'>.spec.strategy.rollingUpdate</font>; its value, 600, is "
    "correct, but the field actually lives at <font face='Mono'>.spec.progressDeadlineSeconds"
    "</font>. The sample is too small to estimate per-type error rates. A script exports a "
    "stratified sample for human review; only verdicts entered by a person should be "
    "reported as human-validated accuracy. The known weaknesses are:"
)
bullets(
    [
        "<b>Lexical bias.</b> Questions are written from the passage they are evaluated against, "
        "and so share its vocabulary. This favours lexical retrieval, a known property of "
        "passage-derived benchmarks [16]. The semantic type only partly counters it.",
        "<b>Same model family.</b> The generator is also the default answer model. This does not "
        "affect retrieval metrics, but generation metrics must account for it [14].",
        "<b>Uneven label quality.</b> Multi-hop and comparison items have correct evidence spans "
        "more often than good reference answers: their retrieval labels are more reliable than "
        "their answer labels.",
        "<b>Example-specific questions</b> (“How many replicas are initially scheduled for "
        "the Deployment 'nginx'?”) are answerable but less natural than real queries.",
        "<b>Coverage and size.</b> 208 of 551 documents; 13–42 test items per type.",
    ]
)

# ------------------------------------------------------------------ 7 Metrics
H1("Retrieval metrics")
P(
    "All metrics use binary relevance over a single ranked list per query, computed once at "
    "the deepest K. Let <i>R</i> be the set of relevant chunks and <i>top-K</i> the first "
    "K retrieved chunks:"
)
table(
    [
        ["Metric", "Definition"],
        ["Recall@K", "|R ∩ top-K| / |R|"],
        ["Precision@K", "|R ∩ top-K| / K"],
        ["Hit@K", "1 if R ∩ top-K is non-empty, else 0"],
        [
            "MRR",
            "1 / rank of the first relevant chunk (0 if none within the deepest K), averaged "
            "over queries",
        ],
        [
            "NDCG@K [6]",
            "DCG@K / IDCG@K, with DCG = Σ rel<sub>i</sub> / log<sub>2</sub>(i+1) over "
            "binary gains; duplicate hits earn no extra gain; IDCG places min(|R|, K) relevant items "
            "first",
        ],
        [
            "<b>Evidence Recall@K</b>",
            "Fraction of the item's evidence spans for which at least "
            "one covering chunk is in top-K",
        ],
        [
            "Doc Recall@K",
            "Fraction of relevant documents among the first K distinct documents in the ranking",
        ],
    ],
    [0.22 * TW, 0.78 * TW],
    "Metric definitions. All are unit-tested against hand-computed values.",
)
P(
    "<b>Why Evidence Recall is the primary recall metric.</b> With overlapping chunks, one "
    "evidence span can be covered by two chunks. A system that retrieves just one of them has "
    "all the evidence but only 50% chunk recall. Evidence Recall instead asks whether each "
    "required piece of evidence was found; for a multi-hop item, whether <i>both</i> halves "
    "were retrieved. Unanswerable items have no evidence, so they are excluded from retrieval "
    "metrics and reported as excluded; generation evaluation scores them on abstention."
)
P(
    "<b>Uncertainty.</b> Per-query metrics are averaged. We report 95% percentile-bootstrap "
    "intervals [13] (2,000 resamples of queries, fixed seed) for the headline metrics. With "
    "198 queries overall, and only 13–42 per type, point estimates alone would overstate how "
    "precise these results are."
)
P(
    "<b>Paired comparisons.</b> All systems answer the same questions, so a difference "
    "between two systems is bootstrapped per query: the per-query differences are "
    "resampled. This is much tighter than comparing two independent intervals. We also "
    "report how many queries each system wins, loses and ties."
)
P(
    "<b>Selection protocol.</b> Analyzer options, BM25 parameters and the fusion "
    "configuration were chosen on the dev split only (88 answerable items). For fusion, the "
    "selection rule was fixed before tuning (§8.1). The test split was evaluated once per "
    "system, with the final configuration. Every dev run is kept as a committed report."
)

# ------------------------------------------------------------------ 8 Results
H1("Retrieval results")
P(
    "All runs use dataset version <font face='Mono'>c6009ef85e53</font>. On the test split, "
    "198 answerable items are evaluated (21 unanswerable items are excluded), with 0 "
    "unmapped evidence spans, and reruns reproduce every value exactly. Dense retrieval uses "
    "bge-small-en-v1.5 with the <font face='Mono'>Title &gt; Section</font> prefix over a "
    "cosine HNSW index."
)

H2("Configuration choices on the dev split")
H3("BM25 analyzer and parameters")
table(
    [
        ["Analyzer (k1 = 1.2, b = 0.75)", "EvR@5", "EvR@10", "MRR", "NDCG@10"],
        ["<i>dense retrieval (reference)</i>", "0.748", "0.847", "0.617", "0.594"],
        ["words", "0.818", "0.871", "0.780", "0.729"],
        ["words + stem", "0.820", "0.849", "0.782", "0.731"],
        ["<b>words + compound (chosen)</b>", "0.827", "0.871", "0.788", "0.737"],
        ["words + compound + camel", "0.817", "0.869", "0.752", "0.712"],
        ["words + compound + camel + stem", "0.820", "0.850", "0.777", "0.726"],
    ],
    [0.44 * TW, 0.14 * TW, 0.14 * TW, 0.14 * TW, 0.14 * TW],
    "BM25 analyzer ablation on the dev split (88 items).",
    align_right_from=1,
)
P(
    "No analyzer variant is significantly better than plain words: every paired 95% "
    "interval includes zero. Compound identifiers changed only 6 of 88 queries (4 better, "
    "2 worse). CamelCase splitting leaned negative on MRR (−0.027, CI [−0.061, +0.005]; 12 "
    "queries worse, 4 better), because splitting identifiers into common words such as "
    "<i>pod</i> dilutes their specificity. We chose words + compound for its best point "
    "estimates on every metric. A k1 × b grid ({0.9, 1.2, 1.5} × {0.5, 0.75, 0.9}) was flat "
    "(EvR@5 0.827–0.845, a spread of one or two queries), so the standard k1 = 1.2, b = 0.75 "
    "were kept rather than fitting noise."
)
H3("Fusion configuration")
P(
    "The two retrievers are complementary on dev. Their combined top 5 covers 0.911 of the "
    "required evidence spans, against 0.827 for BM25 and 0.748 for dense retrieval: 15 spans "
    "are found only by dense retrieval and 30 only by BM25. The selection rule, fixed before "
    "tuning, keeps the textbook default (RRF, equal weights, k = 60) unless another "
    "configuration improves dev Evidence Recall@5 by more than one query's worth "
    "(1/88 ≈ 0.011), with ties broken on MRR."
)
table(
    [
        ["Configuration (dense weight w; BM25 = 1 − w)", "EvR@5", "EvR@10", "MRR", "NDCG@10"],
        ["RRF, w = 0.5, k = 60 (default)", "0.826", "0.917", "0.780", "0.730"],
        [
            "RRF, k = 60, w = 0.3 / 0.4 / 0.6 / 0.7",
            "0.803–0.826",
            "0.891–0.917",
            "0.703–0.805",
            "0.672–0.744",
        ],
        [
            "RRF, k = 10, w = 0.3 / 0.4 / 0.5 / 0.6 / 0.7",
            "0.825–0.860",
            "0.874–0.933",
            "0.695–0.815",
            "0.660–0.761",
        ],
        [
            "linear, w = 0.3 / 0.4 / 0.6 / 0.7",
            "0.830–0.854",
            "0.894–0.917",
            "0.731–0.819",
            "0.695–0.770",
        ],
        ["<b>linear, w = 0.5 (chosen)</b>", "0.866", "0.917", "0.800", "0.756"],
    ],
    [0.44 * TW, 0.14 * TW, 0.14 * TW, 0.14 * TW, 0.14 * TW],
    "Fusion grid on the dev split (ranges summarise rows; all 15 runs are committed).",
    align_right_from=1,
)
P("Four observations follow from this grid:")
bullets(
    [
        "<b>The textbook default is no better than BM25 alone</b> (paired difference +0.001). "
        "Equal-weight RRF at k = 60 lets the weaker dense list dilute BM25.",
        "<b>RRF with k = 10 beats k = 60 at every weight</b>, a consistent pattern rather than "
        "one lucky cell.",
        "<b>Weighting dense above 0.5 hurts throughout.</b>",
        "<b>The winner, linear fusion with w = 0.5, gains +0.040 Evidence Recall@5</b> (paired "
        "CI [+0.011, +0.085]). The evidence is thin: only 4 of 88 queries changed, all for "
        "the better. Picking the best of 15 configurations invites a winner's-curse effect, "
        "so the test result is the number to trust.",
    ]
)

H2("Test-split comparison")
table(
    [
        ["Metric", "Dense", "BM25", "Hybrid", "Hybrid − dense (95% CI)", "Hybrid − BM25 (95% CI)"],
        [
            "Evidence Recall@5",
            "0.701",
            "0.791",
            "<b>0.802</b>",
            "<b>+0.101</b> [+0.056, +0.146]",
            "+0.011 [−0.024, +0.048]",
        ],
        [
            "Evidence Recall@10",
            "0.761",
            "0.844",
            "<b>0.859</b>",
            "<b>+0.098</b> [+0.056, +0.140]",
            "+0.015 [−0.018, +0.048]",
        ],
        [
            "MRR",
            "0.624",
            "<b>0.725</b>",
            "0.723",
            "<b>+0.099</b> [+0.059, +0.139]",
            "−0.002 [−0.034, +0.029]",
        ],
        [
            "NDCG@10",
            "0.595",
            "<b>0.715</b>",
            "0.714",
            "<b>+0.119</b> [+0.085, +0.150]",
            "−0.001 [−0.025, +0.024]",
        ],
        ["EvR@5 95% CI", "0.641–0.759", "0.740–0.842", "0.750–0.852", "", ""],
        ["Latency p50 / p95", "24 / 32 ms", "3.9 / 5.5 ms", "39 / 44 ms", "", ""],
    ],
    [0.20 * TW, 0.115 * TW, 0.115 * TW, 0.11 * TW, 0.23 * TW, 0.23 * TW],
    "Test-split comparison (n = 198). Differences are paired bootstrap estimates. BM25 − "
    "dense: Evidence Recall@5 +0.090 [+0.034, +0.143], MRR +0.101 [+0.046, +0.155].",
    align_right_from=1,
)
P(
    "BM25 is significantly better than dense retrieval on every headline metric. It wins 36 "
    "queries at Evidence Recall@5 and loses 13. Hybrid fusion is significantly better than "
    "dense retrieval, but not significantly different from BM25 overall (12 queries better, "
    "8 worse, 178 tied). Hybrid has the highest Evidence Recall@20 of the three (0.909), at "
    "about ten times BM25's latency."
)
figure(
    results_chart(),
    "Evidence Recall@5 by question type for the three retrievers (test split). Values are "
    "also in Table 14.",
)
table(
    [
        ["Type", "n", "Dense", "BM25", "Hybrid", "Hybrid @20"],
        ["semantic", "31", "0.672", "0.710", "<b>0.806</b>", "0.903"],
        ["multi_hop", "26", "0.641", "0.689", "<b>0.760</b>", "0.881"],
        ["comparison", "13", "0.449", "0.368", "<b>0.468</b>", "0.924"],
        ["keyword_heavy", "31", "0.871", "<b>0.935</b>", "<b>0.935</b>", "1.000"],
        ["adversarial", "17", "0.824", "<b>1.000</b>", "<b>1.000</b>", "1.000"],
        ["simple_factual", "42", "0.821", "<b>0.905</b>", "0.881", "0.929"],
        ["numerical", "21", "0.571", "<b>0.952</b>", "0.809", "1.000"],
        ["ambiguous", "17", "0.471", "0.471", "0.471", "0.529"],
        ["All", "198", "0.701", "0.791", "0.802", "0.909"],
    ],
    [0.24 * TW, 0.1 * TW, 0.165 * TW, 0.165 * TW, 0.165 * TW, 0.165 * TW],
    "Evidence Recall@5 by question type; best per row in bold.",
    align_right_from=1,
    bold_last=True,
)

H2("Discussion")
P("Per-type samples are small (13–42 items), so these are directions, not conclusions:")
bullets(
    [
        "<b>Part of BM25's lead is an artefact of the dataset.</b> Adversarial questions are "
        "generated from a fact statement written in the passage's own words and near-quote "
        "their evidence; BM25 retrieves all of them (1.00). Numerical questions share exact "
        "numbers and units with their evidence (BM25 +0.38 over dense). On semantic questions, "
        "which were filtered for low word overlap, BM25's lead shrinks to +0.04. That is a "
        "fairer indication of the difference on paraphrased queries (§6.7).",
        "<b>Hybrid gains where questions are least lexical</b> (semantic +0.10, multi-hop "
        "+0.07, comparison +0.10 over BM25), and <b>loses where exact matching decides</b> "
        "(numerical −0.14, simple factual −0.02). It is best or tied-best on 6 of 8 types, "
        "which makes it the most robust of the three. We expect real user queries to be less "
        "lexically aligned with the documentation than these passage-derived questions, but "
        "that is an expectation, not a measurement.",
        "<b>Ambiguous questions stay at 0.47 for every method.</b> Fusion combines retrievers, "
        "but neither can find evidence for a question that lacks the key terms. Recall "
        "barely grows with depth (0.53 at 20), so this is a query problem that only query "
        "rewriting can address.",
        "<b>Much evidence is retrieved but ranked low.</b> Hybrid's Evidence Recall rises "
        "from 0.80 at 5 to 0.91 at 20, and for comparison questions from 0.47 to 0.92. A "
        "cross-encoder reranker over the hybrid candidate list is the matching next step.",
        "<b>Multi-hop has high MRR but lower Evidence Recall.</b> Dense retrieval's MRR on "
        "multi-hop is 0.78, but its Evidence Recall@5 is only 0.64: one half of the evidence "
        "is found early and the other half often is not.",
    ]
)

# ------------------------------------------------------------------ 9 Limitations
H1("Limitations")
bullets(
    [
        "<b>Synthetic data without human validation.</b> The results measure retrieval against "
        "machine-generated, machine-validated labels. The sample audit estimates that a minority "
        "of items are flawed, but it is not a substitute for human review.",
        "<b>A single corpus and a single embedding model.</b> Conclusions about techniques may "
        "not transfer to other domains; larger embedders (e.g. 768-d) are an open experiment.",
        "<b>A small generator.</b> A 3B model limits both question quality and, in later phases, "
        "grading and judging quality. All generation metrics will be reported together with the "
        "model that produced them. The same experiments can be re-run with a stronger model by "
        "changing configuration only.",
        "<b>Code samples are not inlined.</b> <font face='Mono'>code_sample</font> shortcodes "
        "reference manifests outside the fetched subtree, so answers that exist only inside "
        "example YAML are absent from the corpus.",
        "<b>Lexical bias shapes the retrieval comparison.</b> Passage-derived questions favour "
        "BM25, so the overall ranking of methods on this set may not transfer to real queries. "
        "The per-type results and the semantic subset are the more transferable signal.",
        "<b>Configuration choices rest on a small dev split.</b> With 88 dev items, several "
        "choices were decided by a handful of queries. Selection rules were fixed in advance "
        "and the test split was evaluated once, but a winner's-curse effect remains possible.",
        "<b>Only retrieval is evaluated so far.</b> Faithfulness, answer relevance, citation "
        "correctness, end-to-end latency and cost are not yet measured over the dataset.",
    ]
)

# ------------------------------------------------------------------ 10 Future work
H1("Roadmap")
P(
    "The next phases add techniques one at a time and evaluate each on the same test split "
    "and metrics:"
)
bullets(
    [
        "cross-encoder reranking of the hybrid candidate list (Phase 7)",
        "parent-child context expansion (Phase 8)",
        "query analysis and rewriting (Phase 9)",
        "corrective retrieval, with a small-model relevance grader (Phase 10)",
        "a Self-RAG-inspired loop with bounded iterations (Phase 11)",
        "claim-level verification (Phase 12)",
    ]
)
P(
    "Tracing and an experiment runner that records quality, latency, token usage and cost for "
    "each configuration follow. The generation evaluation will report deterministic signals "
    "(abstention, invalid citations) separately from judge-based scores, and will validate "
    "the judge against human labels before relying on it."
)

# ------------------------------------------------------------------ 11 Reproducibility
H1("Reproducibility")
CODE(
    "docker compose up -d postgres                        # PostgreSQL 16 + pgvector 0.8.6, host port 5434\n"
    'pip install -e ".[dev,local-llm]"\n'
    "mlx_lm.server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8080\n"
    "python scripts/fetch_corpus.py                       # kubernetes/website @ 5dd61e1\n"
    "EMBEDDING_DEVICE=mps python scripts/ingest.py --corpus kubernetes\n"
    "python scripts/audit_chunk_tokens.py                 # Table 3\n"
    "python scripts/create_eval_dataset.py                # seed 42; top-up: --extend --seed 43\n"
    "python scripts/run_retrieval_eval.py --retriever dense --split test\n"
    "python scripts/run_retrieval_eval.py --retriever bm25 --split test\n"
    "python scripts/run_retrieval_eval.py --retriever hybrid --split test  # Tables 13-14\n"
    "python scripts/compare_retrieval_runs.py data/experiments/<run-a> data/experiments/<run-b>\n"
    "BM25_STEMMING=true python scripts/run_retrieval_eval.py --retriever bm25 --split dev \\\n"
    "    --out-dir data/experiments/ablations --label words-compound-stem   # dev ablations"
)
table(
    [
        ["Item", "Value"],
        ["Hardware", "Apple M1, 8 GB RAM (development laptop)"],
        [
            "Software",
            "Python 3.12.6; FastAPI 0.141.1; SQLAlchemy 2.0.54; Pydantic 2.13.5; "
            "sentence-transformers 6.1.0; torch 2.14.0; mlx-lm 0.31.3; PostgreSQL 16.15; pgvector 0.8.6",
        ],
        ["Models", "BAAI/bge-small-en-v1.5; mlx-community/Qwen2.5-3B-Instruct-4bit"],
        [
            "Corpus",
            "kubernetes/website 5dd61e17b222c0220281657b843a3e15c86b3cfe (docs v1.37), CC BY 4.0",
        ],
        [
            "Dataset",
            "data/eval/kubernetes_v1.jsonl, version c6009ef85e53 (316 items; seeds 42, 43)",
        ],
        [
            "Code",
            "Commits f2854e3 (Phase 1), 24659a9 (Phase 2), ab6ad5d (Phase 3), f46239c (Phase 4), "
            "9f49caf (Phase 5), 726e2a7 (Phase 6)",
        ],
        [
            "Key defaults",
            "CHILD_CHUNK_SIZE 360, CHILD_CHUNK_OVERLAP 60, PARENT_CHUNK_SIZE 1600; "
            "cosine; TOP_K 10; BM25 words+compound, k1 1.2, b 0.75; fusion linear 0.5/0.5 over 50 "
            "candidates per retriever; LLM temperature 0; 167 tests passing",
        ],
    ],
    [0.18 * TW, 0.82 * TW],
    "Environment and artefacts.",
)
P(
    "Documentation of the Kubernetes project is © The Kubernetes Authors and is used under "
    "CC BY 4.0.",
    caption,
)

# ------------------------------------------------------------------ References
H1("References")
refs = [
    "Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for "
    "Knowledge-Intensive NLP Tasks. <i>NeurIPS</i>.",
    "Yan, S.-Q., Gu, J.-C., Zhu, Y., and Ling, Z.-H. (2024). Corrective Retrieval Augmented "
    "Generation. arXiv:2401.15884.",
    "Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. (2024). Self-RAG: Learning to "
    "Retrieve, Generate, and Critique through Self-Reflection. <i>ICLR</i>. arXiv:2310.11511.",
    "Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and "
    "Beyond. <i>Foundations and Trends in Information Retrieval</i>, 3(4).",
    "Cormack, G. V., Clarke, C. L. A., and Büttcher, S. (2009). Reciprocal Rank Fusion "
    "Outperforms Condorcet and Individual Rank Learning Methods. <i>SIGIR</i>.",
    "Järvelin, K. and Kekäläinen, J. (2002). Cumulated Gain-Based Evaluation of IR "
    "Techniques. <i>ACM Transactions on Information Systems</i>, 20(4).",
    "Malkov, Y. A. and Yashunin, D. A. (2020). Efficient and Robust Approximate Nearest "
    "Neighbor Search Using Hierarchical Navigable Small World Graphs. <i>IEEE TPAMI</i>, 42(4).",
    "Xiao, S., Liu, Z., Zhang, P., and Muennighoff, N. (2023). C-Pack: Packaged Resources To "
    "Advance General Chinese Embedding. arXiv:2309.07597.",
    "Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). RAGAS: Automated "
    "Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.",
    "Tang, L., Laban, P., and Durrett, G. (2024). MiniCheck: Efficient Fact-Checking of LLMs "
    "on Grounding Documents. <i>EMNLP</i>. arXiv:2404.10774.",
    "Nogueira, R., Jiang, Z., Pradeep, R., and Lin, J. (2020). Document Ranking with a "
    "Pretrained Sequence-to-Sequence Model. <i>Findings of EMNLP</i>.",
    "Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.",
    "Efron, B. and Tibshirani, R. J. (1993). <i>An Introduction to the Bootstrap</i>. "
    "Chapman &amp; Hall.",
    "Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench "
    "and Chatbot Arena. <i>NeurIPS Datasets and Benchmarks</i>.",
    "Saad-Falcon, J., Khattab, O., Potts, C., and Zaharia, M. (2024). ARES: An Automated "
    "Evaluation Framework for Retrieval-Augmented Generation Systems. <i>NAACL</i>. "
    "arXiv:2311.09476.",
    "Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021). BEIR: A "
    "Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. "
    "<i>NeurIPS Datasets and Benchmarks</i>.",
]
for i, r in enumerate(refs, start=1):
    P(f"[{i}]&nbsp;&nbsp;{r}", ref_style)

doc.build(story)
print("wrote", OUT)
