"""Generate cell type validation algorithm explanation PDF.

Output: outputs/celltype_algorithm.pdf
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOL_ROOT = PROJECT_ROOT / "outputs" / "cosmx_molecular_analysis"
OUT_PDF  = PROJECT_ROOT / "outputs" / "celltype_algorithm.pdf"

BG   = "#0D1117"
CARD = "#161B22"
CARD2= "#1C2128"
WHITE= "#E6EDF3"
GRAY = "#8B949E"
BLUE = "#58A6FF"
GREEN= "#3FB950"
RED  = "#F85149"
ORANGE="#D29922"
PURPLE="#BC8CFF"
TEAL = "#2A9D8F"

PALETTE = {
    "Neuron": "#E63946", "Astrocyte": "#457B9D", "Microglia": "#2A9D8F",
    "Oligodendrocyte": "#E9C46A", "OPC": "#F4A261", "Endothelial": "#9B2226",
    "Pericyte_VSMC": "#9D4EDD", "LowConfidence": "#CCCCCC",
}

MARKER_GENES = {
    "Neuron":        ["SNAP25", "MAP2", "RBFOX3", "SYT1", "SYP"],
    "Astrocyte":     ["GFAP", "AQP4", "ALDH1L1", "S100B"],
    "Microglia":     ["AIF1", "C1QA", "C1QB", "C1QC", "TYROBP", "P2RY12", "TMEM119"],
    "Oligodendrocyte":["MBP", "PLP1", "MOG", "OLIG2"],
    "OPC":           ["PDGFRA", "CSPG4"],
    "Endothelial":   ["CLDN5", "PECAM1", "FLT1"],
    "Pericyte_VSMC": ["ACTA2", "PDGFRB", "RGS5"],
}


def set_dark(fig, *axes):
    fig.patch.set_facecolor(BG)
    for ax in axes:
        if ax is not None:
            ax.set_facecolor(BG)


def save(pdf, fig):
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_box(ax, x, y, w, h, label, sublabel="", color=BLUE, fontsize=10, transform=None):
    tr = transform or ax.transAxes
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.01",
                           facecolor=color + "33", edgecolor=color,
                           linewidth=1.5, transform=tr, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.008 if sublabel else 0), label,
            transform=tr, ha="center", va="center",
            fontsize=fontsize, color=WHITE, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.030, sublabel,
                transform=tr, ha="center", va="center",
                fontsize=fontsize - 2, color=GRAY, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=GRAY, transform=None):
    tr = transform or ax.transAxes
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 xycoords=tr, textcoords=tr,
                 arrowprops=dict(arrowstyle="-|>", color=color,
                                 lw=1.5, mutation_scale=14))


# ══════════════════════════════════════════════════════════════════════════════
with PdfPages(str(OUT_PDF)) as pdf:

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 — Title
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))
    set_dark(fig, ax); ax.axis("off")

    ax.text(0.5, 0.82, "Cell Type Validation",
            transform=ax.transAxes, ha="center", fontsize=30,
            fontweight="bold", color=WHITE)
    ax.text(0.5, 0.73, "Algorithm & Pipeline Documentation",
            transform=ax.transAxes, ha="center", fontsize=18, color=BLUE)
    ax.text(0.5, 0.63, "CosMx Spatial Transcriptomics — FOV00001 (AD)",
            transform=ax.transAxes, ha="center", fontsize=13, color=GRAY)

    # Horizontal rule
    ax.plot([0.1, 0.9], [0.58, 0.58], color=BLUE, linewidth=0.8, alpha=0.5,
            transform=ax.transAxes)

    # Summary boxes
    summary = [
        ("7", "Cell types"),
        ("30", "Marker genes"),
        ("509", "Cells (FOV1)"),
        ("0.15", "Margin threshold"),
    ]
    for i, (val, lbl) in enumerate(summary):
        xp = 0.15 + i * 0.23
        rect = FancyBboxPatch((xp - 0.09, 0.42), 0.18, 0.12,
                               boxstyle="round,pad=0.01",
                               facecolor=CARD, edgecolor=BLUE + "88",
                               linewidth=1, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(xp, 0.505, val, transform=ax.transAxes,
                ha="center", fontsize=20, color=BLUE, fontweight="bold")
        ax.text(xp, 0.435, lbl, transform=ax.transAxes,
                ha="center", fontsize=9, color=GRAY)

    ax.text(0.5, 0.32,
            "Pipeline:  Transcript Assignment  →  AnnData Build  →  QC Filter\n"
            "→  Normalize / Cluster  →  Marker Scoring  →  Cell Type Prediction",
            transform=ax.transAxes, ha="center", fontsize=11,
            color=GRAY, linespacing=1.8)

    ax.text(0.5, 0.10, "2026-06-09  |  cosmx_molecular_analysis pipeline",
            transform=ax.transAxes, ha="center", fontsize=9, color=GRAY + "88",
            style="italic")

    d = pdf.infodict()
    d["Title"] = "Cell Type Validation Algorithm"
    save(pdf, fig)
    print("Page 1: title")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2 — Full pipeline flowchart
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))
    set_dark(fig, ax); ax.axis("off")
    fig.suptitle("Full Pipeline Overview", fontsize=14, fontweight="bold",
                 color=WHITE, y=0.97)

    steps = [
        (0.50, 0.88, "① Transcript\nAssignment",     "tx_file.csv.gz\n→ mask pixel lookup",   BLUE),
        (0.50, 0.73, "② Build AnnData",               "cell × gene\ncount matrix",              GREEN),
        (0.50, 0.58, "③ QC Filtering",                "min_transcripts\nmin_genes filter",      ORANGE),
        (0.50, 0.43, "④ Normalize &\nCluster",        "log1p / PCA / k-NN\nLeiden / UMAP",     PURPLE),
        (0.50, 0.28, "⑤ Marker\nScoring",             "mean expr per\ncell type markers",      TEAL),
        (0.50, 0.13, "⑥ Cell Type\nPrediction",       "argmax score\n+ margin threshold",      RED),
    ]

    for x, y, label, sub, col in steps:
        draw_box(ax, x, y, 0.32, 0.10, label, sub, color=col, fontsize=10)

    for i in range(len(steps) - 1):
        _, y1, _, _, _ = steps[i]
        _, y2, _, _, _ = steps[i+1]
        arrow(ax, 0.50, y1 - 0.052, 0.50, y2 + 0.052)

    # Side annotations
    annotations = [
        (0.82, 0.88, BLUE,   "script: 01_assign_transcripts.py\nsrc: transcript_assignment.py"),
        (0.82, 0.73, GREEN,  "script: 02_build_anndata.py\nsrc: anndata_builder.py"),
        (0.82, 0.58, ORANGE, "script: 03_qc_filter.py\nsrc: qc.py"),
        (0.82, 0.43, PURPLE, "script: 04_normalize_cluster.py\nsrc: normalization.py / clustering.py"),
        (0.82, 0.28, TEAL,   "script: 05_cell_typing.py\nsrc: cell_typing.py"),
        (0.82, 0.13, RED,    "output: cell_type_table.csv\nGUI: 06_CellType_Validation.py"),
    ]
    for x, y, col, txt in annotations:
        ax.text(x, y, txt, transform=ax.transAxes,
                ha="center", va="center", fontsize=6.5, color=col,
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col+"18",
                          edgecolor=col+"55", linewidth=0.8))

    ax.plot([0.695, 0.695], [0.05, 0.95], color=GRAY + "33",
            linewidth=0.8, linestyle="--", transform=ax.transAxes)

    save(pdf, fig)
    print("Page 2: pipeline flowchart")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3 — Step 1: Transcript Assignment
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    set_dark(fig)
    fig.suptitle("Step 1 — Transcript Assignment", fontsize=14,
                 fontweight="bold", color=WHITE, y=0.97)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_diag = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])
    set_dark(fig, ax_diag, ax_info)
    ax_diag.axis("off"); ax_info.axis("off")

    # Left: diagram of mask lookup
    ax_diag.set_title("Pixel-based mask lookup", color=BLUE, fontsize=11, pad=8)

    # Draw fake mask grid
    grid_n = 6
    for i in range(grid_n):
        for j in range(grid_n):
            val = 0
            if 2 <= i <= 4 and 1 <= j <= 3:
                val = 1
            elif 1 <= i <= 2 and 4 <= j <= 5:
                val = 2
            color = (PALETTE["Neuron"] + "55" if val == 1 else
                     PALETTE["Astrocyte"] + "55" if val == 2 else "#22272E")
            rect = FancyBboxPatch((j * 0.14 + 0.05, i * 0.12 + 0.18),
                                   0.13, 0.11,
                                   boxstyle="square,pad=0.005",
                                   facecolor=color,
                                   edgecolor=GRAY + "55", linewidth=0.5,
                                   transform=ax_diag.transAxes)
            ax_diag.add_patch(rect)
            if val > 0:
                ax_diag.text(j*0.14+0.115, i*0.12+0.235, str(val),
                             transform=ax_diag.transAxes,
                             ha="center", va="center", fontsize=8,
                             color=WHITE, fontweight="bold")

    # Transcript dot
    ax_diag.scatter([0.38], [0.54], s=120, c=GREEN, zorder=5,
                    transform=ax_diag.transAxes, marker="*")
    ax_diag.text(0.38, 0.60, "transcript\n(x=2, y=3)", transform=ax_diag.transAxes,
                 ha="center", fontsize=8, color=GREEN)
    ax_diag.annotate("",
                      xy=(0.38, 0.55), xytext=(0.38, 0.49),
                      xycoords=ax_diag.transAxes, textcoords=ax_diag.transAxes,
                      arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2))

    ax_diag.text(0.5, 0.10,
                 "cell_id = FOV00001_" + "1",
                 transform=ax_diag.transAxes, ha="center",
                 fontsize=9, color=PALETTE["Neuron"],
                 bbox=dict(boxstyle="round", facecolor=PALETTE["Neuron"]+"22",
                           edgecolor=PALETTE["Neuron"]+"88"))

    ax_diag.text(0.5, 0.92, "Expanded mask (cpsam)",
                 transform=ax_diag.transAxes, ha="center",
                 fontsize=9, color=GRAY)

    # Right: algorithm description
    ax_info.set_title("Algorithm details", color=BLUE, fontsize=11, pad=8)

    lines = [
        ("Input",           BLUE,   True),
        ("  tx_file.csv.gz — per-transcript records:", WHITE, False),
        ("  x_local_px, y_local_px, target (gene)", GRAY, False),
        ("  fov, qv (quality value)", GRAY, False),
        ("", WHITE, False),
        ("Mask lookup",     BLUE,   True),
        ("  label = expanded_mask[y_px, x_px]", WHITE, False),
        ("  if label > 0:", WHITE, False),
        ("    cell_id = f\"{fov_name}_{label}\"", GREEN, False),
        ("  else:  → 'unassigned'", RED, False),
        ("", WHITE, False),
        ("Count matrix build",BLUE, True),
        ("  pivot(cell_id × gene) → count", WHITE, False),
        ("  each cell gets its assigned tx counts", GRAY, False),
        ("", WHITE, False),
        ("Fallback (FOV37/43)",ORANGE, True),
        ("  native centroid → custom mask lookup", WHITE, False),
        ("  native exprMat counts mapped to", GRAY, False),
        ("  custom cell IDs", GRAY, False),
        ("", WHITE, False),
        ("Output",          GREEN,  True),
        ("  rna_counts_matrix.csv.gz", WHITE, False),
        ("  assignment_summary.csv", WHITE, False),
    ]
    y_pos = 0.96
    for text, color, bold in lines:
        ax_info.text(0.04, y_pos, text, transform=ax_info.transAxes,
                     fontsize=8.5, color=color,
                     fontweight="bold" if bold else "normal",
                     fontfamily="monospace" if text.startswith("  ") else "sans-serif",
                     va="top")
        y_pos -= 0.040 if bold else 0.036

    save(pdf, fig)
    print("Page 3: transcript assignment")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 4 — Step 4: Normalize & Cluster
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    set_dark(fig)
    fig.suptitle("Step 4 — Normalization & Clustering", fontsize=14,
                 fontweight="bold", color=WHITE, y=0.97)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    ax_flow = fig.add_subplot(gs[0])
    ax_param = fig.add_subplot(gs[1])
    set_dark(fig, ax_flow, ax_param)
    ax_flow.axis("off"); ax_param.axis("off")

    # Left: normalization flow
    ax_flow.set_title("Normalization → Embedding", color=PURPLE, fontsize=11, pad=8)

    norm_steps = [
        (0.5, 0.90, "Raw counts matrix\n(cell × gene)", CARD),
        (0.5, 0.75, "Library-size normalize\ntarget_sum = 10,000", PURPLE+"33"),
        (0.5, 0.60, "log1p transform\nlog(x + 1)", PURPLE+"33"),
        (0.5, 0.45, "Highly variable genes\nn_top_genes = 2,000", PURPLE+"33"),
        (0.5, 0.30, "PCA\nn_pcs = 30", BLUE+"33"),
        (0.5, 0.15, "k-NN graph  →  Leiden\n+ UMAP embedding", GREEN+"33"),
    ]
    for x, y, label, fc in norm_steps:
        rect = FancyBboxPatch((x-0.35, y-0.06), 0.70, 0.10,
                               boxstyle="round,pad=0.01",
                               facecolor=fc, edgecolor=GRAY+"88",
                               linewidth=1, transform=ax_flow.transAxes)
        ax_flow.add_patch(rect)
        ax_flow.text(x, y, label, transform=ax_flow.transAxes,
                     ha="center", va="center", fontsize=8.5, color=WHITE)

    for i in range(len(norm_steps)-1):
        _, y1, _, _ = norm_steps[i]
        _, y2, _, _ = norm_steps[i+1]
        arrow(ax_flow, 0.5, y1-0.062, 0.5, y2+0.062)

    ax_flow.text(0.88, 0.755, "scanpy\npp.normalize_total", transform=ax_flow.transAxes,
                 ha="center", fontsize=6.5, color=GRAY,
                 bbox=dict(boxstyle="round", facecolor=CARD, edgecolor=GRAY+"44"))
    ax_flow.text(0.88, 0.605, "scanpy\npp.log1p", transform=ax_flow.transAxes,
                 ha="center", fontsize=6.5, color=GRAY,
                 bbox=dict(boxstyle="round", facecolor=CARD, edgecolor=GRAY+"44"))
    ax_flow.text(0.88, 0.455, "scanpy\npp.highly_variable_genes", transform=ax_flow.transAxes,
                 ha="center", fontsize=6.5, color=GRAY,
                 bbox=dict(boxstyle="round", facecolor=CARD, edgecolor=GRAY+"44"))
    ax_flow.text(0.88, 0.305, "scanpy\ntl.pca", transform=ax_flow.transAxes,
                 ha="center", fontsize=6.5, color=GRAY,
                 bbox=dict(boxstyle="round", facecolor=CARD, edgecolor=GRAY+"44"))
    ax_flow.text(0.88, 0.155, "scanpy\ntl.leiden / tl.umap", transform=ax_flow.transAxes,
                 ha="center", fontsize=6.5, color=GRAY,
                 bbox=dict(boxstyle="round", facecolor=CARD, edgecolor=GRAY+"44"))

    # Right: parameters table
    ax_param.set_title("Parameters", color=PURPLE, fontsize=11, pad=8)

    params_table = [
        ("Parameter",          "Value",     "Role"),
        ("rna_method",         "log1p",     "Normalization method"),
        ("target_sum",         "10,000",    "Library size target"),
        ("n_top_genes",        "2,000",     "HVG selection"),
        ("n_pcs",              "30",        "PCA dimensions"),
        ("n_neighbors",        "15",        "k-NN graph"),
        ("resolution",         "0.5",       "Leiden clustering"),
    ]
    col_x = [0.05, 0.42, 0.66]
    y_p = 0.92
    for i, (p, v, r) in enumerate(params_table):
        fc = BLUE+"22" if i == 0 else (CARD if i % 2 == 0 else CARD2)
        bg = FancyBboxPatch((0.02, y_p - 0.045), 0.96, 0.048,
                             boxstyle="square,pad=0.003",
                             facecolor=fc, edgecolor=GRAY+"22",
                             linewidth=0.5, transform=ax_param.transAxes)
        ax_param.add_patch(bg)
        bold = (i == 0)
        for j, (x_c, txt) in enumerate(zip(col_x, [p, v, r])):
            ax_param.text(x_c, y_p - 0.020, txt, transform=ax_param.transAxes,
                          fontsize=8.5 if not bold else 8,
                          color=WHITE if bold else (BLUE if j == 1 else WHITE),
                          fontweight="bold" if bold else "normal", va="center")
        y_p -= 0.052

    ax_param.text(0.5, 0.52,
                  "Purpose of clustering:\nLeiden labels are stored as adata.obs['leiden']\n"
                  "and used as a reference for understanding the\ncell population structure,\n"
                  "independent of the marker-based annotation.",
                  transform=ax_param.transAxes, ha="center", va="center",
                  fontsize=9, color=GRAY, linespacing=1.6,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD,
                            edgecolor=PURPLE+"55", linewidth=1))

    save(pdf, fig)
    print("Page 4: normalization & clustering")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5 — Step 5: Marker Scoring (core algorithm)
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    set_dark(fig)
    fig.suptitle("Step 5 — Marker-based Cell Type Scoring  (Core Algorithm)",
                 fontsize=13, fontweight="bold", color=WHITE, y=0.97)

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45,
                           top=0.92, bottom=0.05)
    ax_formula = fig.add_subplot(gs[0, 0])
    ax_markers = fig.add_subplot(gs[0, 1])
    ax_score   = fig.add_subplot(gs[1, 0])
    ax_margin  = fig.add_subplot(gs[1, 1])
    set_dark(fig, ax_formula, ax_markers, ax_score, ax_margin)

    # ── Formula panel ──
    ax_formula.axis("off")
    ax_formula.set_title("Score formula", color=TEAL, fontsize=11, pad=6)

    ax_formula.text(0.5, 0.88,
                    "score(cell_i, type_t)  =",
                    transform=ax_formula.transAxes,
                    ha="center", fontsize=10, color=WHITE, fontweight="bold")
    ax_formula.text(0.5, 0.74,
                    "mean( X_norm[cell_i, markers_t] )",
                    transform=ax_formula.transAxes,
                    ha="center", fontsize=10, color=TEAL,
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=TEAL+"22",
                              edgecolor=TEAL+"88", linewidth=1.2))

    ax_formula.text(0.5, 0.60,
                    "X_norm = log1p(counts / lib_size × 10k)",
                    transform=ax_formula.transAxes,
                    ha="center", fontsize=8.5, color=GRAY)

    ax_formula.text(0.5, 0.48, "markers_t = genes for type t\n(only those present in panel)",
                    transform=ax_formula.transAxes,
                    ha="center", fontsize=8.5, color=GRAY, linespacing=1.5)

    ax_formula.text(0.5, 0.30, "→  Score matrix shape:\n(n_cells × 7 cell types)",
                    transform=ax_formula.transAxes,
                    ha="center", fontsize=9, color=WHITE, linespacing=1.5)

    # Missing markers note
    ax_formula.text(0.5, 0.13,
                    "⚠  Markers absent from the CosMx panel\nare silently excluded from the mean",
                    transform=ax_formula.transAxes,
                    ha="center", fontsize=8, color=ORANGE, linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=ORANGE+"18",
                              edgecolor=ORANGE+"55", linewidth=0.8))

    # ── Marker gene table ──
    ax_markers.axis("off")
    ax_markers.set_title("Marker genes (marker_genes.yaml)", color=TEAL, fontsize=11, pad=6)

    y_m = 0.96
    for ct, genes in MARKER_GENES.items():
        color = PALETTE.get(ct, "#888888")
        rect = FancyBboxPatch((0.01, y_m - 0.09), 0.98, 0.092,
                               boxstyle="round,pad=0.005",
                               facecolor=color + "22", edgecolor=color + "66",
                               linewidth=0.8, transform=ax_markers.transAxes)
        ax_markers.add_patch(rect)
        ax_markers.text(0.04, y_m - 0.044, ct,
                        transform=ax_markers.transAxes,
                        fontsize=8, color=color, fontweight="bold", va="center")
        ax_markers.text(0.28, y_m - 0.044,
                        "  ".join(genes),
                        transform=ax_markers.transAxes,
                        fontsize=7.5, color=WHITE, va="center",
                        fontfamily="monospace")
        ax_markers.text(0.99, y_m - 0.044, f"n={len(genes)}",
                        transform=ax_markers.transAxes,
                        fontsize=7, color=GRAY, va="center", ha="right")
        y_m -= 0.114

    # ── Score bar chart (synthetic example) ──
    ax_score.set_facecolor(CARD)
    ax_score.set_title("Example: score profile of one cell", color=TEAL, fontsize=10, pad=6)

    ct_names = list(MARKER_GENES.keys())
    example_scores = [0.82, 0.12, 0.08, 0.05, 0.06, 0.04, 0.07]
    bar_colors = [PALETTE.get(c, "#888") for c in ct_names]
    bars = ax_score.barh(ct_names, example_scores, color=bar_colors, alpha=0.85,
                          edgecolor=BG, linewidth=0.5)
    ax_score.axvline(0.82, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.8)
    ax_score.axvline(0.12, color=ORANGE, linewidth=1.0, linestyle=":", alpha=0.8)
    ax_score.set_xlabel("Score", color=WHITE, fontsize=9)
    ax_score.tick_params(colors=WHITE, labelsize=8)
    ax_score.spines[:].set_color(GRAY+"44")
    for spine in ax_score.spines.values():
        spine.set_linewidth(0.5)
    ax_score.text(0.84, 0.95, "top1 = Neuron\n(0.82)", transform=ax_score.transAxes,
                  fontsize=7.5, color=GREEN, va="top")
    ax_score.text(0.84, 0.73, "top2 = Astrocyte\n(0.12)", transform=ax_score.transAxes,
                  fontsize=7.5, color=ORANGE, va="top")

    # ── Margin threshold diagram ──
    ax_margin.axis("off")
    ax_margin.set_facecolor(CARD)
    ax_margin.set_title("LowConfidence decision rule", color=RED, fontsize=10, pad=6)

    ax_margin.text(0.5, 0.92,
                   "predicted_type = argmax( score_matrix )",
                   transform=ax_margin.transAxes, ha="center",
                   fontsize=9, color=WHITE, fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD2,
                             edgecolor=BLUE+"55"))

    ax_margin.text(0.5, 0.78,
                   "margin = top1_score  −  top2_score",
                   transform=ax_margin.transAxes, ha="center",
                   fontsize=9, color=WHITE, fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD2,
                             edgecolor=BLUE+"55"))

    # Decision tree
    ax_margin.text(0.5, 0.62, "margin ≥ 0.15 ?",
                   transform=ax_margin.transAxes, ha="center",
                   fontsize=10, color=WHITE, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD,
                             edgecolor=WHITE+"66"))
    # YES branch
    ax_margin.annotate("", xy=(0.20, 0.42), xytext=(0.38, 0.57),
                        xycoords=ax_margin.transAxes, textcoords=ax_margin.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))
    ax_margin.text(0.20, 0.38, "YES →\npredicted_type",
                   transform=ax_margin.transAxes, ha="center",
                   fontsize=9, color=GREEN,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN+"22",
                             edgecolor=GREEN+"88"))
    # NO branch
    ax_margin.annotate("", xy=(0.78, 0.42), xytext=(0.62, 0.57),
                        xycoords=ax_margin.transAxes, textcoords=ax_margin.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5))
    ax_margin.text(0.78, 0.38, "NO →\nLowConfidence",
                   transform=ax_margin.transAxes, ha="center",
                   fontsize=9, color=RED,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=RED+"22",
                             edgecolor=RED+"88"))

    ax_margin.text(0.5, 0.16,
                   "Threshold = 0.15\n"
                   "→ If top-1 and top-2 scores are too close,\n"
                   "   the cell type is ambiguous → LowConfidence",
                   transform=ax_margin.transAxes, ha="center",
                   fontsize=8.5, color=GRAY, linespacing=1.6,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD2,
                             edgecolor=RED+"33"))

    save(pdf, fig)
    print("Page 5: marker scoring")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 6 — FOV00001 actual results
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    set_dark(fig)
    fig.suptitle("FOV00001 — Actual Annotation Results",
                 fontsize=14, fontweight="bold", color=WHITE, y=0.97)

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.50,
                           top=0.92, bottom=0.06)
    ax_ct   = fig.add_subplot(gs[0, 0])
    ax_sc   = fig.add_subplot(gs[0, 1])
    ax_mgn  = fig.add_subplot(gs[1, 0])
    ax_tx   = fig.add_subplot(gs[1, 1])
    for ax in [ax_ct, ax_sc, ax_mgn, ax_tx]:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=WHITE, labelsize=8)
        ax.spines[:].set_color(GRAY + "55")

    # Load real data
    ct = pd.read_csv(MOL_ROOT / "cell_type_table.csv")
    fov1 = ct[ct["fov_name"] == "FOV00001"]

    # ── Cell type distribution bar ──
    counts = fov1["predicted_cell_type"].value_counts()
    ax_ct.set_title("Cell type distribution", color=WHITE, fontsize=10, pad=6)
    bar_ct = [c for c in ["Neuron","Astrocyte","Microglia","Oligodendrocyte",
                           "OPC","Endothelial","Pericyte_VSMC","LowConfidence"]
              if c in counts.index]
    vals = [counts[c] for c in bar_ct]
    cols = [PALETTE.get(c,"#888") for c in bar_ct]
    ax_ct.barh(bar_ct, vals, color=cols, alpha=0.85, edgecolor=BG, linewidth=0.3)
    for i, (v, c) in enumerate(zip(vals, bar_ct)):
        ax_ct.text(v + 2, i, str(v), va="center", fontsize=7.5, color=WHITE)
    ax_ct.set_xlabel("n cells", color=WHITE, fontsize=8)
    ax_ct.invert_yaxis()

    # ── Score distribution violin ──
    try:
        import anndata as ad
        adata = ad.read_h5ad(str(MOL_ROOT / "anndata" / "rna_annotated.h5ad"))
        fov_idx = [c for c in adata.obs_names if c.startswith("FOV00001_")]
        if fov_idx and "cell_type_scores" in adata.obsm:
            score_cols = adata.uns.get("cell_type_score_columns", [])
            sdf = pd.DataFrame(adata.obsm["cell_type_scores"],
                               index=adata.obs_names, columns=score_cols)
            sdf_fov = sdf.loc[fov_idx]
            ax_sc.set_title("Marker score distribution (FOV1)", color=WHITE, fontsize=10, pad=6)
            data_list = [sdf_fov[c].dropna().values for c in score_cols if c in sdf_fov.columns]
            bp = ax_sc.boxplot(data_list, patch_artist=True,
                               medianprops=dict(color=WHITE, lw=1.5),
                               whiskerprops=dict(color=GRAY),
                               capprops=dict(color=GRAY),
                               flierprops=dict(marker=".", markersize=2, color=GRAY))
            for patch, ct_name in zip(bp["boxes"], score_cols):
                patch.set_facecolor(PALETTE.get(ct_name, "#888") + "88")
            ax_sc.set_xticks(range(1, len(score_cols)+1))
            ax_sc.set_xticklabels([c[:5] for c in score_cols],
                                   rotation=30, ha="right", fontsize=7)
            ax_sc.set_ylabel("Score", color=WHITE, fontsize=8)
    except Exception:
        ax_sc.axis("off")
        ax_sc.text(0.5, 0.5, "AnnData load failed", ha="center",
                   transform=ax_sc.transAxes, color=GRAY)

    # ── Margin distribution ──
    ax_mgn.set_title("Confidence margin distribution", color=WHITE, fontsize=10, pad=6)
    if "cell_type_margin" in fov1.columns:
        margins = fov1["cell_type_margin"].dropna()
        ax_mgn.hist(margins, bins=30, color=TEAL, alpha=0.8,
                    edgecolor=BG, linewidth=0.3)
        ax_mgn.axvline(0.15, color=RED, linewidth=1.5, linestyle="--")
        ax_mgn.text(0.16, ax_mgn.get_ylim()[1]*0.9 if ax_mgn.get_ylim()[1] > 0 else 1,
                    "threshold\n= 0.15", color=RED, fontsize=7.5)
        ax_mgn.set_xlabel("margin (top1 − top2)", color=WHITE, fontsize=8)
        ax_mgn.set_ylabel("n cells", color=WHITE, fontsize=8)

    # ── Transcripts per cell by type ──
    ax_tx.set_title("Transcripts/cell by type", color=WHITE, fontsize=10, pad=6)
    if "n_transcripts" in fov1.columns:
        order = ["Neuron","Astrocyte","Microglia","Oligodendrocyte",
                 "OPC","Endothelial","Pericyte_VSMC","LowConfidence"]
        tx_data = [fov1[fov1["predicted_cell_type"]==c]["n_transcripts"].dropna().values
                   for c in order if c in fov1["predicted_cell_type"].values]
        labs = [c[:6] for c in order if c in fov1["predicted_cell_type"].values]
        bp2 = ax_tx.boxplot(tx_data, patch_artist=True,
                             medianprops=dict(color=WHITE, lw=1.5),
                             whiskerprops=dict(color=GRAY),
                             capprops=dict(color=GRAY),
                             flierprops=dict(marker=".", markersize=2, color=GRAY))
        for patch, ct_name in zip(bp2["boxes"],
                                   [c for c in order if c in fov1["predicted_cell_type"].values]):
            patch.set_facecolor(PALETTE.get(ct_name, "#888") + "88")
        ax_tx.set_xticks(range(1, len(labs)+1))
        ax_tx.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
        ax_tx.set_ylabel("n transcripts", color=WHITE, fontsize=8)

    save(pdf, fig)
    print("Page 6: FOV1 results")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 7 — Validation GUI & output files
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))
    set_dark(fig, ax); ax.axis("off")
    fig.suptitle("Step 6 — Visualization & Manual Validation GUI",
                 fontsize=14, fontweight="bold", color=WHITE, y=0.97)

    # Left column: GUI tabs
    gui_tabs = [
        ("Cell Type Map",        "Enhanced DAPI + cell type color overlay\n(α blend of mask and background)"),
        ("Marker Expression",    "Per-gene expression heatmap on mask\n(RNA genes / obs columns selectable)"),
        ("Cell Inspector",       "Click a cell → show type, score, margin\ntop expressed genes, morphology"),
        ("QC Dashboard",         "Per-FOV composition bar, violin plots\ntranscript/gene distributions"),
        ("Manual Review",        "Override predicted cell type per cell\nexport manual_celltype_review.csv"),
    ]
    y_g = 0.88
    ax.text(0.01, 0.94, "GUI Tabs (06_CellType_Validation.py)", transform=ax.transAxes,
            fontsize=11, color=BLUE, fontweight="bold")
    for tab, desc in gui_tabs:
        rect = FancyBboxPatch((0.01, y_g - 0.075), 0.44, 0.072,
                               boxstyle="round,pad=0.01",
                               facecolor=BLUE+"18", edgecolor=BLUE+"66",
                               linewidth=1, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.04, y_g - 0.024, tab, transform=ax.transAxes,
                fontsize=9, color=BLUE, fontweight="bold", va="center")
        ax.text(0.04, y_g - 0.055, desc, transform=ax.transAxes,
                fontsize=7.5, color=GRAY, va="center")
        y_g -= 0.090

    # Right column: output files
    ax.text(0.52, 0.94, "Output Files", transform=ax.transAxes,
            fontsize=11, color=GREEN, fontweight="bold")
    outputs = [
        ("cell_type_table.csv",          GREEN,
         "cell_id, predicted_cell_type, score, margin\nfov_name, condition, x_um, y_um"),
        ("rna_annotated.h5ad",           TEAL,
         "AnnData: normalized expr + UMAP\nobs: predicted_cell_type, leiden\nobsm: cell_type_scores, spatial"),
        ("cell_typing_plots/\numap_by_cell_type.png",  PURPLE,
         "UMAP colored by predicted cell type"),
        ("cell_typing_plots/\nmarker_score_heatmap.png", PURPLE,
         "Heatmap: mean score per cell type × marker"),
        ("spatial/\nspatial_cell_type_map_FOV*.png", ORANGE,
         "Per-FOV spatial scatter plot (x_um, y_um)\ncolored by predicted cell type"),
    ]
    y_o = 0.88
    for fname, col, desc in outputs:
        rect = FancyBboxPatch((0.52, y_o - 0.090), 0.47, 0.086,
                               boxstyle="round,pad=0.01",
                               facecolor=col+"18", edgecolor=col+"66",
                               linewidth=1, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.55, y_o - 0.028, fname, transform=ax.transAxes,
                fontsize=8.5, color=col, fontweight="bold", va="center",
                fontfamily="monospace")
        ax.text(0.55, y_o - 0.068, desc, transform=ax.transAxes,
                fontsize=7.5, color=GRAY, va="center")
        y_o -= 0.100

    # Limitations box
    ax.text(0.5, 0.065,
            "Current Limitations  —  Work in Progress\n"
            "① Marker scoring does not use unsupervised cluster labels (Leiden) for annotation → "
            "potential mismatch between cluster and marker-based types\n"
            "② LowConfidence ~35% of FOV1 cells → high ambiguity driven by low transcript counts or missing panel markers\n"
            "③ Cell type annotation is run only on cpsam segmentation masks → other models lack cell type labels",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color=ORANGE, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=ORANGE+"14",
                      edgecolor=ORANGE+"55", linewidth=1.2))

    save(pdf, fig)
    print("Page 7: GUI & outputs")

    d = pdf.infodict()
    d["Title"]   = "Cell Type Validation Algorithm"
    d["Subject"] = "CosMx spatial transcriptomics cell type annotation pipeline"

print(f"\nDone → {OUT_PDF}")
