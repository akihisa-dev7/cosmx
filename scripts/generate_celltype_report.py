"""Generate cell type comparison PDF report (AD vs Control).

Usage:
    python scripts/generate_celltype_report.py
Output:
    outputs/celltype_report.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "cell_segmentation_pipeline" / "app"))

# ── Paths ──────────────────────────────────────────────────────────────────────
MOL_ROOT      = PROJECT_ROOT / "outputs" / "cosmx_molecular_analysis"
MASK_DIR      = PROJECT_ROOT / "outputs" / "pilot_4fov" / "expanded_masks"
RAW_DIR       = PROJECT_ROOT / "outputs_ssd" / "pilot_segmentation" / "raw_images"
UMAP_PNG      = MOL_ROOT / "cell_typing_plots" / "umap_by_cell_type.png"
HEATMAP_PNG   = MOL_ROOT / "cell_typing_plots" / "marker_score_heatmap.png"
OUT_PDF       = PROJECT_ROOT / "outputs" / "celltype_report.pdf"

# ── Color palette (matches GUI) ────────────────────────────────────────────────
PALETTE: dict[str, str] = {
    "Neuron":          "#E63946",
    "Astrocyte":       "#457B9D",
    "Microglia":       "#2A9D8F",
    "Oligodendrocyte": "#E9C46A",
    "OPC":             "#F4A261",
    "Endothelial":     "#9B2226",
    "Pericyte_VSMC":   "#9D4EDD",
    "LowConfidence":   "#CCCCCC",
    "Unknown":         "#888888",
}
ORDERED_TYPES = [
    "Neuron", "Astrocyte", "Microglia", "Oligodendrocyte",
    "OPC", "Endothelial", "Pericyte_VSMC", "LowConfidence",
]

# FOV metadata
FOVS = [
    {"fov": "FOV00001", "condition": "AD",      "label": "FOV 1 (AD)"},
    {"fov": "FOV00007", "condition": "AD",      "label": "FOV 7 (AD)"},
    {"fov": "FOV00037", "condition": "Control", "label": "FOV 37 (Control)"},
    {"fov": "FOV00043", "condition": "Control", "label": "FOV 43 (Control)"},
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def find_raw(fov: str) -> Path | None:
    for p in RAW_DIR.glob(f"{fov}_*raw_original.tif"):
        return p
    for p in RAW_DIR.glob(f"{fov}_*raw.tif"):
        return p
    return None


def find_mask(fov: str) -> Path | None:
    for p in MASK_DIR.glob(f"{fov}_*cpsam_expanded_mask.tif"):
        return p
    for p in MASK_DIR.glob(f"{fov}_*expanded*.tif"):
        return p
    return None


def norm_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def build_ct_rgb(mask: np.ndarray, ct_table: pd.DataFrame, fov: str) -> np.ndarray:
    fov_cells = ct_table[ct_table["fov_name"] == fov]
    ct_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for _, row in fov_cells.iterrows():
        cid = str(row.get("cell_id", ""))
        ct  = str(row.get("predicted_cell_type", "Unknown"))
        try:
            lbl = int(cid.rsplit("_", 1)[-1])
        except ValueError:
            continue
        color = PALETTE.get(ct, PALETTE["Unknown"])
        r, g, b = hex_to_rgb(color)
        ct_rgb[mask == lbl] = [r, g, b]
    return ct_rgb


def blend(raw_gray: np.ndarray, ct_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = np.stack([raw_gray, raw_gray, raw_gray], axis=-1).astype(np.float32)
    fg   = ct_rgb.any(axis=-1)
    out  = base.copy()
    out[fg] = (1 - alpha) * base[fg] + alpha * ct_rgb[fg].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def legend_patches(types: list[str]) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=PALETTE.get(t, "#888888"), label=t)
        for t in types
    ]


# ── Load data ──────────────────────────────────────────────────────────────────
ct_table = pd.read_csv(MOL_ROOT / "cell_type_table.csv")
print(f"Loaded cell_type_table: {len(ct_table)} cells")


# ── Composition stats ──────────────────────────────────────────────────────────
ct_counts = ct_table.groupby(["condition", "predicted_cell_type"]).size().unstack(fill_value=0)
ct_frac   = (ct_counts.T / ct_counts.sum(axis=1)).T * 100  # percent
n_per_cond = ct_table.groupby("condition").size()

# Per-FOV counts
fov_counts = ct_table.groupby(["fov_name", "condition", "predicted_cell_type"]).size().unstack(fill_value=0)
fov_frac   = (fov_counts.T / fov_counts.sum(axis=1)).T * 100


# ══════════════════════════════════════════════════════════════════════════════
# Build PDF
# ══════════════════════════════════════════════════════════════════════════════
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

with PdfPages(str(OUT_PDF)) as pdf:

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 — Title
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#0D1117")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0D1117")
    ax.axis("off")

    ax.text(0.5, 0.72, "CosMx Spatial Transcriptomics",
            ha="center", va="center", fontsize=28, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.62, "Cell Type Distribution Analysis",
            ha="center", va="center", fontsize=20,
            color="#8AB4F8", transform=ax.transAxes)
    ax.text(0.5, 0.50, "AD vs. Control",
            ha="center", va="center", fontsize=32, fontweight="bold",
            color="#E63946", transform=ax.transAxes)

    # legend boxes
    legend_y = 0.32
    col_x = [0.18, 0.38, 0.58, 0.78]
    types_row1 = ["Neuron", "Astrocyte", "Microglia", "Oligodendrocyte"]
    types_row2 = ["OPC", "Endothelial", "Pericyte_VSMC", "LowConfidence"]
    for i, (t, x) in enumerate(zip(types_row1, col_x)):
        rect = mpatches.FancyBboxPatch((x - 0.07, legend_y - 0.03), 0.14, 0.055,
                                       boxstyle="round,pad=0.01",
                                       facecolor=PALETTE[t], edgecolor="white",
                                       linewidth=0.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, legend_y + 0.008, t, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", transform=ax.transAxes)
    for i, (t, x) in enumerate(zip(types_row2, col_x)):
        rect = mpatches.FancyBboxPatch((x - 0.07, legend_y - 0.09), 0.14, 0.055,
                                       boxstyle="round,pad=0.01",
                                       facecolor=PALETTE[t], edgecolor="white",
                                       linewidth=0.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, legend_y - 0.062, t, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", transform=ax.transAxes)

    # Dataset summary
    ax.text(0.5, 0.14,
            f"Dataset:  {len(FOVS)} FOVs  |  "
            f"AD: {n_per_cond.get('AD', 0):,} cells (2 FOVs)  |  "
            f"Control: {n_per_cond.get('Control', 0):,} cells (2 FOVs)",
            ha="center", va="center", fontsize=11, color="#AAAAAA",
            transform=ax.transAxes)
    ax.text(0.5, 0.06, "2026-06-09   —   Pilot study (n=4 FOVs)",
            ha="center", va="center", fontsize=9, color="#666666",
            transform=ax.transAxes)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Page 1: title done")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2 — Cell Type Color Maps (2×2 grid: AD left, Control right)
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Cell Type Color Maps  —  AD vs. Control",
                 fontsize=16, fontweight="bold", color="white", y=0.98)

    col_labels = ["AD", "Control"]
    for col_idx, condition in enumerate(["AD", "Control"]):
        fov_list = [f["fov"] for f in FOVS if f["condition"] == condition]
        for row_idx, fov in enumerate(fov_list[:2]):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("#111111")

            raw_path  = find_raw(fov)
            mask_path = find_mask(fov)

            if raw_path and mask_path:
                raw  = tifffile.imread(str(raw_path))
                mask = tifffile.imread(str(mask_path)).astype(np.int32)
                if raw.ndim == 3:
                    raw = raw[0] if raw.shape[0] < raw.shape[-1] else raw[..., 0]
                gray    = norm_uint8(raw)
                ct_rgb  = build_ct_rgb(mask, ct_table, fov)
                blended = blend(gray, ct_rgb, alpha=0.55)
                ax.imshow(blended)
            else:
                ax.set_facecolor("#222222")
                ax.text(0.5, 0.5, f"{fov}\nimage not found",
                        ha="center", va="center", color="#888888", fontsize=10,
                        transform=ax.transAxes)

            fov_label = next((f["label"] for f in FOVS if f["fov"] == fov), fov)
            title_color = "#FF6B6B" if condition == "AD" else "#6BCB77"
            ax.set_title(fov_label, fontsize=12, color=title_color, fontweight="bold", pad=6)
            ax.axis("off")

    # Column header labels
    for col_idx, (cond, color) in enumerate(zip(["AD", "Control"], ["#E63946", "#2A9D8F"])):
        axes[0][col_idx].annotate(
            f"── {cond} ──", xy=(0.5, 1.12), xycoords="axes fraction",
            ha="center", fontsize=14, color=color, fontweight="bold"
        )

    # Shared legend at bottom
    patches = legend_patches(ORDERED_TYPES)
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               facecolor="#1A1A2E", edgecolor="#444444", labelcolor="white",
               bbox_to_anchor=(0.5, 0.07), framealpha=0.9)

    # Model name note
    fig.text(0.01, 0.03,
             "Segmentation model: CellPose SAM (cpsam)  |  Mask: expanded (30 px)",
             fontsize=8, color="#8AB4F8", va="bottom", style="italic")

    # Work-in-progress warning
    fig.text(0.99, 0.03,
             "⚠  Work in progress — segmentation is under active improvement."
             "  Results are preliminary and may appear imprecise.",
             fontsize=8, color="#FF6B35", va="bottom", ha="right", style="italic")

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Page 2: color maps done")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3 — Cell Type Composition Comparison
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Cell Type Composition  —  AD vs. Control",
                 fontsize=16, fontweight="bold", color="white", y=0.97)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_bar  = fig.add_subplot(gs[0])
    ax_pie  = fig.add_subplot(gs[1])

    for ax in [ax_bar, ax_pie]:
        ax.set_facecolor("#111827")

    # ── Grouped bar chart ──
    present_types = [t for t in ORDERED_TYPES if t in ct_frac.columns]
    x     = np.arange(len(present_types))
    width = 0.35
    conds = [c for c in ["AD", "Control"] if c in ct_frac.index]

    for i, cond in enumerate(conds):
        vals = [ct_frac.loc[cond, t] if t in ct_frac.columns else 0 for t in present_types]
        color = "#E63946" if cond == "AD" else "#2A9D8F"
        bars = ax_bar.bar(x + i * width - width / 2, vals, width,
                          label=cond, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        for bar, val in zip(bars, vals):
            if val >= 1.5:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f"{val:.1f}%", ha="center", va="bottom",
                            fontsize=7, color="white")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(present_types, rotation=35, ha="right",
                           fontsize=9, color="white")
    ax_bar.set_ylabel("Fraction (%)", color="white", fontsize=11)
    ax_bar.set_title("Cell type fraction by condition", color="white", fontsize=12, pad=8)
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#444444")
    ax_bar.legend(facecolor="#1A1A2E", edgecolor="#444444", labelcolor="white", fontsize=10)
    ax_bar.set_ylim(0, ct_frac.values.max() * 1.22)
    ax_bar.yaxis.label.set_color("white")

    # ── Stacked horizontal bar (per FOV) ──
    fov_order = [f["fov"] for f in FOVS]
    fov_labels_map = {f["fov"]: f["label"] for f in FOVS}
    fov_frac_plot = fov_frac.reindex(fov_order).fillna(0)

    bottoms = np.zeros(len(fov_order))
    for t in present_types:
        vals = fov_frac_plot[t].values if t in fov_frac_plot.columns else np.zeros(len(fov_order))
        ax_pie.barh(range(len(fov_order)), vals, left=bottoms,
                    color=PALETTE.get(t, "#888888"), label=t, edgecolor="white", linewidth=0.3)
        bottoms += vals

    ax_pie.set_yticks(range(len(fov_order)))
    ax_pie.set_yticklabels(
        [fov_labels_map.get(f, f) for f in fov_order],
        fontsize=10, color="white"
    )
    ax_pie.set_xlabel("Fraction (%)", color="white", fontsize=11)
    ax_pie.set_title("Per-FOV cell type composition", color="white", fontsize=12, pad=8)
    ax_pie.tick_params(colors="white")
    ax_pie.spines[:].set_color("#444444")
    ax_pie.set_xlim(0, 105)

    # Divider line between AD and Control FOVs
    ax_pie.axhline(1.5, color="#888888", linewidth=1, linestyle="--", alpha=0.6)
    ax_pie.text(101, 0.5, "AD", color="#E63946", fontsize=9, va="center", fontweight="bold")
    ax_pie.text(101, 2.5, "Ctrl", color="#2A9D8F", fontsize=9, va="center", fontweight="bold")

    patches = legend_patches(present_types)
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               facecolor="#1A1A2E", edgecolor="#444444", labelcolor="white",
               bbox_to_anchor=(0.5, 0.00), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Page 3: composition done")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 4 — LowConfidence & Segmentation Improvement Potential
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Segmentation Quality  —  LowConfidence Cells",
                 fontsize=16, fontweight="bold", color="white", y=0.97)

    gs2 = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)
    ax_lc  = fig.add_subplot(gs2[0])
    ax_msg = fig.add_subplot(gs2[1])

    for ax in [ax_lc, ax_msg]:
        ax.set_facecolor("#111827")

    # ── LowConfidence per FOV bar ──
    lc_vals = []
    fov_lbls = []
    bar_colors = []
    for meta in FOVS:
        fov = meta["fov"]
        total = len(ct_table[ct_table["fov_name"] == fov])
        lc    = len(ct_table[(ct_table["fov_name"] == fov) &
                              (ct_table["predicted_cell_type"] == "LowConfidence")])
        pct   = lc / total * 100 if total > 0 else 0
        lc_vals.append(pct)
        fov_lbls.append(meta["label"])
        bar_colors.append("#E63946" if meta["condition"] == "AD" else "#2A9D8F")

    bars = ax_lc.bar(fov_lbls, lc_vals, color=bar_colors, edgecolor="white",
                     linewidth=0.6, alpha=0.85)
    for bar, val in zip(bars, lc_vals):
        ax_lc.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f"{val:.1f}%", ha="center", va="bottom",
                   fontsize=11, color="white", fontweight="bold")

    ax_lc.set_ylabel("LowConfidence (%)", color="white", fontsize=11)
    ax_lc.set_title("LowConfidence ratio per FOV", color="white", fontsize=12, pad=8)
    ax_lc.tick_params(colors="white")
    ax_lc.set_xticklabels(fov_lbls, rotation=15, ha="right", fontsize=9, color="white")
    ax_lc.spines[:].set_color("#444444")
    ax_lc.set_ylim(0, 60)
    ax_lc.axhline(35, color="#FF6B35", linewidth=1.2, linestyle="--", alpha=0.8)
    ax_lc.text(3.5, 35.8, "~35% avg", color="#FF6B35", fontsize=8, ha="right")

    # ── Message panel ──
    ax_msg.axis("off")

    msg_lines = [
        ("Current Status", 0.88, "#8AB4F8", 14, True),
        ("", 0.80, "white", 10, False),
        ("• Cell type differences are already visible", 0.76, "white", 11, False),
        ("  between AD and Control FOVs", 0.70, "#AAAAAA", 11, False),
        ("", 0.64, "white", 10, False),
        ("• ~35% of all cells are classified as", 0.60, "white", 11, False),
        ("  LowConfidence — unresolved by the", 0.54, "#AAAAAA", 11, False),
        ("  current segmentation model", 0.48, "#AAAAAA", 11, False),
        ("", 0.42, "white", 10, False),
        ("Next Step", 0.36, "#6BCB77", 14, True),
        ("", 0.30, "white", 10, False),
        ("• Improving segmentation precision will", 0.26, "white", 11, False),
        ("  recover these unassigned cells, enabling", 0.20, "#AAAAAA", 11, False),
        ("  more robust AD vs. Control comparisons", 0.14, "#AAAAAA", 11, False),
    ]

    for text, y, color, size, bold in msg_lines:
        ax_msg.text(0.05, y, text, transform=ax_msg.transAxes,
                    fontsize=size, color=color,
                    fontweight="bold" if bold else "normal",
                    va="center")

    # Highlight box
    highlight = mpatches.FancyBboxPatch(
        (0.03, 0.40), 0.94, 0.52,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor="#8AB4F8", facecolor="#1A1A2E", alpha=0.6,
        transform=ax_msg.transAxes
    )
    ax_msg.add_patch(highlight)

    highlight2 = mpatches.FancyBboxPatch(
        (0.03, 0.06), 0.94, 0.30,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor="#6BCB77", facecolor="#0D2010", alpha=0.6,
        transform=ax_msg.transAxes
    )
    ax_msg.add_patch(highlight2)

    # Re-draw text over boxes
    for text, y, color, size, bold in msg_lines:
        ax_msg.text(0.05, y, text, transform=ax_msg.transAxes,
                    fontsize=size, color=color,
                    fontweight="bold" if bold else "normal",
                    va="center", zorder=5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Page 4: LowConfidence done")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5 — UMAP (existing PNG)
    # ─────────────────────────────────────────────────────────────────────────
    if UMAP_PNG.exists():
        from PIL import Image as PILImage
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")
        img = np.array(PILImage.open(str(UMAP_PNG)))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("UMAP — Cell Type Clustering",
                     fontsize=14, fontweight="bold", color="white", pad=10)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("Page 5: UMAP done")
    else:
        print("Page 5: UMAP not found, skipping")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 6 — Marker Score Heatmap (existing PNG)
    # ─────────────────────────────────────────────────────────────────────────
    if HEATMAP_PNG.exists():
        from PIL import Image as PILImage
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")
        img = np.array(PILImage.open(str(HEATMAP_PNG)))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Marker Score Heatmap by Cell Type",
                     fontsize=14, fontweight="bold", color="white", pad=10)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("Page 6: Heatmap done")
    else:
        print("Page 6: Heatmap not found, skipping")

    # ─────────────────────────────────────────────────────────────────────────
    # PDF metadata
    # ─────────────────────────────────────────────────────────────────────────
    d = pdf.infodict()
    d["Title"]   = "CosMx Cell Type Analysis — AD vs Control"
    d["Author"]  = "CosMx Analysis Pipeline"
    d["Subject"] = "Spatial transcriptomics cell type distribution"
    d["Keywords"] = "CosMx, spatial transcriptomics, AD, cell type, segmentation"

print(f"\nDone → {OUT_PDF}")
