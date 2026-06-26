"""Additional SVG exports for FOV00001 — adds to existing folder structure.

New outputs:
  Existing model folders (01-07): + qc_panel.svg, + cell_type_map_na.svg (02-06)
  07_baysor_cpsam:                + baysor_polygon.svg
  09_transcript_density/          NEW — density maps SVG
  10_enhancement_comparison/      NEW — preprocessing methods comparison SVG
  11_zplane_stats/                NEW — Z-plane depth stats SVG
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import tifffile
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR  = PROJECT_ROOT / "outputs" / "pilot_4fov"
MOL_ROOT   = PROJECT_ROOT / "outputs" / "cosmx_molecular_analysis"
SSD        = PROJECT_ROOT / "outputs_ssd"
OUT_ROOT   = PROJECT_ROOT / "outputs" / "FOV00001_svg_export"

PALETTE: dict[str, str] = {
    "Neuron": "#E63946", "Astrocyte": "#457B9D", "Microglia": "#2A9D8F",
    "Oligodendrocyte": "#E9C46A", "OPC": "#F4A261", "Endothelial": "#9B2226",
    "Pericyte_VSMC": "#9D4EDD", "LowConfidence": "#CCCCCC", "Unknown": "#888888",
}


def save_svg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="svg", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path.relative_to(PROJECT_ROOT)}")


def note(fig: plt.Figure, title: str, sub: str = "") -> None:
    fig.suptitle(title, fontsize=11, fontweight="bold", color="white", y=0.98)
    if sub:
        fig.text(0.5, 0.01, sub, ha="center", fontsize=7,
                 color="#888888", style="italic")


def norm_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def embed_png(ax, path: Path, max_dim: int = 900) -> None:
    """Load PNG and display in axes."""
    img = np.array(PILImage.open(str(path)).convert("RGB"))
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        img = np.array(PILImage.fromarray(img).resize((nw, nh), PILImage.LANCZOS))
    ax.imshow(img, aspect="equal")
    ax.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
# 1. QC panels — add to each model folder
# ══════════════════════════════════════════════════════════════════════════════
print("\n[QC panels]")

QC_PANEL_MAP = {
    "02_stardist":              "FOV00001_AD_F_stardist_qc_panel.png",
    "03_stardist_g10":          "FOV00001_AD_F_stardist_g10_qc_panel.png",
    "04_instanseg":             "FOV00001_AD_F_instanseg_qc_panel.png",
    "05_cellpose_cpsam_membrane": "FOV00001_AD_F_cellpose_cpsam_membrane_qc_panel.png",
}
QC_PANELS_DIR = PILOT_DIR / "qc" / "panels"

# cpsam QC panel lives in cell_segmentation_pipeline
cpsam_qc_src = PROJECT_ROOT / "outputs/cell_segmentation_pipeline/qc/panels/FOV00001_AD_F_cellpose_cpsam_qc_panel.png"
if cpsam_qc_src.exists():
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    embed_png(ax, cpsam_qc_src)
    note(fig, "QC Panel — CellPose SAM (cpsam)",
         "FOV00001 (AD)  |  cell size distribution, mask quality")
    save_svg(fig, OUT_ROOT / "01_cpsam" / "qc_panel.svg")

for folder, fname in QC_PANEL_MAP.items():
    src = QC_PANELS_DIR / fname
    if not src.exists():
        print(f"  SKIP (not found): {fname}")
        continue
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    embed_png(ax, src)
    model_name = folder.split("_", 1)[1].replace("_", " ")
    note(fig, f"QC Panel — {model_name}",
         "FOV00001 (AD)  |  cell size distribution, mask quality")
    save_svg(fig, OUT_ROOT / folder / "qc_panel.svg")


# ══════════════════════════════════════════════════════════════════════════════
# 2. cell_type_map — placeholder for models without molecular analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n[cell_type_map placeholders for 02-06]")

NO_CT_FOLDERS = {
    "02_stardist":              "StarDist",
    "03_stardist_g10":          "StarDist g10",
    "04_instanseg":             "InstanSeg",
    "05_cellpose_cpsam_membrane": "CellPose SAM + Membrane",
    "06_cosmx_original":        "CosMx Original (native)",
}

for folder, model_name in NO_CT_FOLDERS.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.axis("off")

    ax.text(0.5, 0.62,
            "Cell Type Map — Not Available",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="#FF6B35", transform=ax.transAxes)
    ax.text(0.5, 0.48,
            f"Model: {model_name}",
            ha="center", va="center", fontsize=12, color="#AAAAAA",
            transform=ax.transAxes)
    ax.text(0.5, 0.36,
            "Cell type annotation requires molecular analysis\n"
            "(transcript assignment + clustering) to be re-run\n"
            "with this model's segmentation mask.",
            ha="center", va="center", fontsize=10, color="#888888",
            transform=ax.transAxes, linespacing=1.6)

    # Show palette for reference
    x0, y0 = 0.15, 0.18
    for i, (ct, color) in enumerate(PALETTE.items()):
        if ct == "Unknown":
            continue
        col = i % 4
        row = i // 4
        xp = x0 + col * 0.20
        yp = y0 - row * 0.07
        rect = mpatches.FancyBboxPatch((xp, yp - 0.025), 0.05, 0.04,
                                        boxstyle="round,pad=0.003",
                                        facecolor=color, edgecolor="white",
                                        linewidth=0.4, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(xp + 0.06, yp - 0.005, ct, transform=ax.transAxes,
                fontsize=6.5, color="#AAAAAA", va="center")

    note(fig, f"Cell Type Map — {model_name}",
         "FOV00001 (AD)  |  re-run molecular analysis to enable this view")
    save_svg(fig, OUT_ROOT / folder / "cell_type_map_na.svg")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Baysor polygon SVG — 07_baysor_cpsam
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Baysor polygons — 07_baysor_cpsam]")

RAW_PATH = PROJECT_ROOT / "outputs_ssd/pilot_segmentation/raw_images/FOV00001_AD_F_raw_original.tif"
poly_path = PROJECT_ROOT / "outputs/cell_segmentation_pipeline/baysor/FOV00001/segmentation_polygons_2d.json"

if poly_path.exists() and RAW_PATH.exists():
    raw = tifffile.imread(str(RAW_PATH))
    if raw.ndim == 3:
        raw = raw[0] if raw.shape[0] < raw.shape[-1] else raw[..., 0]
    gray = norm_uint8(raw)
    H, W = gray.shape

    # Downsample for display
    MAX_DIM = 1024
    scale = MAX_DIM / max(H, W)
    gds = np.array(PILImage.fromarray(gray).resize(
        (int(W * scale), int(H * scale)), PILImage.LANCZOS))

    with open(poly_path) as f:
        geojson = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.imshow(gds, cmap="gray", vmin=0, vmax=255, aspect="equal")

    rng = np.random.default_rng(42)
    for feat in geojson["features"]:
        geom = feat["geometry"]
        if geom["type"] != "Polygon":
            continue
        coords = np.array(geom["coordinates"][0])  # [[x,y], ...]
        xs = coords[:, 0] * scale
        ys = coords[:, 1] * scale
        color = rng.random(3) * 0.7 + 0.3
        ax.fill(xs, ys, alpha=0.25, color=color)
        ax.plot(xs, ys, color=color, linewidth=0.5, alpha=0.9)

    ax.set_xlim(0, gds.shape[1])
    ax.set_ylim(gds.shape[0], 0)
    ax.axis("off")

    n_poly = len(geojson["features"])
    note(fig,
         f"Baysor Polygons — {n_poly} cells (cpsam prior)",
         "FOV00001 (AD)  |  Baysor RNA-guided polygon boundaries on DAPI")
    save_svg(fig, OUT_ROOT / "07_baysor_cpsam" / "baysor_polygon.svg")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Transcript density — 09_transcript_density/
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Transcript density — 09_transcript_density]")

DENSITY_DIR  = SSD / "explore_02_transcript_density"
DENSITY_FOLDER = OUT_ROOT / "09_transcript_density"

# 4a. Multi-sigma density comparison
sigma_files = {
    "σ=0.5 (sharp)": DENSITY_DIR / "FOV00001_density_sigma05.tif",
    "σ=15":          DENSITY_DIR / "FOV00001_density_sigma15.tif",
    "σ=30":          DENSITY_DIR / "FOV00001_density_sigma30.tif",
    "σ=60 (smooth)": DENSITY_DIR / "FOV00001_density_sigma60.tif",
}

valid = {k: v for k, v in sigma_files.items() if v.exists()}
if valid:
    fig, axes = plt.subplots(1, len(valid), figsize=(4 * len(valid), 5))
    fig.patch.set_facecolor("#0D1117")
    if len(valid) == 1:
        axes = [axes]
    for ax, (title, fpath) in zip(axes, valid.items()):
        ax.set_facecolor("#0D1117")
        density = tifffile.imread(str(fpath)).astype(np.float32)
        MAX = 1024
        sc  = MAX / max(density.shape)
        dds = np.array(PILImage.fromarray(density).resize(
            (int(density.shape[1] * sc), int(density.shape[0] * sc)),
            PILImage.LANCZOS))
        im = ax.imshow(dds, cmap="hot", aspect="equal")
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(colors="white")
    note(fig, "Transcript Density Maps — multi-sigma",
         "FOV00001 (AD)  |  RNA transcript spatial density at different smoothing scales")
    save_svg(fig, DENSITY_FOLDER / "transcript_density_sigma.svg")

# 4b. Membrane density
mem_path = DENSITY_DIR / "FOV00001_membrane_density.tif"
if mem_path.exists():
    density = tifffile.imread(str(mem_path)).astype(np.float32)
    sc = 1024 / max(density.shape)
    dds = np.array(PILImage.fromarray(density).resize(
        (int(density.shape[1] * sc), int(density.shape[0] * sc)), PILImage.LANCZOS))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#0D1117")
    # raw DAPI side
    if RAW_PATH.exists():
        raw2 = tifffile.imread(str(RAW_PATH))
        if raw2.ndim == 3:
            raw2 = raw2[0] if raw2.shape[0] < raw2.shape[-1] else raw2[..., 0]
        gds2 = np.array(PILImage.fromarray(norm_uint8(raw2)).resize(
            (dds.shape[1], dds.shape[0]), PILImage.LANCZOS))
        axes[0].imshow(gds2, cmap="gray", aspect="equal")
        axes[0].set_title("DAPI", color="white", fontsize=10)
        axes[0].axis("off")
    im = axes[1].imshow(dds, cmap="plasma", aspect="equal")
    axes[1].set_title("Membrane-proximal transcript density", color="white", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.04).ax.tick_params(colors="white")
    for ax in axes:
        ax.set_facecolor("#0D1117")
    note(fig, "Membrane Transcript Density",
         "FOV00001 (AD)  |  transcript density near predicted cell boundaries")
    save_svg(fig, DENSITY_FOLDER / "membrane_density.svg")

# 4c. Local entropy
ent_path = DENSITY_DIR / "FOV00001_local_entropy.tif"
if ent_path.exists():
    ent = tifffile.imread(str(ent_path)).astype(np.float32)
    sc  = 1024 / max(ent.shape)
    eds = np.array(PILImage.fromarray(ent).resize(
        (int(ent.shape[1] * sc), int(ent.shape[0] * sc)), PILImage.LANCZOS))
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    im = ax.imshow(eds, cmap="viridis", aspect="equal")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Entropy").ax.tick_params(colors="white")
    note(fig, "Local Transcript Entropy",
         "FOV00001 (AD)  |  spatial entropy of RNA composition (high = mixed gene expression)")
    save_svg(fig, DENSITY_FOLDER / "local_entropy.svg")

# 4d. Z-plane density PNG → SVG
zplane_png = DENSITY_DIR / "FOV00001_zplane_density.png"
if zplane_png.exists():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    embed_png(ax, zplane_png)
    note(fig, "Transcript Density per Z-plane",
         "FOV00001 (AD)  |  transcript count distribution across Z depth layers")
    save_svg(fig, DENSITY_FOLDER / "zplane_density.svg")

# 4e. density_vs_nuclear PNG → SVG
dvn_png = DENSITY_DIR / "FOV00001_density_vs_nuclear.png"
if dvn_png.exists():
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    embed_png(ax, dvn_png)
    note(fig, "Transcript Density vs. Nuclear Signal",
         "FOV00001 (AD)  |  correlation between DAPI intensity and local transcript density")
    save_svg(fig, DENSITY_FOLDER / "density_vs_nuclear.svg")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Enhancement comparison — 10_enhancement_comparison/
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Enhancement comparison — 10_enhancement_comparison]")

ENH_DIR    = SSD / "image_enhancement_comparison" / "enhanced_tifs"
ENH_FOLDER = OUT_ROOT / "10_enhancement_comparison"

ENH_METHODS = {
    "Raw ch0 (DAPI)":     PROJECT_ROOT / "outputs_ssd/pilot_segmentation/raw_images/FOV00001_AD_F_raw_original.tif",
    "Raw ch2":            ENH_DIR / "FOV00001_AD_Frontal_raw_ch2.tif",
    "Top-hat":            ENH_DIR / "FOV00001_AD_Frontal_tophat.tif",
    "CLAHE":              ENH_DIR / "FOV00001_AD_Frontal_clahe.tif",
    "Top-hat + CLAHE":    ENH_DIR / "FOV00001_AD_Frontal_tophat_clahe.tif",
    "Top-hat + Gaussian": ENH_DIR / "FOV00001_AD_Frontal_tophat_gauss.tif",
}

valid_enh = {k: v for k, v in ENH_METHODS.items() if Path(v).exists()}
if valid_enh:
    n = len(valid_enh)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    fig.patch.set_facecolor("#0D1117")
    axes_flat = np.array(axes).flatten()

    for ax, (title, fpath) in zip(axes_flat, valid_enh.items()):
        ax.set_facecolor("#111111")
        try:
            img = tifffile.imread(str(fpath))
            if img.ndim == 3:
                img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
            img8 = norm_uint8(img)
            MAX = 700
            sc  = MAX / max(img8.shape)
            ids = np.array(PILImage.fromarray(img8).resize(
                (int(img8.shape[1] * sc), int(img8.shape[0] * sc)),
                PILImage.LANCZOS))
            ax.imshow(ids, cmap="gray", vmin=0, vmax=255, aspect="equal")
            ax.set_title(title, color="white", fontsize=9, pad=4)
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                    color="red", transform=ax.transAxes, fontsize=7)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    note(fig, "Image Enhancement Comparison",
         "FOV00001 (AD)  |  effect of preprocessing on nuclear signal clarity")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_svg(fig, ENH_FOLDER / "enhancement_comparison.svg")

# Enhancement zoom comparison (existing PNG)
zoom_png = SSD / "image_enhancement_comparison/previews/FOV00001_AD_Frontal_zoom_comparison.png"
if zoom_png.exists():
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    embed_png(ax, zoom_png)
    note(fig, "Enhancement Zoom Comparison",
         "FOV00001 (AD)  |  zoomed region showing preprocessing effect on nuclear boundaries")
    save_svg(fig, ENH_FOLDER / "enhancement_zoom.svg")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Z-plane stats — 11_zplane_stats/
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Z-plane stats — 11_zplane_stats]")

ZPLANE_DIR    = SSD / "explore_04_zplane_analysis"
ZPLANE_FOLDER = OUT_ROOT / "11_zplane_stats"

# 6a. Z-plane stats PNG
zstats_png = ZPLANE_DIR / "FOV00001_zplane_stats.png"
if zstats_png.exists():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    embed_png(ax, zstats_png)
    note(fig, "Z-plane Statistics",
         "FOV00001 (AD)  |  transcript count and quality metrics per Z depth")
    save_svg(fig, ZPLANE_FOLDER / "zplane_stats.svg")

# 6b. Transcript count per Z-filter from CSV files
z_csvs = {
    "z0–7 (all)": ZPLANE_DIR / "FOV00001_tx_filtered_z0to7.csv.gz",
    "z1–6":       ZPLANE_DIR / "FOV00001_tx_filtered_z1to6.csv.gz",
    "z2–5 (core)":ZPLANE_DIR / "FOV00001_tx_filtered_z2to5.csv.gz",
}

valid_z = {k: v for k, v in z_csvs.items() if v.exists()}
if valid_z:
    rows_data = []
    for label, fpath in valid_z.items():
        df = pd.read_csv(fpath)
        rows_data.append({
            "Z filter":      label,
            "N transcripts": len(df),
            "N genes":       df["target"].nunique() if "target" in df.columns else 0,
        })
    stats_df = pd.DataFrame(rows_data)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#0D1117")
    colors = ["#4CC9F0", "#F4A261", "#6BCB77"]
    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444444")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    axes[0].bar(stats_df["Z filter"], stats_df["N transcripts"],
                color=colors, edgecolor="#0D1117", linewidth=0.4)
    axes[0].set_title("Transcripts retained per Z filter")
    axes[0].set_ylabel("Count")
    for bar, val in zip(axes[0].patches, stats_df["N transcripts"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                     f"{val:,}", ha="center", fontsize=9, color="white")

    axes[1].bar(stats_df["Z filter"], stats_df["N genes"],
                color=colors, edgecolor="#0D1117", linewidth=0.4)
    axes[1].set_title("Unique genes per Z filter")
    axes[1].set_ylabel("Count")
    for bar, val in zip(axes[1].patches, stats_df["N genes"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f"{val:,}", ha="center", fontsize=9, color="white")

    note(fig, "Z-plane Filter Comparison",
         "FOV00001 (AD)  |  effect of Z-plane range on retained transcripts and gene diversity")
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    save_svg(fig, ZPLANE_FOLDER / "zplane_filter_comparison.svg")


# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("COMPLETE OUTPUT SUMMARY")
print(f"{'='*70}")
for folder in sorted(OUT_ROOT.glob("*/")):
    svgs = sorted(folder.glob("*.svg"))
    tag  = "NEW" if folder.name[0] in "09101112" else ""
    print(f"\n{folder.name}  {tag}")
    for s in svgs:
        print(f"  ✓ {s.name}")

print(f"\nAll SVGs → {OUT_ROOT}")
