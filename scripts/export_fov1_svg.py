"""Export FOV00001 segmentation results to SVG per model.

Output structure:
  outputs/FOV00001_svg_export/
    01_cpsam/
    02_stardist/
    03_stardist_g10/
    04_instanseg/
    05_cellpose_cpsam_membrane/
    06_cosmx_original/
    07_baysor_cpsam/
    08_baysor_stardist_g10/

Each folder contains the SVGs that can be produced from available data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import tifffile
from skimage import measure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "cell_segmentation_pipeline" / "app"))

# ── Paths ──────────────────────────────────────────────────────────────────────
PILOT_DIR = PROJECT_ROOT / "outputs" / "pilot_4fov"
MOL_ROOT  = PROJECT_ROOT / "outputs" / "cosmx_molecular_analysis"
OUT_ROOT  = PROJECT_ROOT / "outputs" / "FOV00001_svg_export"

RAW_PATH      = PROJECT_ROOT / "outputs_ssd/pilot_segmentation/raw_images/FOV00001_AD_F_raw_original.tif"
ENHANCED_PATH = PILOT_DIR / "enhanced" / "FOV00001_AD_F_tophat_enhanced.tif"

# ── Cell type palette ──────────────────────────────────────────────────────────
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
ORDERED_TYPES = list(PALETTE.keys())


# ── Helpers ────────────────────────────────────────────────────────────────────
def norm_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def load_raw() -> np.ndarray:
    raw = tifffile.imread(str(RAW_PATH))
    if raw.ndim == 3:
        raw = raw[0] if raw.shape[0] < raw.shape[-1] else raw[..., 0]
    return norm_uint8(raw)


def load_mask(path: Path) -> np.ndarray:
    m = tifffile.imread(str(path)).astype(np.int32)
    return m


def random_color_mask(mask: np.ndarray) -> np.ndarray:
    """Assign a random color to each label (for generic mask visualization)."""
    rng = np.random.default_rng(42)
    n = int(mask.max()) + 1
    colors = rng.integers(40, 230, size=(n, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # background black
    rgb = colors[mask]
    return rgb


def contour_overlay(gray: np.ndarray, mask: np.ndarray,
                    color: str = "#00FF88", lw: float = 0.4) -> None:
    """Draw cell contours on current matplotlib axes."""
    # Downsample for speed
    scale = 1024 / max(gray.shape)
    from PIL import Image as PILImage
    gds = np.array(PILImage.fromarray(gray).resize(
        (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
        PILImage.LANCZOS
    ))
    mds = np.array(PILImage.fromarray(mask.astype(np.uint16)).resize(
        (int(mask.shape[1] * scale), int(mask.shape[0] * scale)),
        PILImage.NEAREST
    ))
    plt.imshow(gds, cmap="gray", vmin=0, vmax=255, aspect="equal")
    # Draw contours for each cell (limit to avoid SVG bloat)
    for lbl in np.unique(mds):
        if lbl == 0:
            continue
        contours = measure.find_contours(mds == lbl, 0.5)
        for cnt in contours:
            plt.plot(cnt[:, 1], cnt[:, 0], color=color, linewidth=lw, alpha=0.85)


def save_svg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="svg", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path.relative_to(PROJECT_ROOT)}")


def make_title_note(fig: plt.Figure, title: str, note: str = "") -> None:
    fig.suptitle(title, fontsize=11, fontweight="bold", color="white",
                 y=0.98)
    if note:
        fig.text(0.5, 0.01, note, ha="center", fontsize=7,
                 color="#888888", style="italic")


# ── Load shared data ───────────────────────────────────────────────────────────
print("Loading raw image and cell type table...")
gray = load_raw()
ct_table = pd.read_csv(MOL_ROOT / "cell_type_table.csv")
fov1_ct  = ct_table[ct_table["fov_name"] == "FOV00001"].copy()
# label → cell_type dict
label_to_ct = {}
for _, row in fov1_ct.iterrows():
    try:
        lbl = int(str(row["cell_id"]).rsplit("_", 1)[-1])
        label_to_ct[lbl] = str(row["predicted_cell_type"])
    except ValueError:
        pass


def cell_type_rgb(mask: np.ndarray) -> np.ndarray:
    """Build cell-type colored RGB from mask using label_to_ct."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for lbl, ct in label_to_ct.items():
        color = PALETTE.get(ct, PALETTE["Unknown"])
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        rgb[mask == lbl] = [r, g, b]
    return rgb


def blend_ct(gray: np.ndarray, ct_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    fg   = ct_rgb.any(axis=-1)
    out  = base.copy()
    out[fg] = (1 - alpha) * base[fg] + alpha * ct_rgb[fg].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def downsample(arr: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    from PIL import Image as PILImage
    h, w = arr.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return arr
    nw, nh = int(w * scale), int(h * scale)
    if arr.ndim == 2:
        return np.array(PILImage.fromarray(arr.astype(np.uint16)).resize((nw, nh), PILImage.NEAREST))
    else:
        return np.array(PILImage.fromarray(arr.astype(np.uint8)).resize((nw, nh), PILImage.NEAREST))


def ct_legend_patches() -> list[mpatches.Patch]:
    return [mpatches.Patch(color=PALETTE[t], label=t) for t in ORDERED_TYPES if t != "Unknown"]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
MODELS = [
    {
        "folder":    "01_cpsam",
        "label":     "CellPose SAM (cpsam)",
        "mask_path": PILOT_DIR / "expanded_masks" / "FOV00001_AD_F_cpsam_expanded_mask.tif",
        "has_ct":    True,
        "has_baysor":True,
        "baysor_mask": PILOT_DIR / "baysor/FOV00001/cpsam/baysor_mask.tif",
    },
    {
        "folder":    "02_stardist",
        "label":     "StarDist",
        "mask_path": PILOT_DIR / "expanded_masks" / "FOV00001_AD_F_stardist_expanded_mask.tif",
        "has_ct":    False,
        "has_baysor":False,
    },
    {
        "folder":    "03_stardist_g10",
        "label":     "StarDist g10",
        "mask_path": PILOT_DIR / "expanded_masks" / "FOV00001_AD_F_stardist_g10_expanded_mask.tif",
        "has_ct":    False,
        "has_baysor":True,
        "baysor_mask": PILOT_DIR / "baysor/FOV00001/stardist_g10_s025/segmentation_cell_stats.csv",
    },
    {
        "folder":    "04_instanseg",
        "label":     "InstanSeg",
        "mask_path": PILOT_DIR / "expanded_masks" / "FOV00001_AD_F_instanseg_expanded_mask.tif",
        "has_ct":    False,
        "has_baysor":False,
    },
    {
        "folder":    "05_cellpose_cpsam_membrane",
        "label":     "CellPose SAM + Membrane",
        "mask_path": PILOT_DIR / "expanded_masks" / "FOV00001_AD_F_cellpose_cpsam_membrane_expanded_mask.tif",
        "has_ct":    False,
        "has_baysor":False,
    },
    {
        "folder":    "06_cosmx_original",
        "label":     "CosMx Original (native)",
        "mask_path": PILOT_DIR / "masks" / "FOV00001_AD_F_cosmx_nuclear_mask.tif",
        "has_ct":    False,
        "has_baysor":False,
    },
    {
        "folder":    "07_baysor_cpsam",
        "label":     "Baysor (cpsam prior)",
        "mask_path": PILOT_DIR / "baysor/FOV00001/cpsam/baysor_mask.tif",
        "has_ct":    False,
        "has_baysor":False,
        "is_baysor": True,
        "baysor_stats": PILOT_DIR / "baysor/FOV00001/cpsam/segmentation_cell_stats.csv",
    },
    {
        "folder":    "08_baysor_stardist_g10",
        "label":     "Baysor (stardist_g10 prior)",
        "mask_path": None,   # no mask tif, use cell_stats only
        "has_ct":    False,
        "has_baysor":False,
        "is_baysor": True,
        "baysor_stats": PILOT_DIR / "baysor/FOV00001/stardist_g10_s025/segmentation_cell_stats.csv",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Generate per-model outputs
# ══════════════════════════════════════════════════════════════════════════════
for m in MODELS:
    folder    = OUT_ROOT / m["folder"]
    label     = m["label"]
    mask_path = m.get("mask_path")
    print(f"\n[{m['folder']}] {label}")

    # Load mask if available
    mask = None
    if mask_path and Path(mask_path).exists():
        mask = load_mask(Path(mask_path))
    elif mask_path:
        print(f"  WARN: mask not found: {mask_path}")

    # ── 1. segment_mask.svg ────────────────────────────────────────────────
    if mask is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")
        cm = random_color_mask(mask)
        ax.imshow(downsample(cm, 1024), aspect="equal")
        ax.axis("off")
        n_cells = int((mask > 0).any()) and int(np.unique(mask[mask > 0]).shape[0])
        make_title_note(fig,
                        f"Segment Mask — {label}",
                        f"FOV00001 (AD)  |  {n_cells} cells  |  model: {label}")
        save_svg(fig, folder / "segment_mask.svg")

    # ── 2. overlay.svg ─────────────────────────────────────────────────────
    if mask is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")
        ax.axis("off")

        gds = downsample(gray, 1024)
        mds = downsample(mask, 1024)

        ax.imshow(gds, cmap="gray", vmin=0, vmax=255, aspect="equal")
        for lbl_val in np.unique(mds):
            if lbl_val == 0:
                continue
            contours = measure.find_contours((mds == lbl_val).astype(float), 0.5)
            for cnt in contours:
                ax.plot(cnt[:, 1], cnt[:, 0], color="#00FF88",
                        linewidth=0.35, alpha=0.8)

        make_title_note(fig,
                        f"DAPI + Mask Overlay — {label}",
                        f"FOV00001 (AD)  |  DAPI (gray) + cell boundaries (green)  |  model: {label}")
        save_svg(fig, folder / "overlay.svg")

    # ── 3. cell_type_map.svg (cpsam only) ─────────────────────────────────
    if m.get("has_ct") and mask is not None:
        ct_rgb  = cell_type_rgb(mask)
        blended = blend_ct(downsample(gray, 1024),
                           downsample(ct_rgb, 1024), alpha=0.55)

        fig, (ax_main, ax_leg) = plt.subplots(1, 2, figsize=(10, 8),
                                               gridspec_kw={"width_ratios": [5, 1]})
        fig.patch.set_facecolor("#0D1117")
        for ax in [ax_main, ax_leg]:
            ax.set_facecolor("#0D1117")

        ax_main.imshow(blended, aspect="equal")
        ax_main.axis("off")

        ax_leg.axis("off")
        ax_leg.set_title("Cell types", color="white", fontsize=9, pad=4)
        counts = fov1_ct["predicted_cell_type"].value_counts()
        y = 0.95
        for ct in ORDERED_TYPES:
            if ct == "Unknown":
                continue
            color = PALETTE[ct]
            cnt   = int(counts.get(ct, 0))
            rect = mpatches.FancyBboxPatch(
                (0.05, y - 0.025), 0.25, 0.04,
                boxstyle="round,pad=0.005",
                facecolor=color, edgecolor="white", linewidth=0.4,
                transform=ax_leg.transAxes
            )
            ax_leg.add_patch(rect)
            ax_leg.text(0.37, y - 0.005, f"{ct}\n({cnt})",
                        transform=ax_leg.transAxes,
                        fontsize=7, color="white", va="center")
            y -= 0.10

        make_title_note(fig,
                        f"Cell Type Map — {label}",
                        f"FOV00001 (AD)  |  509 cells annotated  |  "
                        f"model: {label}  |  ⚠ Work in progress")
        save_svg(fig, folder / "cell_type_map.svg")

    # ── 4. baysor_mask.svg (cpsam baysor) ─────────────────────────────────
    if m.get("has_baysor") and m.get("baysor_mask"):
        bpath = Path(m["baysor_mask"])
        if bpath.suffix == ".tif" and bpath.exists():
            bmask = load_mask(bpath)
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            fig.patch.set_facecolor("#0D1117")
            for ax in axes:
                ax.set_facecolor("#0D1117")
                ax.axis("off")

            # Left: baysor mask colored
            axes[0].imshow(downsample(random_color_mask(bmask), 1024), aspect="equal")
            axes[0].set_title(f"Baysor mask  ({bmask.max()} cells)",
                              color="#8AB4F8", fontsize=10)

            # Right: baysor overlay on DAPI
            gds = downsample(gray, 1024)
            mds = downsample(bmask, 1024)
            axes[1].imshow(gds, cmap="gray", vmin=0, vmax=255, aspect="equal")
            for lbl_val in np.unique(mds):
                if lbl_val == 0:
                    continue
                contours = measure.find_contours((mds == lbl_val).astype(float), 0.5)
                for cnt in contours:
                    axes[1].plot(cnt[:, 1], cnt[:, 0], color="#FFD166",
                                 linewidth=0.35, alpha=0.8)
            axes[1].set_title("Baysor overlay on DAPI",
                               color="#FFD166", fontsize=10)

            make_title_note(fig,
                            f"Baysor Segmentation — {label} prior",
                            f"FOV00001 (AD)  |  Baysor RNA-guided boundary refinement")
            save_svg(fig, folder / "baysor_seg.svg")

    # ── 5. baysor_cell_stats.svg (scatter: n_transcripts vs area) ─────────
    bstats_path = m.get("baysor_stats")
    if bstats_path and Path(bstats_path).exists():
        bstats = pd.read_csv(bstats_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0D1117")
        for ax in axes:
            ax.set_facecolor("#111827")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444444")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")

        # n_transcripts histogram
        if "n_transcripts" in bstats.columns:
            axes[0].hist(bstats["n_transcripts"], bins=40,
                         color="#4CC9F0", edgecolor="#0D1117", linewidth=0.3)
            axes[0].set_xlabel("Transcripts per cell")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Transcripts per cell (Baysor)")

        # area histogram
        area_col = next((c for c in ["area", "cell_area", "nucleus_area"]
                         if c in bstats.columns), None)
        if area_col:
            axes[1].hist(bstats[area_col], bins=40,
                         color="#F4A261", edgecolor="#0D1117", linewidth=0.3)
            axes[1].set_xlabel(f"{area_col} (px²)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Cell area distribution (Baysor)")

        make_title_note(fig,
                        f"Baysor Cell Stats — {label} prior",
                        f"FOV00001 (AD)  |  {len(bstats)} cells")
        save_svg(fig, folder / "baysor_cell_stats.svg")

print(f"\nDone. Output: {OUT_ROOT}")


# ══════════════════════════════════════════════════════════════════════════════
# Report: what was generated and what can be added
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY — what was generated per folder")
print("="*70)
for folder in sorted(OUT_ROOT.glob("*/")):
    svgs = sorted(folder.glob("*.svg"))
    print(f"\n{folder.name}:")
    for s in svgs:
        print(f"  ✓ {s.name}")
    if not svgs:
        print("  (no output)")
