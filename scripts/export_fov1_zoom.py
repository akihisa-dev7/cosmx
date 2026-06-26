"""Export FOV00001 high-resolution ROI-zoom SVGs per model.

Unlike export_fov1_svg.py (which downsamples the morphology image to 1024px),
this script crops a region of interest (ROI) and renders it at the *native*
4256px resolution so individual cells stay sharp when zoomed in.

Output structure:
  outputs/FOV00001_zoom_export/
    _locator.svg                 ← overview showing the tile grid + numbers
    01_cpsam/
       tile_00_overlay.svg
       tile_00_celltype.svg
       ...
    07_baysor_cpsam/
       tile_00_overlay.svg
       ...

How to choose what to zoom:
  1. Open _locator.svg to see the FOV split into a numbered grid.
  2. Open the tile_NN_*.svg for the region you want — it is full resolution.

Tuning (edit the CONFIG block):
  GRID      = (rows, cols)  how finely to split the FOV.
  ROIS      = explicit [(x0, y0, x1, y1), ...] crops; if set, overrides GRID.
  RENDER_DPI / margin etc.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tifffile
from skimage import measure

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── CONFIG ─────────────────────────────────────────────────────────────────────
GRID = (2, 2)            # split the FOV into rows x cols tiles
ROIS: list[tuple[int, int, int, int]] = []   # explicit crops (x0,y0,x1,y1); overrides GRID if non-empty
RENDER_DPI = 200         # embedded-image resolution; higher = sharper but bigger file
CONTOUR_LW = 0.5         # contour line width (full-res, so can be thin)
MAX_LABELS_PER_TILE = 4000   # safety cap on contour count per tile

# ── Paths ──────────────────────────────────────────────────────────────────────
PILOT_DIR = PROJECT_ROOT / "outputs" / "pilot_4fov"
MOL_ROOT  = PROJECT_ROOT / "outputs" / "cosmx_molecular_analysis"
OUT_ROOT  = PROJECT_ROOT / "outputs" / "FOV00001_zoom_export"
RAW_PATH  = PROJECT_ROOT / "outputs_ssd/pilot_segmentation/raw_images/FOV00001_AD_F_raw_original.tif"

PALETTE: dict[str, str] = {
    "Neuron": "#E63946", "Astrocyte": "#457B9D", "Microglia": "#2A9D8F",
    "Oligodendrocyte": "#E9C46A", "OPC": "#F4A261", "Endothelial": "#9B2226",
    "Pericyte_VSMC": "#9D4EDD", "LowConfidence": "#CCCCCC", "Unknown": "#888888",
}
ORDERED_TYPES = list(PALETTE.keys())

MODELS = [
    {"folder": "01_cpsam", "label": "CellPose SAM (cpsam)",
     "mask": PILOT_DIR / "expanded_masks/FOV00001_AD_F_cpsam_expanded_mask.tif", "has_ct": True},
    {"folder": "02_stardist", "label": "StarDist",
     "mask": PILOT_DIR / "expanded_masks/FOV00001_AD_F_stardist_expanded_mask.tif", "has_ct": False},
    {"folder": "03_stardist_g10", "label": "StarDist g10",
     "mask": PILOT_DIR / "expanded_masks/FOV00001_AD_F_stardist_g10_expanded_mask.tif", "has_ct": False},
    {"folder": "04_instanseg", "label": "InstanSeg",
     "mask": PILOT_DIR / "expanded_masks/FOV00001_AD_F_instanseg_expanded_mask.tif", "has_ct": False},
    {"folder": "05_cellpose_cpsam_membrane", "label": "CellPose SAM + Membrane",
     "mask": PILOT_DIR / "expanded_masks/FOV00001_AD_F_cellpose_cpsam_membrane_expanded_mask.tif", "has_ct": False},
    {"folder": "06_cosmx_original", "label": "CosMx Original (native)",
     "mask": PILOT_DIR / "masks/FOV00001_AD_F_cosmx_nuclear_mask.tif", "has_ct": False},
    {"folder": "07_baysor_cpsam", "label": "Baysor (cpsam prior)",
     "mask": PILOT_DIR / "baysor/FOV00001/cpsam/baysor_mask.tif", "has_ct": False},
]


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


def tiles_from_grid(h: int, w: int, grid: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    rows, cols = grid
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append((xs[c], ys[r], xs[c + 1], ys[r + 1]))
    return out


def hex_to_rgb(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def fig_for_crop(crop_w: int, crop_h: int):
    """Figure sized so the embedded raster keeps ~native pixels."""
    fw, fh = crop_w / RENDER_DPI, crop_h / RENDER_DPI
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=RENDER_DPI)
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def draw_contours(ax, mask_crop: np.ndarray, color: str) -> None:
    labels = np.unique(mask_crop)
    labels = labels[labels != 0]
    if labels.size > MAX_LABELS_PER_TILE:
        print(f"    WARN: {labels.size} labels in tile, capping at {MAX_LABELS_PER_TILE}")
        labels = labels[:MAX_LABELS_PER_TILE]
    for lbl in labels:
        for cnt in measure.find_contours((mask_crop == lbl).astype(float), 0.5):
            ax.plot(cnt[:, 1], cnt[:, 0], color=color, linewidth=CONTOUR_LW, alpha=0.85)


def save_svg(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    → {path.relative_to(PROJECT_ROOT)}")


# ── Load shared data ───────────────────────────────────────────────────────────
print("Loading raw image...")
gray = load_raw()
H, W = gray.shape
print(f"Raw image: {W}x{H}")

TILES = ROIS if ROIS else tiles_from_grid(H, W, GRID)
print(f"{len(TILES)} ROI tiles")

# cpsam cell-type lookup
label_to_ct: dict[int, str] = {}
ct_csv = MOL_ROOT / "cell_type_table.csv"
if ct_csv.exists():
    ct_table = pd.read_csv(ct_csv)
    fov1_ct = ct_table[ct_table["fov_name"] == "FOV00001"]
    for _, row in fov1_ct.iterrows():
        try:
            label_to_ct[int(str(row["cell_id"]).rsplit("_", 1)[-1])] = str(row["predicted_cell_type"])
        except ValueError:
            pass


# ── Locator overview ───────────────────────────────────────────────────────────
print("\n[locator] overview with tile grid")
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#0D1117"); ax.axis("off")
from PIL import Image as PILImage
small = np.array(PILImage.fromarray(gray).resize((1024, int(1024 * H / W)), PILImage.LANCZOS))
sx, sy = 1024 / W, (1024 * H / W) / H
ax.imshow(small, cmap="gray", vmin=0, vmax=255, aspect="equal")
for i, (x0, y0, x1, y1) in enumerate(TILES):
    ax.add_patch(mpatches.Rectangle((x0 * sx, y0 * sy), (x1 - x0) * sx, (y1 - y0) * sy,
                                    fill=False, edgecolor="#00FF88", linewidth=1.2))
    ax.text((x0 + x1) / 2 * sx, (y0 + y1) / 2 * sy, f"{i:02d}",
            color="#FFD166", fontsize=14, fontweight="bold", ha="center", va="center")
fig.suptitle("FOV00001 — ROI tile locator", color="white", fontsize=12, fontweight="bold")
save_svg(fig, OUT_ROOT / "_locator.svg")


# ── Per-model, per-tile ────────────────────────────────────────────────────────
for m in MODELS:
    mpath = Path(m["mask"])
    print(f"\n[{m['folder']}] {m['label']}")
    if not mpath.exists():
        print(f"  WARN: mask not found: {mpath}")
        continue
    mask = tifffile.imread(str(mpath)).astype(np.int32)
    if mask.shape != (H, W):
        print(f"  resizing mask {mask.shape} -> {(H, W)}")
        mask = np.array(PILImage.fromarray(mask.astype(np.int32)).resize((W, H), PILImage.NEAREST))

    for i, (x0, y0, x1, y1) in enumerate(TILES):
        g = gray[y0:y1, x0:x1]
        mc = mask[y0:y1, x0:x1]
        ch, cw = g.shape

        # overlay: DAPI + contours
        fig, ax = fig_for_crop(cw, ch)
        ax.imshow(g, cmap="gray", vmin=0, vmax=255, aspect="equal")
        draw_contours(ax, mc, "#00FF88")
        save_svg(fig, OUT_ROOT / m["folder"] / f"tile_{i:02d}_overlay.svg")

        # cell-type overlay (cpsam only)
        if m.get("has_ct") and label_to_ct:
            base = np.stack([g, g, g], axis=-1).astype(np.float32)
            ct_rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
            for lbl in np.unique(mc):
                if lbl == 0:
                    continue
                ct = label_to_ct.get(int(lbl))
                if ct:
                    ct_rgb[mc == lbl] = hex_to_rgb(PALETTE.get(ct, PALETTE["Unknown"]))
            fg = ct_rgb.any(axis=-1)
            out = base.copy()
            out[fg] = 0.45 * base[fg] + 0.55 * ct_rgb[fg].astype(np.float32)
            fig, ax = fig_for_crop(cw, ch)
            ax.imshow(np.clip(out, 0, 255).astype(np.uint8), aspect="equal")
            draw_contours(ax, mc, "#FFFFFF")
            save_svg(fig, OUT_ROOT / m["folder"] / f"tile_{i:02d}_celltype.svg")

print(f"\nDone. Output: {OUT_ROOT}")
print("Open _locator.svg first, then the tile_NN_*.svg for the region you want.")
