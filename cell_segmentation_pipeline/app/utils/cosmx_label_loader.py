"""Streamlit-cached loaders for CosMx per_fov_decoded data."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parents[3]
PER_FOV_ROOT   = PROJECT_ROOT / "raw_data" / "pilot_4fov" / "slide1_RNA" / "per_fov_decoded"
DEFAULT_SEG_ROOT = str(PROJECT_ROOT / "outputs" / "pilot_4fov")


def _cache_data(fn):
    try:
        import streamlit as st
        return st.cache_data(show_spinner=False)(fn)
    except Exception:
        return fn


def _cache_resource(fn):
    try:
        import streamlit as st
        return st.cache_resource(show_spinner=False)(fn)
    except Exception:
        return fn


# ── Per-FOV directory helper ───────────────────────────────────────────────────

def get_fov_dir(fov_name: str, per_fov_root: str | None = None) -> Optional[Path]:
    root = Path(per_fov_root) if per_fov_root else PER_FOV_ROOT
    d = root / fov_name
    return d if d.exists() else None


def list_available_fovs(per_fov_root: str | None = None) -> list[str]:
    root = Path(per_fov_root) if per_fov_root else PER_FOV_ROOT
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("FOV"))


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_compartment_raw(fov_name: str, per_fov_root: str) -> Optional[np.ndarray]:
    d = get_fov_dir(fov_name, per_fov_root)
    if d is None:
        return None
    import tifffile
    matches = sorted(d.glob("CompartmentLabels_*.tif"))
    return tifffile.imread(str(matches[0])).astype(np.uint8) if matches else None


def _load_cell_labels_raw(fov_name: str, per_fov_root: str) -> Optional[np.ndarray]:
    d = get_fov_dir(fov_name, per_fov_root)
    if d is None:
        return None
    import tifffile
    matches = sorted(d.glob("CellLabels_*.tif"))
    return tifffile.imread(str(matches[0])).astype(np.int32) if matches else None


def _load_cell_stats_raw(fov_name: str, per_fov_root: str) -> Optional[pd.DataFrame]:
    d = get_fov_dir(fov_name, per_fov_root)
    if d is None:
        return None
    matches = sorted(d.glob("Run_*_Cell_Stats_*.csv"))
    return pd.read_csv(str(matches[0])) if matches else None


load_compartment   = _cache_data(_load_compartment_raw)
load_cell_labels_c = _cache_data(_load_cell_labels_raw)
load_cell_stats_c  = _cache_data(_load_cell_stats_raw)


# ── Combined: build CosMxLabelData ────────────────────────────────────────────

@_cache_resource
def load_cosmx_label_data(fov_name: str, per_fov_root: str):
    """Return CosMxLabelData or None."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from cell_segmentation_pipeline.src.label_based_qc import CosMxLabelData

    comp  = _load_compartment_raw(fov_name, per_fov_root)
    cells = _load_cell_labels_raw(fov_name, per_fov_root)
    stats = _load_cell_stats_raw(fov_name, per_fov_root)
    if comp is None or cells is None:
        return None
    return CosMxLabelData(comp, cells, stats, fov_name)


# ── Comparison results (heavy, cached) ────────────────────────────────────────

@_cache_data
def run_comparison_cached(
    fov_name:      str,
    model_name:    str,
    per_fov_root:  str,
    seg_root:      str,
    match_dist_px: float,
    boundary_tol:  int,
):
    """Run label-based comparison for one FOV.  Cached by all params."""
    import sys, glob as _glob
    sys.path.insert(0, str(PROJECT_ROOT))
    from cell_segmentation_pipeline.src.label_based_qc import compare_custom_vs_cosmx
    import tifffile

    cosmx = load_cosmx_label_data(fov_name, per_fov_root)
    if cosmx is None:
        return None

    mask_files = _glob.glob(
        f"{seg_root}/masks/{fov_name}*{model_name}*nuclear_mask.tif"
    )
    if not mask_files:
        return None
    custom_mask = tifffile.imread(mask_files[0]).astype(np.int32)

    return compare_custom_vs_cosmx(
        custom_mask, cosmx,
        match_dist_px=match_dist_px,
        boundary_tol=boundary_tol,
    )


# ── Colour helpers ─────────────────────────────────────────────────────────────

LAYER_COLORS: dict[str, tuple[int, int, int]] = {
    "cosmx_nucleus":  (70,  130, 230),   # blue
    "cosmx_cell":     (50,  200,  80),   # green
    "cosmx_membrane": (160,  50, 200),   # purple
    "custom_mask":    (230,  57,  70),   # red
    "overlap":        (255, 215,   0),   # yellow
    "diff_fn":        (255, 140,   0),   # orange (missed by custom)
    "diff_fp":        (200,  80, 200),   # magenta (extra in custom)
    "otsu_mask":      (100, 200, 200),   # cyan
}


def build_rgb_overlay(
    gray_uint8:    np.ndarray,
    layers:        dict[str, np.ndarray],  # name → bool/uint8 mask
    colors:        dict[str, tuple[int, int, int]] | None = None,
    alpha:         float = 0.5,
    boundary_only: bool  = False,
    dilate_px:     int   = 1,
) -> np.ndarray:
    """Blend multiple labelled layers onto a grayscale background.

    Parameters
    ----------
    gray_uint8 : 2D uint8
    layers     : {layer_name: bool mask}
    colors     : {layer_name: (R,G,B)}  (uses LAYER_COLORS as fallback)
    alpha      : opacity for non-zero mask pixels
    boundary_only : if True, only show boundary pixels (XOR erosion)
    dilate_px  : dilation for display boundaries (0 = none)
    """
    from cell_segmentation_pipeline.src.label_based_qc import extract_boundary

    if colors is None:
        colors = LAYER_COLORS

    base = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1).astype(np.float32)
    out  = base.copy()

    for name, mask in layers.items():
        if mask is None or not mask.any():
            continue
        bm = mask.astype(bool)
        if boundary_only:
            bm = extract_boundary(bm, dilate=dilate_px)
        if not bm.any():
            continue
        c = colors.get(name, (200, 200, 200))
        col = np.array(c, dtype=np.float32)
        out[bm] = (1 - alpha) * out[bm] + alpha * col

    return np.clip(out, 0, 255).astype(np.uint8)
