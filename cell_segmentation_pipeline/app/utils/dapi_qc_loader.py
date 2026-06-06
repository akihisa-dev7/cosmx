"""Streamlit-compatible cached loaders for DAPI QC data.

Can also be imported outside Streamlit (caching is silently skipped).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── Lazy st.cache_data wrapper ────────────────────────────────────────────────

def _cache(fn):
    """Wrap with st.cache_data if Streamlit is available, else identity."""
    try:
        import streamlit as st
        return st.cache_data(show_spinner=False)(fn)
    except Exception:
        return fn


# ── CSV loaders ───────────────────────────────────────────────────────────────

def _load_qc_summary_raw(qc_root: str) -> "pd.DataFrame | None":
    """Load dapi_mask_qc_summary.csv from qc_root."""
    path = Path(qc_root) / "dapi_mask_qc_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(str(path))


load_qc_summary = _cache(_load_qc_summary_raw)


def _load_qc_per_cell_raw(qc_root: str) -> "pd.DataFrame | None":
    """Load dapi_mask_qc_per_cell.csv from qc_root."""
    path = Path(qc_root) / "dapi_mask_qc_per_cell.csv"
    if not path.exists():
        return None
    return pd.read_csv(str(path))


load_qc_per_cell = _cache(_load_qc_per_cell_raw)


# ── On-the-fly QC runner ──────────────────────────────────────────────────────

def _run_fov_qc_raw(
    raw: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    fov_id: str,
    model_name: str,
    min_area_px: int = 50,
    max_area_px: int = 10000,
    iou_threshold: float = 0.3,
) -> dict:
    """Run DAPI QC for a single FOV and return the result dict."""
    import sys
    _utils_dir = Path(__file__).resolve().parents[2]   # cell_segmentation_pipeline/
    _proj_root = _utils_dir.parent                      # CosMx_2026/
    if str(_proj_root) not in sys.path:
        sys.path.insert(0, str(_proj_root))

    from cell_segmentation_pipeline.src.dapi_qc import run_fov_dapi_qc
    return run_fov_dapi_qc(
        raw=raw,
        enhanced=enhanced,
        mask=mask,
        fov_id=fov_id,
        model_name=model_name,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        iou_threshold=iou_threshold,
    )


# st.cache_data cannot handle numpy arrays + dict with DataFrames across
# Streamlit's hash boundary reliably for large arrays, so we use a simple
# session-state memoisation in the page itself. The function below is provided
# as a plain cached wrapper for simpler use-cases.

try:
    import streamlit as st

    @st.cache_data(show_spinner="Running DAPI QC…", max_entries=8)
    def run_fov_qc_cached(
        raw: np.ndarray,
        enhanced: np.ndarray,
        mask: np.ndarray,
        fov_id: str,
        model_name: str,
        min_area_px: int = 50,
        max_area_px: int = 10000,
        iou_threshold: float = 0.3,
    ) -> dict:
        """Cached version of run_fov_dapi_qc (Streamlit context)."""
        return _run_fov_qc_raw(
            raw, enhanced, mask, fov_id, model_name,
            min_area_px, max_area_px, iou_threshold,
        )

except Exception:
    # Outside Streamlit context — just alias the raw function
    run_fov_qc_cached = _run_fov_qc_raw


# ── Visualization loader ──────────────────────────────────────────────────────

def list_vis_images(qc_root: str, fov_id: str, model_name: str) -> dict[str, Path]:
    """Return dict of {description: Path} for pre-computed PNG files."""
    vis_dir = Path(qc_root) / "visualizations"
    prefix  = f"{fov_id}_" if fov_id else ""
    if model_name:
        prefix += f"{model_name}_"

    names = {
        "DAPI + Mask Overlay":      f"{prefix}dapi_mask_overlay.png",
        "DAPI Objects vs Masks":    f"{prefix}dapi_objects_vs_masks.png",
        "Low Confidence Masks":     f"{prefix}low_confidence_masks.png",
        "Missed DAPI Objects":      f"{prefix}missed_dapi_objects.png",
        "Quality Histograms":       f"{prefix}mask_quality_histograms.png",
    }
    return {desc: vis_dir / fname for desc, fname in names.items()}
