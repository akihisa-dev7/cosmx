"""Unified QC metrics: runs both Otsu-based and label-based QC and merges results."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .dapi_qc import run_fov_dapi_qc
from .label_based_qc import (
    CosMxLabelData,
    compare_custom_vs_cosmx,
    load_compartment_labels,
    load_cell_labels,
    load_cell_stats,
    find_per_fov_dir,
)


def run_combined_qc(
    raw:          np.ndarray,
    enhanced:     np.ndarray,
    custom_mask:  np.ndarray,
    fov_name:     str,
    model_name:   str,
    per_fov_root: str,
    # Otsu QC params
    otsu_min_area: int   = 200,
    otsu_iou_thr:  float = 0.3,
    # Label QC params
    match_dist_px: float = 30.0,
    boundary_tol:  int   = 3,
) -> dict:
    """Run both Otsu and CosMx-label QC for one FOV, return combined result.

    Keys in returned dict:
      otsu    : result of run_fov_dapi_qc (dapi_qc.py)
      label   : result of compare_custom_vs_cosmx (label_based_qc.py)
      cosmx   : CosMxLabelData object
      fov_name, model_name
    """
    # ── Otsu-based QC ────────────────────────────────────────────────────────
    otsu_result = run_fov_dapi_qc(
        raw, enhanced, custom_mask,
        fov_id=fov_name,
        model_name=model_name,
        dapi_object_min_area=otsu_min_area,
        iou_threshold=otsu_iou_thr,
    )

    # ── CosMx label QC ───────────────────────────────────────────────────────
    fov_dir = find_per_fov_dir(fov_name, per_fov_root)
    label_result = None
    cosmx_data   = None

    if fov_dir is not None:
        comp  = load_compartment_labels(fov_dir)
        cells = load_cell_labels(fov_dir)
        stats = load_cell_stats(fov_dir)

        if comp is not None and cells is not None:
            cosmx_data   = CosMxLabelData(comp, cells, stats, fov_name)
            label_result = compare_custom_vs_cosmx(
                custom_mask, cosmx_data,
                match_dist_px=match_dist_px,
                boundary_tol=boundary_tol,
            )

    return {
        "fov_name":   fov_name,
        "model_name": model_name,
        "otsu":       otsu_result,
        "label":      label_result,
        "cosmx":      cosmx_data,
    }


def summarise_to_row(combined: dict) -> dict:
    """Flatten the most important metrics into a single dict (one row per FOV)."""
    row: dict = {
        "fov_name":   combined["fov_name"],
        "model_name": combined["model_name"],
    }

    otsu = combined.get("otsu", {})
    skip = {"per_cell_df", "per_dapi_df", "dapi_labels"}
    for k, v in otsu.items():
        if k not in skip and not isinstance(v, (pd.DataFrame, np.ndarray)):
            row[f"otsu_{k}"] = v

    label = combined.get("label")
    if label is not None:
        skip2 = {"per_cell_df", "custom_centroids", "cosmx_centroids", "sanity"}
        for k, v in label.items():
            if k not in skip2 and not isinstance(v, (pd.DataFrame, np.ndarray)):
                row[f"label_{k}"] = v

    return row
