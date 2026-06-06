"""Assign RNA transcripts to custom segmentation cells via mask lookup."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .io import fov_name_to_int, fov_int_to_name, load_mask


def assign_transcripts_to_mask(
    tx_df: pd.DataFrame,
    mask: np.ndarray,
    fov_int: int,
    fov_name: str,
) -> pd.DataFrame:
    """Assign transcripts in one FOV to mask cell labels.

    Uses x_local_px / y_local_px for pixel lookup.
    mask[row=y, col=x] → cell label (0 = background).

    Returns tx_df with added columns: custom_cell_id, assigned (bool).
    """
    df = tx_df[tx_df["fov"] == fov_int].copy()

    # Pixel coordinates (clip to mask bounds)
    h, w = mask.shape
    # Coerce to numeric; non-numeric values become NaN → drop those rows
    df["x_local_px"] = pd.to_numeric(df["x_local_px"], errors="coerce")
    df["y_local_px"] = pd.to_numeric(df["y_local_px"], errors="coerce")
    df = df.dropna(subset=["x_local_px", "y_local_px"])

    xs = df["x_local_px"].values.astype(np.int32)
    ys = df["y_local_px"].values.astype(np.int32)

    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    cell_labels = mask[ys, xs]

    df["custom_cell_id"] = np.where(
        cell_labels > 0,
        fov_name + "_" + cell_labels.astype(str),
        "unassigned",
    )
    df["mask_label"] = cell_labels.astype(np.int32)
    df["assigned"] = cell_labels > 0
    df["fov_name"] = fov_name

    return df


def assign_all_fovs(
    tx_df: pd.DataFrame,
    mask_dir: str,
    fov_names: List[str],
) -> pd.DataFrame:
    """Run mask-lookup assignment for all requested FOVs."""
    from .io import find_mask_file

    results = []
    for fov_name in fov_names:
        fov_int = fov_name_to_int(fov_name)
        mask_path = find_mask_file(mask_dir, fov_name)
        if mask_path is None:
            print(f"  [WARN] No mask found for {fov_name}, skipping")
            continue

        mask = load_mask(mask_path)
        fov_tx = tx_df[tx_df["fov"] == fov_int]
        if len(fov_tx) == 0:
            print(f"  [WARN] No transcripts for {fov_name} (fov={fov_int})")
            continue

        assigned = assign_transcripts_to_mask(fov_tx, mask, fov_int, fov_name)
        n_total = len(assigned)
        n_assigned = assigned["assigned"].sum()
        print(
            f"  {fov_name}: {n_total:,} tx → {n_assigned:,} assigned "
            f"({100*n_assigned/n_total:.1f}%)"
        )
        results.append(assigned)

    if not results:
        raise ValueError("No transcripts assigned for any FOV.")

    return pd.concat(results, ignore_index=True)


def build_counts_matrix(assigned_df: pd.DataFrame) -> pd.DataFrame:
    """Build cell × gene count matrix from assigned transcripts.

    Returns DataFrame with custom_cell_id as index, gene names as columns.
    """
    df = assigned_df[assigned_df["assigned"]].copy()
    pivot = (
        df.groupby(["custom_cell_id", "target"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.index.name = "cell_id"
    return pivot


def assign_native_cells_to_custom_mask(
    native_metadata: pd.DataFrame,
    native_counts: pd.DataFrame,
    mask: np.ndarray,
    fov_int: int,
    fov_name: str,
) -> pd.DataFrame:
    """Map native CosMx cells to custom mask via centroid lookup.

    Used for FOVs where tx_file is not available.
    native_metadata must contain CenterX_local_px, CenterY_local_px, fov, cell_ID.
    native_counts must contain fov, cell_ID, and gene columns.
    Returns cell × gene count DataFrame with custom_cell_id index.
    """
    meta_fov = native_metadata[native_metadata["fov"] == fov_int].copy()
    counts_fov = native_counts[native_counts["fov"] == fov_int].copy()

    if len(meta_fov) == 0 or len(counts_fov) == 0:
        print(f"  [WARN] No native data for {fov_name} (fov={fov_int})")
        return pd.DataFrame()

    h, w = mask.shape
    xs = meta_fov["CenterX_local_px"].values.astype(np.int32)
    ys = meta_fov["CenterY_local_px"].values.astype(np.int32)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    custom_labels = mask[ys, xs]

    meta_fov = meta_fov.copy()
    meta_fov["custom_label"] = custom_labels
    meta_fov["custom_cell_id"] = np.where(
        custom_labels > 0,
        fov_name + "_" + custom_labels.astype(str),
        "unassigned",
    )

    # Build mapping native cell_ID → custom_cell_id
    id_map = meta_fov.set_index("cell_ID")["custom_cell_id"].to_dict()
    counts_fov = counts_fov.copy()
    counts_fov["custom_cell_id"] = counts_fov["cell_ID"].map(id_map).fillna("unassigned")

    # Drop unassigned and system columns
    system_cols = {"fov", "cell_ID", "custom_cell_id"}
    gene_cols = [c for c in counts_fov.columns if c not in system_cols]
    assigned = counts_fov[counts_fov["custom_cell_id"] != "unassigned"]

    if assigned.empty:
        return pd.DataFrame()

    # Aggregate: sum counts per custom_cell_id
    agg = assigned.groupby("custom_cell_id")[gene_cols].sum()
    agg.index.name = "cell_id"

    n_native = len(meta_fov)
    n_assigned = (custom_labels > 0).sum()
    n_custom = (agg > 0).any(axis=1).sum()
    print(
        f"  {fov_name} (native→custom): {n_native} native cells → "
        f"{n_assigned} assigned ({100*n_assigned/n_native:.1f}%) → "
        f"{n_custom} custom cells with counts"
    )
    return agg


def build_assignment_summary(
    assigned_df: pd.DataFrame,
    counts_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-FOV assignment summary statistics."""
    rows = []
    for fov_name in assigned_df["fov_name"].unique():
        sub = assigned_df[assigned_df["fov_name"] == fov_name]
        n_total = len(sub)
        n_assigned = sub["assigned"].sum()

        # Per-cell stats from counts matrix
        fov_cells = [c for c in counts_matrix.index if c.startswith(fov_name + "_")]
        if fov_cells:
            fov_mat = counts_matrix.loc[fov_cells]
            tx_per_cell = fov_mat.sum(axis=1)
            genes_per_cell = (fov_mat > 0).sum(axis=1)
        else:
            tx_per_cell = pd.Series([], dtype=float)
            genes_per_cell = pd.Series([], dtype=float)

        rows.append(
            {
                "fov_name": fov_name,
                "total_transcripts": n_total,
                "assigned_transcripts": int(n_assigned),
                "unassigned_transcripts": int(n_total - n_assigned),
                "assignment_rate": round(n_assigned / n_total, 4) if n_total > 0 else 0,
                "n_cells": len(fov_cells),
                "transcripts_per_cell_median": round(tx_per_cell.median(), 2) if len(tx_per_cell) > 0 else 0,
                "genes_per_cell_median": round(genes_per_cell.median(), 2) if len(genes_per_cell) > 0 else 0,
            }
        )
    return pd.DataFrame(rows)
