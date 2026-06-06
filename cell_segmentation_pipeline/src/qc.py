"""QC metrics computation and report generation."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .visualization import overlay_mask_on_image


# ── Per-FOV QC ────────────────────────────────────────────────────────────────

def compute_fov_qc(
    cell_table: pd.DataFrame,
    cfg: dict,
    fov_id: str = "",
    model_name: str = "",
) -> dict:
    """Compute scalar QC metrics for a single FOV.

    Args:
        cell_table: Output of mask_to_cell_table().
        cfg:        Full pipeline config dict.
        fov_id:     FOV identifier string.
        model_name: Segmentation model name.

    Returns:
        Dict with per-FOV summary statistics.
    """
    qc_cfg = cfg.get("qc", {})
    min_area = qc_cfg.get("min_cell_area_px", 50)
    max_area = qc_cfg.get("max_cell_area_px", 10000)

    n_total = len(cell_table)
    if n_total == 0:
        return {
            "fov": fov_id, "model": model_name, "n_cells": 0,
            "mean_area_px": None, "median_area_px": None,
            "mean_diam_px": None, "median_diam_px": None,
            "pct_too_small": None, "pct_too_large": None,
        }

    areas = cell_table["area"].values
    diams = cell_table["est_diam_px"].values

    return {
        "fov": fov_id,
        "model": model_name,
        "n_cells": n_total,
        "mean_area_px": float(np.mean(areas)),
        "median_area_px": float(np.median(areas)),
        "mean_diam_px": float(np.mean(diams)),
        "median_diam_px": float(np.median(diams)),
        "pct_too_small": float(np.mean(areas < min_area) * 100),
        "pct_too_large": float(np.mean(areas > max_area) * 100),
    }


def aggregate_qc(results: list[dict]) -> pd.DataFrame:
    """Combine per-FOV QC dicts into a summary DataFrame."""
    return pd.DataFrame(results)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_size_distribution(
    cell_table: pd.DataFrame,
    path: str | Path,
    fov_id: str = "",
    model_name: str = "",
) -> None:
    """Histogram of estimated cell diameters for a single FOV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    diams = cell_table["est_diam_px"]
    areas = cell_table["area"]

    sns.histplot(diams, bins=40, kde=True, ax=axes[0])
    axes[0].set_xlabel("Estimated diameter (px)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{fov_id} {model_name} — diameter distribution")

    sns.histplot(areas, bins=40, kde=True, ax=axes[1])
    axes[1].set_xlabel("Area (px²)")
    axes[1].set_title(f"{fov_id} {model_name} — area distribution")

    plt.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def plot_cellcount_comparison(
    summary_df: pd.DataFrame,
    path: str | Path,
    cosmx_ref: dict[str, int] | None = None,
) -> None:
    """Bar chart comparing cell counts across FOVs and models.

    Args:
        summary_df: Output of aggregate_qc().
        path:       Output PNG path.
        cosmx_ref:  Optional dict {fov_id: cosmx_cell_count} for reference bars.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(summary_df) * 1.2), 5), dpi=150)

    x = np.arange(len(summary_df))
    width = 0.35

    bars = ax.bar(x, summary_df["n_cells"], width, label="pipeline", color="steelblue")

    if cosmx_ref is not None:
        ref_vals = [cosmx_ref.get(fov, 0) for fov in summary_df["fov"]]
        ax.bar(x + width, ref_vals, width, label="CosMx reference", color="orange", alpha=0.7)

    ax.set_xticks(x + width / 2 if cosmx_ref else x)
    ax.set_xticklabels(
        [f"{r['fov']}\n{r.get('model','')}" for _, r in summary_df.iterrows()],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel("Cell count")
    ax.set_title("Cell count per FOV")
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


# ── 4-panel QC image ─────────────────────────────────────────────────────────

def save_qc_panel(
    raw: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    expanded_mask: np.ndarray,
    path: str | Path,
    fov_id: str = "",
    model_name: str = "",
) -> None:
    """Save 2×2 panel: raw / enhanced / mask overlay / expanded overlay.

    Args:
        raw:           Original DAPI channel image (H, W).
        enhanced:      Preprocessed image (H, W).
        mask:          Nuclear label mask (H, W).
        expanded_mask: Expanded label mask (H, W).
        path:          Output PNG path.
        fov_id:        Used in subplot titles.
        model_name:    Used in subplot titles.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_nuc  = int(mask.max())
    n_exp  = int(expanded_mask.max())

    def _norm(img: np.ndarray) -> np.ndarray:
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn) if mx > mn else img.astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=120)
    ax = axes.ravel()

    ax[0].imshow(_norm(raw), cmap="gray")
    ax[0].set_title(f"{fov_id} — raw DAPI", fontsize=10)
    ax[0].axis("off")

    ax[1].imshow(_norm(enhanced), cmap="gray")
    ax[1].set_title(f"{fov_id} — enhanced", fontsize=10)
    ax[1].axis("off")

    ax[2].imshow(overlay_mask_on_image(enhanced, mask, alpha=0.45))
    ax[2].set_title(f"{model_name} — nuclear mask  ({n_nuc} cells)", fontsize=10)
    ax[2].axis("off")

    ax[3].imshow(overlay_mask_on_image(enhanced, expanded_mask, alpha=0.45))
    ax[3].set_title(f"{model_name} — expanded mask ({n_exp} cells)", fontsize=10)
    ax[3].axis("off")

    plt.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
