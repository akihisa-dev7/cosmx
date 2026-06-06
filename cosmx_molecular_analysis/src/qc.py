"""QC filtering and visualization for CosMx RNA data."""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("anndata and matplotlib are required.")


QC_DEFAULTS = {
    "min_transcripts_per_cell": 5,
    "max_transcripts_per_cell": 5000,
    "min_genes_per_cell": 3,
    "max_negative_probe_fraction": 0.10,
    "min_cell_area_um2": 10,
    "max_cell_area_um2": 8000,
}


def compute_qc_metrics(adata: "ad.AnnData") -> "ad.AnnData":
    """Add QC metrics to adata.obs if not already present."""
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        total = np.array(X.sum(axis=1)).flatten()
        n_genes = np.array((X > 0).sum(axis=1)).flatten()
    else:
        total = X.sum(axis=1)
        n_genes = (X > 0).sum(axis=1)

    adata.obs["n_transcripts"] = total.astype(int)
    adata.obs["n_genes"] = n_genes.astype(int)

    # Negative probe fraction
    neg_cols = [g for g in adata.var_names if g.startswith("NegPrb") or g.startswith("FalseCode")]
    if neg_cols and sp.issparse(X):
        neg_idx = [adata.var_names.get_loc(g) for g in neg_cols if g in adata.var_names]
        neg_counts = np.array(X[:, neg_idx].sum(axis=1)).flatten()
    elif neg_cols:
        neg_idx = [list(adata.var_names).index(g) for g in neg_cols if g in adata.var_names]
        neg_counts = X[:, neg_idx].sum(axis=1)
    else:
        neg_counts = np.zeros(adata.n_obs)

    total_safe = np.where(total > 0, total, 1)
    adata.obs["neg_probe_fraction"] = neg_counts / total_safe

    return adata


def filter_cells(
    adata: "ad.AnnData",
    min_transcripts: int = QC_DEFAULTS["min_transcripts_per_cell"],
    max_transcripts: int = QC_DEFAULTS["max_transcripts_per_cell"],
    min_genes: int = QC_DEFAULTS["min_genes_per_cell"],
    max_neg_fraction: float = QC_DEFAULTS["max_negative_probe_fraction"],
    min_area_um2: float = QC_DEFAULTS["min_cell_area_um2"],
    max_area_um2: float = QC_DEFAULTS["max_cell_area_um2"],
) -> tuple:
    """Apply QC filters. Returns (filtered_adata, qc_summary_df)."""
    adata = compute_qc_metrics(adata)

    mask = pd.Series(True, index=adata.obs_names)

    mask &= adata.obs["n_transcripts"] >= min_transcripts
    mask &= adata.obs["n_transcripts"] <= max_transcripts
    mask &= adata.obs["n_genes"] >= min_genes
    mask &= adata.obs["neg_probe_fraction"] <= max_neg_fraction

    if "cell_area_um2" in adata.obs.columns:
        valid_area = adata.obs["cell_area_um2"].notna()
        mask &= ~valid_area | (
            (adata.obs["cell_area_um2"] >= min_area_um2)
            & (adata.obs["cell_area_um2"] <= max_area_um2)
        )

    n_before = adata.n_obs
    adata_filtered = adata[mask.values].copy()
    n_after = adata_filtered.n_obs

    summary = pd.DataFrame(
        [
            {
                "metric": "cells_before_qc",
                "value": n_before,
            },
            {
                "metric": "cells_after_qc",
                "value": n_after,
            },
            {
                "metric": "cells_removed",
                "value": n_before - n_after,
            },
            {
                "metric": "retention_rate",
                "value": round(n_after / n_before, 4) if n_before > 0 else 0,
            },
            {
                "metric": "median_transcripts_post_qc",
                "value": round(adata_filtered.obs["n_transcripts"].median(), 2),
            },
            {
                "metric": "median_genes_post_qc",
                "value": round(adata_filtered.obs["n_genes"].median(), 2),
            },
        ]
    )
    return adata_filtered, summary


def save_qc_plots(adata: "ad.AnnData", output_dir: str) -> None:
    """Save QC histogram plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    adata = compute_qc_metrics(adata)

    _hist(
        adata.obs["n_transcripts"],
        "Transcripts per cell",
        Path(output_dir) / "transcripts_per_cell_hist.png",
    )
    _hist(
        adata.obs["n_genes"],
        "Genes per cell",
        Path(output_dir) / "genes_per_cell_hist.png",
    )

    if "cell_area_um2" in adata.obs.columns:
        _hist(
            adata.obs["cell_area_um2"].dropna(),
            "Cell area (µm²)",
            Path(output_dir) / "cell_area_hist.png",
        )

    if "fov_name" in adata.obs.columns and "assigned" in adata.obs.columns:
        _assignment_rate_by_fov(adata, Path(output_dir) / "assignment_rate_by_fov.png")
    elif "fov_name" in adata.obs.columns:
        _tx_by_fov(adata, Path(output_dir) / "transcripts_by_fov.png")


def _hist(series: pd.Series, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series.dropna(), bins=50, edgecolor="none", alpha=0.8, color="#4C72B0")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Cells", fontsize=12)
    ax.set_title(xlabel, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _assignment_rate_by_fov(adata: "ad.AnnData", path: Path) -> None:
    rates = adata.obs.groupby("fov_name")["assigned"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    rates.plot.bar(ax=ax, color="#4C72B0")
    ax.set_ylabel("Assignment rate")
    ax.set_title("Assignment rate by FOV")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _tx_by_fov(adata: "ad.AnnData", path: Path) -> None:
    median_tx = adata.obs.groupby("fov_name")["n_transcripts"].median()
    fig, ax = plt.subplots(figsize=(8, 4))
    median_tx.plot.bar(ax=ax, color="#4C72B0")
    ax.set_ylabel("Median transcripts per cell")
    ax.set_title("Transcripts per cell by FOV")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
