"""Marker-based cell type annotation for CosMx RNA data."""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.sparse as sp
except ImportError:
    raise ImportError("anndata, scipy, and matplotlib are required.")


def compute_marker_scores(
    adata: "ad.AnnData",
    marker_genes: Dict[str, List[str]],
) -> pd.DataFrame:
    """Compute per-cell, per-cell-type marker scores.

    Score = mean normalized expression of markers present in the panel.
    Returns DataFrame (cells × cell_types).
    """
    gene_list = list(adata.var_names)
    gene_set = set(gene_list)
    scores = {}

    for cell_type, markers in marker_genes.items():
        present = [m for m in markers if m in gene_set]
        if not present:
            scores[cell_type] = np.zeros(adata.n_obs)
            continue

        idx = [gene_list.index(m) for m in present]
        if sp.issparse(adata.X):
            sub = np.array(adata.X[:, idx].todense())
        else:
            sub = adata.X[:, idx]

        scores[cell_type] = sub.mean(axis=1).flatten()

    return pd.DataFrame(scores, index=adata.obs_names)


def annotate_cell_types(
    adata: "ad.AnnData",
    marker_genes: Dict[str, List[str]],
    min_score_margin: float = 0.15,
    low_confidence_label: str = "LowConfidence",
) -> "ad.AnnData":
    """Assign predicted_cell_type to each cell based on highest marker score.

    If the gap between top-1 and top-2 score is less than min_score_margin,
    the cell is labeled as low_confidence_label.
    """
    score_df = compute_marker_scores(adata, marker_genes)

    top1 = score_df.idxmax(axis=1)
    top1_score = score_df.max(axis=1)

    # Second-best score
    def second_max(row):
        sorted_vals = row.sort_values(ascending=False)
        return sorted_vals.iloc[1] if len(sorted_vals) > 1 else 0.0

    top2_score = score_df.apply(second_max, axis=1)

    margin = top1_score - top2_score
    predicted = top1.copy()
    predicted[margin < min_score_margin] = low_confidence_label

    # Store in adata
    adata.obs["predicted_cell_type"] = predicted.values
    adata.obs["cell_type_score"] = top1_score.values
    adata.obs["cell_type_margin"] = margin.values

    # Store all scores in obsm
    adata.obsm["cell_type_scores"] = score_df.values
    adata.uns["cell_type_score_columns"] = list(score_df.columns)

    return adata


def build_cell_type_table(adata: "ad.AnnData") -> pd.DataFrame:
    """Return per-cell cell-type annotation table."""
    cols = ["predicted_cell_type", "cell_type_score", "cell_type_margin"]
    obs_cols = [c for c in cols if c in adata.obs.columns]

    extra = [
        c for c in ["fov_name", "condition", "region", "n_transcripts", "n_genes",
                    "cell_area_um2", "leiden"]
        if c in adata.obs.columns
    ]

    df = adata.obs[obs_cols + extra].copy()
    df.index.name = "cell_id"

    if "spatial" in adata.obsm:
        df["x_um"] = adata.obsm["spatial"][:, 0]
        df["y_um"] = adata.obsm["spatial"][:, 1]

    return df.reset_index()


def save_cell_type_plots(
    adata: "ad.AnnData",
    output_dir: str,
    marker_genes: Dict[str, List[str]],
) -> None:
    """Save UMAP by cell type and marker score heatmap."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "X_umap" in adata.obsm and "predicted_cell_type" in adata.obs.columns:
        _umap_cell_type(adata, Path(output_dir) / "umap_by_cell_type.png")

    _marker_score_heatmap(adata, marker_genes, Path(output_dir) / "marker_score_heatmap.png")


def _umap_cell_type(adata: "ad.AnnData", path: Path) -> None:
    umap = adata.obsm["X_umap"]
    cats = adata.obs["predicted_cell_type"].astype(str)
    unique = sorted(cats.unique())
    cmap = plt.cm.get_cmap("tab20", len(unique))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cat in enumerate(unique):
        sel = cats == cat
        ax.scatter(umap[sel, 0], umap[sel, 1], s=4, alpha=0.7, color=cmap(i), label=cat)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Cell type annotation")
    ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _marker_score_heatmap(
    adata: "ad.AnnData",
    marker_genes: Dict[str, List[str]],
    path: Path,
) -> None:
    if "predicted_cell_type" not in adata.obs.columns:
        return
    if "cell_type_scores" not in adata.obsm:
        return

    score_df = pd.DataFrame(
        adata.obsm["cell_type_scores"],
        index=adata.obs_names,
        columns=adata.uns.get("cell_type_score_columns", list(marker_genes.keys())),
    )
    score_df["predicted_cell_type"] = adata.obs["predicted_cell_type"].values
    mean_scores = score_df.groupby("predicted_cell_type").mean()

    fig, ax = plt.subplots(figsize=(max(8, len(mean_scores.columns)), max(4, len(mean_scores) * 0.6)))
    im = ax.imshow(mean_scores.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(mean_scores.columns)))
    ax.set_xticklabels(mean_scores.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mean_scores.index)))
    ax.set_yticklabels(mean_scores.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean score")
    ax.set_title("Marker score by predicted cell type")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
