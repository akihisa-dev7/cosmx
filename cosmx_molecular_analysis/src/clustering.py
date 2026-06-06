"""PCA, UMAP, and Leiden clustering for RNA AnnData."""
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scanpy as sc
except ImportError:
    raise ImportError("scanpy, anndata, and matplotlib are required.")

sc.settings.verbosity = 1


def run_clustering(
    adata: "ad.AnnData",
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    resolution: float = 0.5,
    use_highly_variable: bool = True,
) -> "ad.AnnData":
    """Run HVG selection → PCA → neighbors → UMAP → Leiden.

    Modifies adata in-place and returns it.
    """
    # Highly variable genes (skip if fewer than n_top_genes)
    if use_highly_variable and adata.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat",
            subset=False,
        )
        use_rep_genes = adata.var["highly_variable"]
    else:
        use_rep_genes = None

    # PCA
    sc.tl.pca(
        adata,
        n_comps=min(n_pcs, adata.n_obs - 1, adata.n_vars - 1),
        use_highly_variable=(use_rep_genes is not None and use_rep_genes.any()),
    )

    # Neighbors
    actual_pcs = adata.obsm["X_pca"].shape[1]
    sc.pp.neighbors(
        adata,
        n_neighbors=min(n_neighbors, adata.n_obs - 1),
        n_pcs=actual_pcs,
    )

    # UMAP
    sc.tl.umap(adata)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=resolution)

    return adata


def save_umap_plots(adata: "ad.AnnData", output_dir: str) -> None:
    """Save UMAP plots colored by cluster, FOV, and condition."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "X_umap" not in adata.obsm:
        print("  [WARN] No UMAP coordinates found, skipping UMAP plots")
        return

    _umap_plot(adata, "leiden", output_dir, "umap_by_cluster.png", title="Leiden cluster")

    if "fov_name" in adata.obs.columns:
        _umap_plot(adata, "fov_name", output_dir, "umap_by_fov.png", title="FOV")

    if "condition" in adata.obs.columns:
        _umap_plot(adata, "condition", output_dir, "umap_by_condition.png", title="Condition")


def _umap_plot(
    adata: "ad.AnnData",
    color_key: str,
    output_dir: str,
    filename: str,
    title: str,
) -> None:
    if color_key not in adata.obs.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    umap = adata.obsm["X_umap"]
    cats = adata.obs[color_key].astype(str)
    unique = sorted(cats.unique())
    cmap = plt.cm.get_cmap("tab20", len(unique))

    for i, cat in enumerate(unique):
        sel = cats == cat
        ax.scatter(umap[sel, 0], umap[sel, 1], s=3, alpha=0.6, color=cmap(i), label=cat)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    if len(unique) <= 20:
        ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
