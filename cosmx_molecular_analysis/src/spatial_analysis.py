"""Spatial analysis: cell type maps, density, composition, neighborhood."""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    import scipy.sparse as sp
except ImportError:
    raise ImportError("anndata, matplotlib, and scipy are required.")


CELL_TYPE_PALETTE = {
    "Neuron": "#E63946",
    "Astrocyte": "#457B9D",
    "Microglia": "#2A9D8F",
    "Oligodendrocyte": "#E9C46A",
    "OPC": "#F4A261",
    "Endothelial": "#9B2226",
    "Pericyte_VSMC": "#9D4EDD",
    "LowConfidence": "#CCCCCC",
    "Unknown": "#AAAAAA",
}


def get_color(cell_type: str) -> str:
    return CELL_TYPE_PALETTE.get(cell_type, "#888888")


def save_spatial_cell_type_maps(
    adata: "ad.AnnData",
    output_dir: str,
    fov_names: Optional[List[str]] = None,
    coord_key: str = "spatial",
) -> None:
    """Save spatial cell type scatter plot for each FOV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if coord_key not in adata.obsm:
        print(f"  [WARN] '{coord_key}' not in adata.obsm, skipping spatial maps")
        return

    if "predicted_cell_type" not in adata.obs.columns:
        print("  [WARN] 'predicted_cell_type' not in obs, skipping spatial maps")
        return

    if fov_names is None and "fov_name" in adata.obs.columns:
        fov_names = sorted(adata.obs["fov_name"].unique())
    elif fov_names is None:
        fov_names = ["all"]

    all_types = sorted(adata.obs["predicted_cell_type"].unique())

    for fov in fov_names:
        if fov == "all":
            sub = adata
        elif "fov_name" in adata.obs.columns:
            sub = adata[adata.obs["fov_name"] == fov]
        else:
            sub = adata

        if sub.n_obs == 0:
            continue

        coords = sub.obsm[coord_key]
        cell_types = sub.obs["predicted_cell_type"].astype(str)

        fig, ax = plt.subplots(figsize=(8, 8))
        for ct in all_types:
            sel = cell_types == ct
            if not sel.any():
                continue
            ax.scatter(
                coords[sel, 0], coords[sel, 1],
                s=8, alpha=0.8,
                color=get_color(ct),
                label=ct,
                linewidths=0,
            )

        ax.set_aspect("equal")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_title(f"Cell type spatial map – {fov} (n={sub.n_obs})")
        ax.legend(
            markerscale=2,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=8,
        )
        fig.tight_layout()
        fname = f"spatial_cell_type_map_{fov}.png"
        fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def compute_cell_type_composition_by_fov(adata: "ad.AnnData") -> pd.DataFrame:
    """Compute cell type fraction per FOV."""
    if "predicted_cell_type" not in adata.obs.columns:
        return pd.DataFrame()

    groupby_col = "fov_name" if "fov_name" in adata.obs.columns else None
    if groupby_col is None:
        return pd.DataFrame()

    counts = (
        adata.obs.groupby([groupby_col, "predicted_cell_type"])
        .size()
        .unstack(fill_value=0)
    )
    fractions = counts.div(counts.sum(axis=1), axis=0).round(4)
    fractions.index.name = "fov_name"

    # Add total cell count
    fractions["n_cells"] = counts.sum(axis=1)

    # Add condition / region if available
    meta_cols = [c for c in ["condition", "region"] if c in adata.obs.columns]
    if meta_cols:
        meta = adata.obs.drop_duplicates(subset=[groupby_col])[
            [groupby_col] + meta_cols
        ].set_index(groupby_col)
        fractions = fractions.join(meta, how="left")

    return fractions.reset_index()


def compute_cell_density_by_fov(adata: "ad.AnnData") -> pd.DataFrame:
    """Compute cell density (cells/FOV area) by FOV."""
    if "fov_name" not in adata.obs.columns:
        return pd.DataFrame()

    density = adata.obs.groupby("fov_name").size().reset_index(name="n_cells")

    if "spatial" in adata.obsm:
        coords_df = pd.DataFrame(
            adata.obsm["spatial"],
            index=adata.obs_names,
            columns=["x_um", "y_um"],
        )
        coords_df["fov_name"] = adata.obs["fov_name"].values
        fov_area = coords_df.groupby("fov_name").apply(
            lambda g: (g.x_um.max() - g.x_um.min() + 1) * (g.y_um.max() - g.y_um.min() + 1)
        ).reset_index(name="fov_area_um2")
        density = density.merge(fov_area, on="fov_name", how="left")
        density["cell_density_per_mm2"] = (density["n_cells"] / density["fov_area_um2"] * 1e6).round(4)

    return density


def compute_neighborhood_summary(
    adata: "ad.AnnData",
    radius_px: float = 100,
    coord_key: str = "spatial_px",
) -> pd.DataFrame:
    """Simple spatial neighbor enrichment: count cell-type neighbors within radius."""
    if coord_key not in adata.obsm:
        coord_key = "spatial"
    if coord_key not in adata.obsm:
        return pd.DataFrame()
    if "predicted_cell_type" not in adata.obs.columns:
        return pd.DataFrame()

    try:
        from scipy.spatial import KDTree
    except ImportError:
        return pd.DataFrame()

    coords = adata.obsm[coord_key]
    cell_types = adata.obs["predicted_cell_type"].values
    unique_types = sorted(set(cell_types))

    tree = KDTree(coords)
    rows = []

    for fov in adata.obs.get("fov_name", pd.Series(["all"] * adata.n_obs)).unique():
        if "fov_name" in adata.obs.columns:
            fov_mask = adata.obs["fov_name"] == fov
            fov_coords = coords[fov_mask]
            fov_types = cell_types[fov_mask]
        else:
            fov_coords = coords
            fov_types = cell_types
            fov = "all"

        if len(fov_coords) < 2:
            continue

        fov_tree = KDTree(fov_coords)
        for i, ct in enumerate(unique_types):
            ct_mask = fov_types == ct
            if not ct_mask.any():
                continue
            ct_coords = fov_coords[ct_mask]
            neighbor_indices = fov_tree.query_ball_point(ct_coords, r=radius_px)
            neighbor_type_counts = {t: 0 for t in unique_types}
            for nbrs in neighbor_indices:
                for j in nbrs:
                    neighbor_type_counts[fov_types[j]] += 1
            row = {"fov_name": fov, "cell_type": ct, "n_cells": ct_mask.sum()}
            row.update({f"neighbors_{t}": v for t, v in neighbor_type_counts.items()})
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
