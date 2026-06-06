"""Build AnnData object from RNA counts and cell metadata."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import anndata as ad
    import scipy.sparse as sp
except ImportError:
    raise ImportError("anndata and scipy are required: pip install anndata scipy")

from .io import fov_name_to_int


# Cell table column → obs column mapping
CELL_TABLE_RENAME = {
    "label": "mask_label",
    "area": "cell_area_px2",
    "centroid_y": "centroid_y_px",
    "centroid_x": "centroid_x_px",
    "eccentricity": "eccentricity",
    "solidity": "solidity",
    "est_diam_px": "est_diam_px",
    "FOV": "fov_name",
    "condition": "condition",
    "region": "region",
    "sample_id": "sample_id",
    "gender": "gender",
    "age": "age",
}

PIXEL_SIZE_UM = 0.12  # µm/px for CosMx SMI


def load_cell_tables_for_fovs(
    cell_table_dir: str, fov_names: List[str]
) -> pd.DataFrame:
    """Load and concatenate cell tables for all FOVs."""
    from .io import find_cell_table_file, load_cell_table

    frames = []
    for fov_name in fov_names:
        path = find_cell_table_file(cell_table_dir, fov_name)
        if path is None:
            print(f"  [WARN] No cell table for {fov_name}")
            continue
        df = load_cell_table(path)
        df = df.rename(columns={c: CELL_TABLE_RENAME.get(c, c) for c in df.columns})
        if "fov_name" not in df.columns:
            df["fov_name"] = fov_name
        # Construct composite cell_id matching counts matrix index
        if "mask_label" in df.columns:
            df["cell_id"] = df["fov_name"] + "_" + df["mask_label"].astype(str)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_anndata(
    counts_matrix: pd.DataFrame,
    cell_tables: pd.DataFrame,
    fov_names: List[str],
    px_size_um: float = PIXEL_SIZE_UM,
) -> "ad.AnnData":
    """Construct AnnData from counts matrix and cell metadata.

    Parameters
    ----------
    counts_matrix:
        cell_id × gene DataFrame (index = cell_id strings).
    cell_tables:
        Merged cell morphology DataFrame with cell_id column.
    fov_names:
        List of FOV names being processed.
    """
    # Align obs to counts index
    obs_df = cell_tables.set_index("cell_id") if "cell_id" in cell_tables.columns else pd.DataFrame()
    common_cells = counts_matrix.index
    obs_aligned = obs_df.reindex(common_cells)

    # Convert counts to sparse matrix
    X = sp.csr_matrix(counts_matrix.values.astype(np.float32))

    # Build var (gene metadata)
    var_df = pd.DataFrame(index=counts_matrix.columns)
    var_df.index.name = "gene"

    adata = ad.AnnData(
        X=X,
        obs=obs_aligned,
        var=var_df,
    )
    adata.obs_names = list(counts_matrix.index)
    adata.var_names = list(counts_matrix.columns)

    # Add spatial coordinates (centroid in µm)
    if "centroid_x_px" in adata.obs.columns and "centroid_y_px" in adata.obs.columns:
        coords = adata.obs[["centroid_x_px", "centroid_y_px"]].values.astype(float)
        coords_um = coords * px_size_um
        adata.obsm["spatial"] = coords_um
        adata.obsm["spatial_px"] = coords
    else:
        # Fallback: zeros
        adata.obsm["spatial"] = np.zeros((adata.n_obs, 2), dtype=float)

    # Add cell area in µm²
    if "cell_area_px2" in adata.obs.columns:
        adata.obs["cell_area_um2"] = adata.obs["cell_area_px2"] * (px_size_um**2)

    # Transcript counts
    adata.obs["n_transcripts"] = np.array(X.sum(axis=1)).flatten().astype(int)
    adata.obs["n_genes"] = np.array((X > 0).sum(axis=1)).flatten().astype(int)

    # FOV integer
    if "fov_name" in adata.obs.columns:
        adata.obs["fov"] = adata.obs["fov_name"].apply(
            lambda x: fov_name_to_int(x) if isinstance(x, str) else np.nan
        )

    return adata
