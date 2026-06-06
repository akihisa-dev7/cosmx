"""Load and preprocess CosMx protein intensity data."""
from typing import List, Optional

import numpy as np
import pandas as pd


SYSTEM_COLS = {"fov", "cell_ID", "cell_id", "cell"}


def load_and_filter_protein(
    exprmat_path: str,
    fovs: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Load protein exprMat, optionally subset to FOVs.

    Returns DataFrame with fov, cell_ID, and protein columns.
    """
    from .io import load_protein_exprmat
    df = load_protein_exprmat(exprmat_path, fovs=fovs)
    df.columns = df.columns.str.strip()
    return df


def get_protein_markers(df: pd.DataFrame) -> List[str]:
    """Return list of protein marker column names (non-system cols)."""
    return [c for c in df.columns if c.lower() not in SYSTEM_COLS]


def build_protein_matrix(df: pd.DataFrame, fov_names_map: dict) -> pd.DataFrame:
    """Build cell × protein matrix with custom cell_id index.

    fov_names_map: {fov_int: fov_name}, e.g. {1: 'FOV00001'}.
    Index format: 'FOV00001_<cell_ID>'.
    """
    markers = get_protein_markers(df)
    df = df.copy()
    df["fov_name"] = df["fov"].map(fov_names_map)
    df = df.dropna(subset=["fov_name"])
    df["cell_id"] = df["fov_name"] + "_" + df["cell_ID"].astype(str)
    mat = df.set_index("cell_id")[markers]
    return mat


def protein_summary_by_fov(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean protein intensity per FOV."""
    markers = get_protein_markers(df)
    summary = df.groupby("fov")[markers].agg(["mean", "median", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["n_cells"] = df.groupby("fov").size()
    return summary.reset_index()


def clr_normalize(mat: pd.DataFrame) -> pd.DataFrame:
    """Centered log-ratio normalization for protein intensities.

    CLR(x_i) = log(x_i / geometric_mean(x)) for each cell.
    Small offset avoids log(0).
    """
    arr = mat.values.astype(float)
    arr = np.where(arr < 0, 0, arr)  # clip negatives to 0
    arr = arr + 1e-6  # pseudo-count
    log_arr = np.log(arr)
    geo_mean_log = log_arr.mean(axis=1, keepdims=True)
    clr_arr = log_arr - geo_mean_log
    return pd.DataFrame(clr_arr, index=mat.index, columns=mat.columns)
