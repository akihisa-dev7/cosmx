"""Normalization methods for RNA and protein data."""
import numpy as np
import pandas as pd

try:
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp
except ImportError:
    raise ImportError("scanpy and anndata are required.")


def normalize_rna(
    adata: "ad.AnnData",
    method: str = "log1p",
    target_sum: float = 10000,
) -> "ad.AnnData":
    """Normalize RNA counts.

    method='log1p': library-size normalize then log1p.
    Stores raw counts in adata.layers['counts'].
    """
    if sp.issparse(adata.X):
        adata.layers["counts"] = adata.X.copy()
    else:
        adata.layers["counts"] = adata.X.copy()

    if method == "log1p":
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    else:
        raise ValueError(f"Unknown RNA normalization method: {method}")

    return adata


def normalize_protein(mat: pd.DataFrame, method: str = "clr") -> pd.DataFrame:
    """Normalize protein intensity matrix.

    method='clr': centered log-ratio normalization per cell.
    """
    from .protein_processing import clr_normalize

    if method == "clr":
        return clr_normalize(mat)
    else:
        raise ValueError(f"Unknown protein normalization method: {method}")
