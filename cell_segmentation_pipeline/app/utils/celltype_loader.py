"""Loaders for cosmx_molecular_analysis outputs used by the validation GUI."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Path helpers ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # CosMx_2026/
MOL_ROOT = PROJECT_ROOT / "outputs" / "cosmx_molecular_analysis"

CELL_TYPE_PALETTE: dict[str, str] = {
    "Neuron":          "#E63946",
    "Astrocyte":       "#457B9D",
    "Microglia":       "#2A9D8F",
    "Oligodendrocyte": "#E9C46A",
    "OPC":             "#F4A261",
    "Endothelial":     "#9B2226",
    "Pericyte_VSMC":   "#9D4EDD",
    "LowConfidence":   "#BBBBBB",
    "Unknown":         "#888888",
    "Manual":          "#FF6B35",
}

# Sorted canonical order for display
CELL_TYPE_ORDER = [
    "Neuron", "Astrocyte", "Microglia", "Oligodendrocyte", "OPC",
    "Endothelial", "Pericyte_VSMC", "LowConfidence", "Unknown",
]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ── Lazy Streamlit cache ───────────────────────────────────────────────────────
def _cache_data(fn):
    try:
        import streamlit as st
        return st.cache_data(show_spinner=False)(fn)
    except Exception:
        return fn


def _cache_resource(fn):
    try:
        import streamlit as st
        return st.cache_resource(show_spinner=False)(fn)
    except Exception:
        return fn


# ── File loaders ──────────────────────────────────────────────────────────────

def _load_cell_type_table_raw(mol_root: str) -> pd.DataFrame | None:
    path = Path(mol_root) / "cell_type_table.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


load_cell_type_table = _cache_data(_load_cell_type_table_raw)


def _load_assignment_summary_raw(mol_root: str) -> pd.DataFrame | None:
    path = Path(mol_root) / "transcript_assignment" / "assignment_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


load_assignment_summary = _cache_data(_load_assignment_summary_raw)


def _load_joint_fov_summary_raw(mol_root: str) -> pd.DataFrame | None:
    path = Path(mol_root) / "integration" / "joint_fov_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


load_joint_fov_summary = _cache_data(_load_joint_fov_summary_raw)


def _load_composition_raw(mol_root: str) -> pd.DataFrame | None:
    path = Path(mol_root) / "spatial" / "cell_type_composition_by_fov.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


load_composition = _cache_data(_load_composition_raw)


@_cache_resource
def load_anndata(mol_root: str):
    """Load rna_annotated.h5ad. Returns AnnData or None."""
    try:
        import anndata as ad
        path = Path(mol_root) / "anndata" / "rna_annotated.h5ad"
        if not path.exists():
            return None
        return ad.read_h5ad(str(path))
    except Exception:
        return None


# ── Cell ID helpers ────────────────────────────────────────────────────────────

def label_to_cell_id(fov_name: str, label: int) -> str:
    """Convert mask label → cell_id string."""
    return f"{fov_name}_{label}"


def cell_id_to_label(cell_id: str) -> int | None:
    """Extract mask label from cell_id string (FOV00001_42 → 42)."""
    parts = cell_id.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def get_fov_cells(cell_type_table: pd.DataFrame, fov_name: str) -> pd.DataFrame:
    """Return rows of cell_type_table belonging to fov_name."""
    col = "fov_name" if "fov_name" in cell_type_table.columns else None
    if col is None:
        # Try to infer from cell_id
        if "cell_id" in cell_type_table.columns:
            return cell_type_table[cell_type_table["cell_id"].str.startswith(fov_name + "_")]
        return cell_type_table
    return cell_type_table[cell_type_table[col] == fov_name]


# ── Colored mask builders ──────────────────────────────────────────────────────

def build_cell_type_colored_mask(
    mask: np.ndarray,
    cell_type_table: pd.DataFrame,
    fov_name: str,
    manual_overrides: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build RGB mask colored by predicted cell type.

    Returns:
        rgb: (H, W, 3) uint8
        label_to_type: dict mapping mask label → cell type string
    """
    fov_cells = get_fov_cells(cell_type_table, fov_name)
    label_to_type: dict[int, str] = {}

    id_col = "cell_id" if "cell_id" in fov_cells.columns else None
    ct_col = "predicted_cell_type" if "predicted_cell_type" in fov_cells.columns else None

    if id_col and ct_col:
        for _, row in fov_cells.iterrows():
            cid = str(row[id_col])
            lbl = cell_id_to_label(cid)
            if lbl is not None:
                ct = str(row[ct_col])
                if manual_overrides and cid in manual_overrides:
                    ct = manual_overrides[cid]
                label_to_type[lbl] = ct

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for lbl, ct in label_to_type.items():
        color = CELL_TYPE_PALETTE.get(ct, CELL_TYPE_PALETTE["Unknown"])
        r, g, b = _hex_to_rgb(color)
        pixels = mask == lbl
        rgb[pixels, 0] = r
        rgb[pixels, 1] = g
        rgb[pixels, 2] = b

    return rgb, {v: k for k, v in label_to_type.items()}


def build_expression_mask(
    mask: np.ndarray,
    adata,  # AnnData
    gene_or_obs: str,
    fov_name: str,
    use_obs: bool = False,
) -> np.ndarray | None:
    """Build float32 mask with per-cell expression values.

    Returns array (H, W) with expression values, 0 for background/unknown cells.
    Returns None if gene not found.
    """
    import scipy.sparse as sp

    h, w = mask.shape
    result = np.zeros((h, w), dtype=np.float32)

    if use_obs:
        if gene_or_obs not in adata.obs.columns:
            return None
        vals = adata.obs[gene_or_obs].values
        ids = list(adata.obs_names)
    else:
        if gene_or_obs not in adata.var_names:
            return None
        gene_idx = list(adata.var_names).index(gene_or_obs)
        X = adata.X
        if sp.issparse(X):
            vals = np.array(X[:, gene_idx].todense()).flatten()
        else:
            vals = X[:, gene_idx].flatten()
        ids = list(adata.obs_names)

    # Build label → expression mapping for this FOV
    label_expr: dict[int, float] = {}
    for cell_id, val in zip(ids, vals):
        if not cell_id.startswith(fov_name + "_"):
            continue
        lbl = cell_id_to_label(cell_id)
        if lbl is not None:
            label_expr[lbl] = float(val)

    if not label_expr:
        return None

    for lbl, val in label_expr.items():
        result[mask == lbl] = val

    return result


# ── Cell info lookups ──────────────────────────────────────────────────────────

def get_cell_info(
    cell_type_table: pd.DataFrame,
    adata,  # AnnData | None
    cell_id: str,
) -> dict[str, Any]:
    """Return all available info for a single cell."""
    info: dict[str, Any] = {"cell_id": cell_id}

    # From cell_type_table
    id_col = "cell_id" if "cell_id" in cell_type_table.columns else None
    if id_col:
        rows = cell_type_table[cell_type_table[id_col] == cell_id]
        if not rows.empty:
            row = rows.iloc[0]
            for col in row.index:
                info[col] = row[col]

    # From AnnData
    if adata is not None and cell_id in adata.obs_names:
        obs_row = adata.obs.loc[cell_id]
        for col in obs_row.index:
            if col not in info:
                info[col] = obs_row[col]

        # Top expressed genes (marker scores)
        import scipy.sparse as sp
        cell_idx = list(adata.obs_names).index(cell_id)
        X = adata.X
        if sp.issparse(X):
            expr = np.array(X[cell_idx, :].todense()).flatten()
        else:
            expr = X[cell_idx, :].flatten()

        top_idx = np.argsort(expr)[::-1][:10]
        top_genes = [(adata.var_names[i], float(expr[i])) for i in top_idx if expr[i] > 0]
        info["top_genes"] = top_genes

        # Cell type scores
        if "cell_type_scores" in adata.obsm:
            score_cols = adata.uns.get("cell_type_score_columns", [])
            scores_row = adata.obsm["cell_type_scores"][cell_idx]
            info["cell_type_scores_dict"] = dict(zip(score_cols, scores_row))

    return info


# ── Available FOVs ─────────────────────────────────────────────────────────────

def get_available_fovs_mol(mol_root: str) -> list[str]:
    """Return list of FOV names that have cell type data."""
    ct = load_cell_type_table(mol_root)
    if ct is None:
        return []
    if "fov_name" in ct.columns:
        return sorted(ct["fov_name"].dropna().unique())
    elif "cell_id" in ct.columns:
        fovs = set()
        for cid in ct["cell_id"]:
            parts = str(cid).rsplit("_", 1)
            if len(parts) == 2:
                fovs.add(parts[0])
        return sorted(fovs)
    return []


# ── Export helpers ────────────────────────────────────────────────────────────

def save_manual_review(
    original_ct_table: pd.DataFrame,
    manual_overrides: dict[str, str],
    output_path: str,
) -> pd.DataFrame:
    """Merge manual overrides into cell_type_table and save."""
    df = original_ct_table.copy()
    if "predicted_cell_type" not in df.columns:
        df["predicted_cell_type"] = "Unknown"
    if "manual_cell_type" not in df.columns:
        df["manual_cell_type"] = df["predicted_cell_type"]
    if "manually_reviewed" not in df.columns:
        df["manually_reviewed"] = False

    id_col = "cell_id" if "cell_id" in df.columns else None
    if id_col and manual_overrides:
        for cell_id, new_type in manual_overrides.items():
            mask = df[id_col] == cell_id
            df.loc[mask, "manual_cell_type"] = new_type
            df.loc[mask, "manually_reviewed"] = True

    df.to_csv(output_path, index=False)
    return df
