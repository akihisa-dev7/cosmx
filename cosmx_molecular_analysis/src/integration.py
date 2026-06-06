"""FOV-level RNA–Protein integration analysis."""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import anndata as ad
except ImportError:
    raise ImportError("anndata and matplotlib are required.")


# RNA marker → Protein marker correspondence
# (rna_cell_type_score_name, protein_column)
# rna_cell_type_score_name matches columns like "score_Microglia" in joint_fov_summary
RNA_PROTEIN_PAIRS = [
    ("Microglia", "CD68"),          # Microglia score → CD68 (macrophage/microglia)
    ("Pericyte_VSMC", "SMA"),       # Pericyte/VSMC score → smooth muscle actin
    ("Endothelial", "CD31"),        # Endothelial score → PECAM1/CD31
    ("Microglia", "CD11b"),         # Microglia → CD11b (integrin)
    ("Neuron", "CD45"),             # Neuron vs immune marker (expected low)
]

# Fallback alternative names for protein markers
PROTEIN_ALIASES = {
    "SMA": ["SMA", "ACTA2", "alpha-SMA"],
    "CD31": ["CD31", "PECAM1"],
    "CD68": ["CD68", "IBA1"],
    "GFAP": ["GFAP"],
    "MBP": ["MBP"],
}


def build_rna_fov_summary(
    adata: "ad.AnnData",
    marker_genes: Dict[str, List[str]],
) -> pd.DataFrame:
    """Compute per-FOV mean marker scores and cell type fractions from RNA."""
    if "fov_name" not in adata.obs.columns:
        return pd.DataFrame()

    rows = []
    fov_names = sorted(adata.obs["fov_name"].unique())

    for fov in fov_names:
        sub = adata[adata.obs["fov_name"] == fov]
        row = {"fov_name": fov, "n_cells_rna": sub.n_obs}

        # Cell type composition
        if "predicted_cell_type" in sub.obs.columns:
            counts = sub.obs["predicted_cell_type"].value_counts()
            total = counts.sum()
            for ct, cnt in counts.items():
                row[f"frac_{ct}"] = round(cnt / total, 4)

        # Marker scores
        if "cell_type_scores" in sub.obsm:
            score_cols = sub.uns.get("cell_type_score_columns", [])
            mean_scores = sub.obsm["cell_type_scores"].mean(axis=0)
            for col, val in zip(score_cols, mean_scores):
                row[f"score_{col}"] = round(float(val), 6)

        # Meta
        for col in ["condition", "region"]:
            if col in sub.obs.columns:
                vals = sub.obs[col].dropna().unique()
                row[col] = vals[0] if len(vals) == 1 else ",".join(map(str, vals))

        rows.append(row)

    return pd.DataFrame(rows)


def build_protein_fov_summary(
    protein_df: pd.DataFrame,
    fov_int_to_name: dict,
    fovs: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Compute per-FOV mean protein intensities."""
    from .protein_processing import get_protein_markers

    if fovs is not None:
        protein_df = protein_df[protein_df["fov"].isin(fovs)]

    markers = get_protein_markers(protein_df)
    agg = protein_df.groupby("fov")[markers].mean()
    agg["n_cells_protein"] = protein_df.groupby("fov").size()
    agg = agg.reset_index()
    agg["fov_name"] = agg["fov"].map(fov_int_to_name)
    return agg


def merge_rna_protein_fov(
    rna_fov: pd.DataFrame,
    protein_fov: pd.DataFrame,
) -> pd.DataFrame:
    """Merge RNA and protein FOV summaries on fov_name."""
    merged = rna_fov.merge(protein_fov, on="fov_name", how="outer", suffixes=("_rna", "_prot"))
    return merged


def compute_rna_protein_correlation(
    joint_df: pd.DataFrame,
    rna_protein_pairs: Optional[List] = None,
) -> pd.DataFrame:
    """Compute Pearson correlation for each RNA–protein pair."""
    if rna_protein_pairs is None:
        rna_protein_pairs = RNA_PROTEIN_PAIRS

    rows = []
    for rna_marker, prot_marker in rna_protein_pairs:
        # RNA column: "score_<cell_type>" format
        rna_col = f"score_{rna_marker.replace(' ', '_')}"
        # Fallback: fuzzy search for rna_marker in column names
        if rna_col not in joint_df.columns:
            candidates = [c for c in joint_df.columns if rna_marker.lower() in c.lower()]
            rna_col = candidates[0] if candidates else None

        # Protein column: direct name or aliases
        prot_col = None
        for candidate in [prot_marker] + PROTEIN_ALIASES.get(prot_marker, []):
            if candidate in joint_df.columns:
                prot_col = candidate
                break

        if rna_col is None or prot_col is None:
            rows.append({
                "rna_marker": rna_marker,
                "protein_marker": prot_marker,
                "rna_column": rna_col,
                "protein_column": prot_col,
                "pearson_r": np.nan,
                "n_fovs": 0,
                "note": "column not found",
            })
            continue

        sub = joint_df[[rna_col, prot_col]].dropna()
        if len(sub) < 3:
            r = np.nan
        else:
            r = float(np.corrcoef(sub[rna_col], sub[prot_col])[0, 1])

        rows.append({
            "rna_marker": rna_marker,
            "protein_marker": prot_marker,
            "rna_column": rna_col,
            "protein_column": prot_col,
            "pearson_r": round(r, 4) if not np.isnan(r) else np.nan,
            "n_fovs": len(sub),
        })

    return pd.DataFrame(rows)


def save_correlation_plots(
    joint_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save scatter plots for each RNA–protein pair."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for _, row in corr_df.iterrows():
        rna_col = row.get("rna_column")
        prot_col = row.get("protein_column")

        if not rna_col or not prot_col:
            continue
        if rna_col not in joint_df.columns or prot_col not in joint_df.columns:
            continue

        sub = joint_df[[rna_col, prot_col]].dropna()
        if len(sub) < 2:
            continue

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(sub[rna_col], sub[prot_col], alpha=0.8, s=50, color="#2A9D8F")

        if len(sub) >= 2 and not np.isnan(row.get("pearson_r", np.nan)):
            # Trend line
            z = np.polyfit(sub[rna_col], sub[prot_col], 1)
            p = np.poly1d(z)
            xline = np.linspace(sub[rna_col].min(), sub[rna_col].max(), 50)
            ax.plot(xline, p(xline), "r--", alpha=0.7)

        r_val = row.get("pearson_r", np.nan)
        ax.set_xlabel(f"RNA score: {row['rna_marker']}")
        ax.set_ylabel(f"Protein intensity: {row['protein_marker']}")
        ax.set_title(f"r={r_val:.3f}" if not np.isnan(r_val) else "correlation")

        rna_safe = row["rna_marker"].replace("/", "_").replace(" ", "_")
        prot_safe = row["protein_marker"].replace("/", "_").replace(" ", "_")
        fname = f"scatter_{rna_safe}_RNA_vs_{prot_safe}_protein.png"
        fig.tight_layout()
        fig.savefig(Path(output_dir) / fname, dpi=120)
        plt.close(fig)
        print(f"  Saved: {fname}")
