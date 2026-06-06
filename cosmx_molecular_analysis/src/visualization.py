"""Shared visualization utilities."""
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required.")


def save_composition_barplot(
    composition_df: pd.DataFrame,
    output_path: str,
    group_col: str = "fov_name",
    title: str = "Cell type composition",
) -> None:
    """Stacked bar chart of cell type fractions per group."""
    df = composition_df.copy()
    if group_col not in df.columns:
        return

    df = df.set_index(group_col)
    frac_cols = [c for c in df.columns if not c.startswith("n_") and c not in {"condition", "region"}]
    frac_cols = [c for c in frac_cols if df[frac_cols].dtypes[c] in [float, np.float64, np.float32]]
    if not frac_cols:
        frac_cols = [c for c in df.columns if df[c].dtype in [float, np.float64, np.float32]]

    if not frac_cols:
        return

    plot_df = df[frac_cols].fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.2), 5))
    plot_df.plot.bar(stacked=True, ax=ax, colormap="tab20", width=0.8)
    ax.set_xlabel(group_col)
    ax.set_ylabel("Fraction")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
