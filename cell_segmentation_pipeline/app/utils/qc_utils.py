"""QC statistics and plotly chart builders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def compute_stats(
    cell_table: pd.DataFrame,
    min_area: int = 50,
    max_area: int = 10000,
) -> dict:
    """Compute scalar QC statistics from a cell table."""
    if cell_table is None or len(cell_table) == 0:
        return {}

    areas = cell_table["area"]
    diams = cell_table.get("est_diam_px", 2 * np.sqrt(areas / np.pi))

    return {
        "n_cells":         int(len(cell_table)),
        "median_area_px":  float(areas.median()),
        "mean_area_px":    float(areas.mean()),
        "median_diam_px":  float(diams.median()),
        "mean_diam_px":    float(diams.mean()),
        "min_area_px":     float(areas.min()),
        "max_area_px":     float(areas.max()),
        "pct_too_small":   float((areas < min_area).mean() * 100),
        "pct_too_large":   float((areas > max_area).mean() * 100),
    }


def area_histogram(cell_table: pd.DataFrame, min_area: int = 50, max_area: int = 10000) -> go.Figure:
    """Plotly histogram of cell areas with threshold lines."""
    areas = cell_table["area"]
    fig = px.histogram(
        cell_table, x="area", nbins=60,
        labels={"area": "Cell area (px²)", "count": "Count"},
        title="Cell area distribution",
        color_discrete_sequence=["#4e79a7"],
    )
    fig.add_vline(x=min_area, line_dash="dash", line_color="red",
                  annotation_text=f"min={min_area}", annotation_position="top right")
    fig.add_vline(x=max_area, line_dash="dash", line_color="orange",
                  annotation_text=f"max={max_area}", annotation_position="top left")
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=300)
    return fig


def diameter_histogram(cell_table: pd.DataFrame) -> go.Figure:
    """Plotly histogram of estimated cell diameters."""
    if "est_diam_px" not in cell_table.columns:
        cell_table = cell_table.copy()
        cell_table["est_diam_px"] = 2 * np.sqrt(cell_table["area"] / np.pi)
    fig = px.histogram(
        cell_table, x="est_diam_px", nbins=50,
        labels={"est_diam_px": "Estimated diameter (px)", "count": "Count"},
        title="Cell diameter distribution",
        color_discrete_sequence=["#59a14f"],
    )
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=300)
    return fig


def model_comparison_bar(results: list[dict]) -> go.Figure:
    """Bar chart comparing cell counts across models / FOVs.

    results: list of dicts with keys 'label' (str) and 'n_cells' (int).
    """
    labels = [r["label"] for r in results]
    counts = [r["n_cells"] for r in results]
    fig = px.bar(
        x=labels, y=counts,
        labels={"x": "", "y": "Cell count"},
        title="Cell count comparison",
        color=counts, color_continuous_scale="blues",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(t=40, b=40, l=20, r=20),
        height=300,
        xaxis_tickangle=-30,
    )
    return fig


def scatter_area_vs_diam(cell_table: pd.DataFrame) -> go.Figure:
    """Scatter: area vs estimated diameter (sanity check for circular assumption)."""
    ct = cell_table.copy()
    if "est_diam_px" not in ct.columns:
        ct["est_diam_px"] = 2 * np.sqrt(ct["area"] / np.pi)
    fig = px.scatter(
        ct, x="area", y="est_diam_px",
        labels={"area": "Area (px²)", "est_diam_px": "Est. diameter (px)"},
        title="Area vs diameter",
        opacity=0.5,
        color_discrete_sequence=["#e15759"],
    )
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=300)
    return fig


def flag_cells(cell_table: pd.DataFrame, min_area: int, max_area: int) -> pd.DataFrame:
    """Return cell table with 'qc_flag' column."""
    ct = cell_table.copy()
    ct["qc_flag"] = "pass"
    ct.loc[ct["area"] < min_area, "qc_flag"] = "too_small"
    ct.loc[ct["area"] > max_area, "qc_flag"] = "too_large"
    return ct
