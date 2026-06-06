"""Overlay composition and Plotly figure builders."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .image_loader import downsample_img, downsample_mask, norm_uint8


# ── Colour LUT ────────────────────────────────────────────────────────────────

def _make_lut(n_labels: int, seed: int = 42) -> np.ndarray:
    """Return (n+1, 3) uint8 RGB LUT. Index 0 = black background."""
    rng = np.random.default_rng(seed)
    lut = np.zeros((n_labels + 1, 3), dtype=np.uint8)
    if n_labels > 0:
        cols = (rng.random((n_labels, 3)) * 200 + 55).astype(np.uint8)  # avoid very dark
        lut[1:] = cols
    return lut


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert integer label mask to (H, W, 3) RGB."""
    n = int(mask.max())
    lut = _make_lut(n)
    return lut[mask]


# ── Overlay composition ───────────────────────────────────────────────────────

def compose_overlay(
    gray: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend grayscale image with random-colour mask.

    Args:
        gray:  2D uint8 (H, W).
        mask:  2D int32 label mask (H, W). 0 = background.
        alpha: mask opacity [0, 1].

    Returns:
        (H, W, 3) uint8 RGB.
    """
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    mask_rgb = colorize_mask(mask).astype(np.float32)
    fg = mask > 0
    blended = base.copy()
    blended[fg] = (1 - alpha) * base[fg] + alpha * mask_rgb[fg]
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── Plotly figure builders ────────────────────────────────────────────────────

def _plotly_layout(title: str = "") -> dict:
    return dict(
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        title=title,
        coloraxis_showscale=False,
        dragmode="pan",
    )


def make_single_fig(
    img: np.ndarray,
    title: str = "",
    max_dim: int = 1024,
    gray: bool = True,
) -> go.Figure:
    """Plotly figure with zoom/pan for a single 2D or RGB image."""
    ds = downsample_img(norm_uint8(img) if img.ndim == 2 else img, max_dim)
    if ds.ndim == 2:
        fig = px.imshow(ds, color_continuous_scale="gray", aspect="equal")
    else:
        fig = px.imshow(ds, aspect="equal")
    fig.update_layout(**_plotly_layout(title))
    fig.update_traces(hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z}<extra></extra>")
    return fig


def make_overlay_fig(
    raw: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    title: str = "",
    max_dim: int = 1024,
) -> go.Figure:
    """Plotly figure of overlay (grayscale + mask colours)."""
    gray_ds = downsample_img(norm_uint8(raw), max_dim)
    mask_ds = downsample_mask(mask, max_dim)
    rgb = compose_overlay(gray_ds, mask_ds, alpha)
    fig = px.imshow(rgb, aspect="equal")
    fig.update_layout(**_plotly_layout(title))
    return fig


def make_comparison_fig(
    panels: list[dict],
    max_dim: int = 512,
) -> go.Figure:
    """Grid of image panels for model/parameter comparison.

    Args:
        panels: list of dicts with keys 'title' (str), 'img' (ndarray 2D or RGB).
    """
    n = len(panels)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[p["title"] for p in panels],
        horizontal_spacing=0.02, vertical_spacing=0.08,
    )

    for i, panel in enumerate(panels):
        row, col = divmod(i, ncols)
        img = panel["img"]
        ds = downsample_img(norm_uint8(img) if img.ndim == 2 else img, max_dim)
        trace = px.imshow(
            ds,
            color_continuous_scale="gray" if ds.ndim == 2 else None,
            aspect="equal",
        ).data[0]
        fig.add_trace(trace, row=row + 1, col=col + 1)
        # force gray colorscale per axis
        if ds.ndim == 2:
            caxis = f"coloraxis{i + 1}" if i > 0 else "coloraxis"
            fig.update_layout(**{caxis: dict(showscale=False, colorscale="gray")})

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=350 * nrows,
        showlegend=False,
    )
    # Hide all color scales
    fig.update_coloraxes(showscale=False)
    return fig
