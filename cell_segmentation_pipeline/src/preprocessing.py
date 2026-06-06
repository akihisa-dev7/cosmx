"""Image enhancement: TopHat, CLAHE, Rolling Ball, and comparison utilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.morphology import disk, white_tophat
from skimage.restoration import rolling_ball


# ── Core filters ─────────────────────────────────────────────────────────────

def apply_tophat(img: np.ndarray, radius: int = 20) -> np.ndarray:
    """White Top-Hat filter to suppress uneven background illumination."""
    return white_tophat(img, disk(radius))


def apply_clahe(img: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation → uint16 output."""
    # equalize_adapthist expects float [0,1] and returns float [0,1]
    float_img = img.astype(np.float64) / (img.max() or 1)
    enhanced = equalize_adapthist(float_img, clip_limit=clip_limit)
    # scale back to original dtype range
    return (enhanced * np.iinfo(np.uint16).max).astype(np.uint16)


def apply_rolling_ball(img: np.ndarray, radius: float = 50.0) -> np.ndarray:
    """Rolling ball background subtraction."""
    background = rolling_ball(img.astype(np.float32), radius=radius)
    subtracted = img.astype(np.float32) - background
    return np.clip(subtracted, 0, None).astype(img.dtype)


# ── Dispatcher ───────────────────────────────────────────────────────────────

def preprocess(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply preprocessing method specified in cfg['method']."""
    method = cfg.get("method", "tophat").lower()
    if method == "tophat":
        return apply_tophat(img, radius=cfg.get("tophat_radius_px", 20))
    if method == "clahe":
        return apply_clahe(img, clip_limit=cfg.get("clahe_clip_limit", 0.03))
    if method == "rolling_ball":
        return apply_rolling_ball(img, radius=cfg.get("rolling_ball_radius_px", 50))
    if method == "none":
        return img
    raise ValueError(f"Unknown preprocessing method: '{method}'")


# ── Comparison panel ─────────────────────────────────────────────────────────

def _norm8(img: np.ndarray) -> np.ndarray:
    """Normalise to uint8 for display."""
    mn, mx = img.min(), img.max()
    if mx == mn:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def save_comparison_png(
    images: dict[str, np.ndarray],
    path: str | Path,
    title: str = "",
) -> None:
    """Save a side-by-side comparison of multiple 2D images.

    Args:
        images: Ordered dict of {label: 2D array}.
        path:   Output PNG path.
        title:  Overall figure title.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=150)
    if n == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, images.items()):
        ax.imshow(_norm8(img), cmap="gray", interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)

    plt.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def build_tophat_comparison(
    raw: np.ndarray,
    radii: list[int],
) -> dict[str, np.ndarray]:
    """Build comparison dict: raw + TopHat at each radius."""
    imgs: dict[str, np.ndarray] = {"raw": raw}
    for r in radii:
        imgs[f"TopHat r={r}"] = apply_tophat(raw, r)
    return imgs
