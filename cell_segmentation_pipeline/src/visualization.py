"""Overlay and comparison visualisation utilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _norm8(img: np.ndarray) -> np.ndarray:
    """Normalise 2D array to uint8 [0, 255]."""
    mn, mx = float(img.min()), float(img.max())
    if mx == mn:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def _random_lut(n_labels: int, seed: int = 42) -> np.ndarray:
    """Return (n_labels+1, 3) float32 RGB lookup table. Index 0 = black."""
    lut = np.zeros((n_labels + 1, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    lut[1:] = rng.random((n_labels, 3))
    return lut


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend grayscale image with random-colour mask overlay.

    Args:
        image: 2D grayscale image (any dtype).
        mask:  Integer label mask (0 = background).
        alpha: Mask opacity (0 = invisible, 1 = opaque).

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    gray = _norm8(image)
    base_rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    n_labels = int(mask.max())
    lut = _random_lut(n_labels)
    mask_rgb = lut[mask] * 255.0   # (H, W, 3) float

    # Only blend where mask > 0
    fg = mask > 0
    blended = base_rgb.copy()
    blended[fg] = (1.0 - alpha) * base_rgb[fg] + alpha * mask_rgb[fg]

    return np.clip(blended, 0, 255).astype(np.uint8)


def save_overlay_png(
    image: np.ndarray,
    mask: np.ndarray,
    path: str | Path,
    alpha: float = 0.4,
) -> None:
    """Save overlay image to PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = overlay_mask_on_image(image, mask, alpha=alpha)
    n_cells = int(mask.max())

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(f"{n_cells} cells", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def make_model_comparison_grid(
    results: list[dict],
    path: str | Path,
) -> None:
    """Save a side-by-side comparison grid across segmentation models.

    Args:
        results: List of dicts, each with keys:
                   'model_name' (str), 'image' (2D array), 'mask' (2D array),
                   'n_cells' (int).
        path:    Output PNG path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), dpi=150)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        rgb = overlay_mask_on_image(res["image"], res["mask"], alpha=0.45)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{res['model_name']}\n{res['n_cells']} cells", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
