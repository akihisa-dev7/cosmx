"""InstanSeg segmenter (optional — graceful import failure)."""
from __future__ import annotations

import numpy as np

from .base import BaseSegmenter


class _NotAvailableSegmenter(BaseSegmenter):
    """Placeholder returned when instanseg is not installed."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def name(self) -> str:
        return "instanseg_unavailable"

    def segment(self, image: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            f"InstanSeg is not available: {self._reason}. "
            "Install with: pip install instanseg-torch"
        )


class InstanSegSegmenter(BaseSegmenter):
    """Wrapper around InstanSeg fluorescence nucleus segmenter.

    Args:
        model:  Model name ('fluorescence_nuclei_and_cells').
        target: 'nuclei' or 'cells'.
    """

    def __init__(
        self,
        model: str = "fluorescence_nuclei_and_cells",
        target: str = "nuclei",
    ) -> None:
        self._model_name = model
        self.target = target
        self._model = None

    def _load_model(self) -> None:
        from instanseg import InstanSeg
        self._model = InstanSeg(self._model_name)

    @property
    def name(self) -> str:
        return "instanseg"

    def segment(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            self._load_model()

        import torch
        # instanseg 0.1.x: eval_small_image accepts numpy or tensor;
        # may return (label_tensor, image_tensor) tuple or bare label tensor.
        # label tensor shape: (B, C, H, W) or (C, H, W) or (H, W)
        result = self._model.eval_small_image(
            image,
            target="nuclei",
            return_image_tensor=False,
        )
        if isinstance(result, (tuple, list)):
            arr = result[0]
        else:
            arr = result
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        # Squeeze all leading singleton dims until 2D
        while arr.ndim > 2:
            arr = arr[0]
        return arr.astype(np.int32)


def get_instanseg(model: str = "fluorescence_nuclei_and_cells", target: str = "nuclei") -> BaseSegmenter:
    """Return InstanSegSegmenter or a placeholder if not installed."""
    try:
        import instanseg  # noqa: F401
        return InstanSegSegmenter(model=model, target=target)
    except ImportError as e:
        return _NotAvailableSegmenter(str(e))
