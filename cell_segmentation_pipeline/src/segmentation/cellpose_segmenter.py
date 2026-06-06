"""Cellpose segmenter: cpsam / nuclei / cyto3, with optional multi-channel support."""
from __future__ import annotations

import numpy as np

from .base import BaseSegmenter


class CellposeSegmenter(BaseSegmenter):
    """Wrapper around Cellpose models.

    Args:
        model:              Model type — 'cpsam', 'nuclei', or 'cyto3'.
        diameter:           Expected cell diameter in pixels. 25 is best for
                            brain nuclei at CosMx resolution (0.12 µm/px).
        cellprob_threshold: Lower = more cells detected. -1.0 captures dim nuclei.
        flow_threshold:     Higher = stricter shape criterion.
        channel:            Input channel index (0 = DAPI). Used only for single-ch.
        cp_channels:        Cellpose channels list [cytoplasm_ch, nucleus_ch] (1-indexed).
                            None = grayscale. [1, 2] = image[:,0] as cyto, [:,1] as nuc.
    """

    def __init__(
        self,
        model: str = "cpsam",
        diameter: float = 25.0,
        cellprob_threshold: float = -1.0,
        flow_threshold: float = 0.6,
        channel: int = 0,
        cp_channels: list | None = None,
    ) -> None:
        self._model_name = model
        self.diameter = diameter
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.channel = channel
        self.cp_channels = cp_channels  # e.g. [1, 2] for Membrane+DAPI
        self._model = None

    def _load_model(self) -> None:
        from cellpose.models import CellposeModel
        self._model = CellposeModel(pretrained_model=self._model_name, gpu=False)

    @property
    def name(self) -> str:
        suffix = "_membrane" if self.cp_channels is not None else ""
        return f"cellpose_{self._model_name}{suffix}"

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment a 2D image. image can be (H,W) grayscale or (H,W,2) multi-ch.

        Multi-channel convention (cp_channels=[1,2]):
          image[:,:,0] = cytoplasm / membrane stain
          image[:,:,1] = nucleus stain (DAPI)

        flow_threshold=0 avoids an int-overflow bug in cellpose 4.x on large images.
        """
        if self._model is None:
            self._load_model()

        result = self._model.eval(
            image,
            diameter=self.diameter,
            channels=self.cp_channels,   # None = grayscale, [1,2] = 2-channel
            cellprob_threshold=self.cellprob_threshold,
            flow_threshold=0.0,
        )
        masks = result[0]
        return masks.astype(np.int32)


class CellposeDAPIMembraneSegmenter(CellposeSegmenter):
    """cpsam with DAPI (nucleus) + Membrane (cytoplasm) dual-channel input.

    Preprocessing is applied inside segment() so callers can pass the raw
    multi-channel TIF array (H, W, 2) where ch0=Membrane, ch1=DAPI,
    both already TopHat-enhanced and normalised to float32 [0,1].
    """

    def __init__(self, diameter: float = 25.0, cellprob_threshold: float = -1.0) -> None:
        super().__init__(
            model="cpsam",
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            cp_channels=[1, 2],  # 1-indexed: ch1=Membrane(cyto), ch2=DAPI(nuc)
        )

    @property
    def name(self) -> str:
        return "cellpose_cpsam_membrane"
