"""Base class and factory for segmentation models."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSegmenter(ABC):
    """Plugin interface for segmentation models."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Run segmentation on a 2D grayscale image.

        Args:
            image: 2D uint16 array (H, W).

        Returns:
            Integer label mask (H, W). 0 = background, 1..N = cell IDs.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in output filenames."""
        ...


# ── Factory ──────────────────────────────────────────────────────────────────

def get_segmenter(model_name: str, cfg: dict) -> BaseSegmenter:
    """Return a configured segmenter by name.

    Supported names:
        cellpose_cpsam, cellpose_nuclei, cellpose_cyto3
        stardist
        instanseg

    Args:
        model_name: One of the names above.
        cfg:        Full pipeline config dict (reads from cfg['segmentation']).
    """
    seg_cfg = cfg.get("segmentation", {})
    name = model_name.lower().strip()

    if name == "cellpose_cpsam_membrane":
        from .cellpose_segmenter import CellposeDAPIMembraneSegmenter
        cp_cfg = seg_cfg.get("cellpose", {}).get("cpsam", {})
        return CellposeDAPIMembraneSegmenter(
            diameter=cp_cfg.get("diameter", 25),
            cellprob_threshold=cp_cfg.get("cellprob_threshold", -1.0),
        )

    if name.startswith("cellpose_"):
        from .cellpose_segmenter import CellposeSegmenter
        model_key = name.split("_", 1)[1]   # cpsam | nuclei | cyto3
        cp_cfg = seg_cfg.get("cellpose", {}).get(model_key, {})
        ch = cfg.get("input", {}).get("channel", 0)
        return CellposeSegmenter(
            model=model_key,
            diameter=cp_cfg.get("diameter", 25),
            cellprob_threshold=cp_cfg.get("cellprob_threshold", -1.0),
            flow_threshold=cp_cfg.get("flow_threshold", 0.6),
            channel=ch,
        )

    if name == "stardist":
        from .stardist_segmenter import StarDistSegmenter
        sd_cfg = seg_cfg.get("stardist", {})
        raw_tiles = sd_cfg.get("n_tiles", None)
        n_tiles = tuple(raw_tiles) if raw_tiles is not None else None
        return StarDistSegmenter(
            model=sd_cfg.get("model", "2D_versatile_fluo"),
            prob_threshold=sd_cfg.get("prob_threshold", 0.3),
            nms_threshold=sd_cfg.get("nms_threshold", 0.4),
            scale=sd_cfg.get("scale", 0.25),
            n_tiles=n_tiles,
            gaussian_sigma=sd_cfg.get("gaussian_sigma", 5.0),
            min_diam_um=sd_cfg.get("min_diam_um", 3.0),
            max_diam_um=sd_cfg.get("max_diam_um", 20.0),
            min_intensity_ratio=sd_cfg.get("min_intensity_ratio", 1.0),
            pixel_size_um=sd_cfg.get("pixel_size_um", 0.12274),
        )

    if name == "instanseg":
        from .instanseg_segmenter import get_instanseg
        is_cfg = seg_cfg.get("instanseg", {})
        return get_instanseg(
            model=is_cfg.get("model", "fluorescence_nuclei_and_cells"),
            target=is_cfg.get("target", "nuclei"),
        )

    if name == "stardist_finetuned":
        from .stardist_segmenter import StarDistSegmenter
        sd_cfg = seg_cfg.get("stardist_finetuned", {})
        model_path = cfg.get("model_path") or sd_cfg.get("model_path")
        if not model_path:
            raise ValueError(
                "stardist_finetuned requires 'model_path' in config or --model_path argument.\n"
                "Default: finetune/models/stardist_brain_v1"
            )
        raw_tiles = sd_cfg.get("n_tiles", None)
        n_tiles = tuple(raw_tiles) if raw_tiles is not None else None
        return StarDistSegmenter(
            model=str(model_path),        # StarDistSegmenter accepts local dir path
            prob_threshold=sd_cfg.get("prob_threshold", 0.5),
            nms_threshold=sd_cfg.get("nms_threshold", 0.4),
            scale=sd_cfg.get("scale", 0.25),
            n_tiles=n_tiles,
            gaussian_sigma=sd_cfg.get("gaussian_sigma", 5.0),
            min_diam_um=sd_cfg.get("min_diam_um", 3.0),
            max_diam_um=sd_cfg.get("max_diam_um", 20.0),
            min_intensity_ratio=sd_cfg.get("min_intensity_ratio", 1.0),
            pixel_size_um=sd_cfg.get("pixel_size_um", 0.12274),
        )

    if name == "baysor":
        from .baysor_segmenter import BaysorSegmenter
        b_cfg = seg_cfg.get("baysor", {})
        return BaysorSegmenter(
            scale_um=b_cfg.get("scale_um", 10.0),
            min_molecules=b_cfg.get("min_molecules", 15),
            prior_confidence=b_cfg.get("prior_confidence", 0.5),
            n_clusters=b_cfg.get("n_clusters", 4),
            pixel_size_um=b_cfg.get("pixel_size_um", 0.12028),
            baysor_bin=b_cfg.get("baysor_bin", "baysor"),
            prior_dilation_px=b_cfg.get("prior_dilation_px", 15),
            use_z=b_cfg.get("use_z", False),
        )

    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Valid: cellpose_cpsam, cellpose_nuclei, cellpose_cyto3, stardist, "
        "stardist_finetuned, instanseg, baysor"
    )
