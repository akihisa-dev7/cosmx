"""StarDist segmenter: 2D_versatile_fluo, tuned for CosMx 0.12 µm/px DAPI.

Key parameters for CosMx:
  - gaussian_sigma: Gaussian blur (px) applied BEFORE normalization.
    At CosMx 0.12µm/px, nuclei are ~57px (7µm) but contain multiple
    bright chromatin spots (~10px). sigma=10 merges the spots into one
    bright blob per nucleus so the model delineates the full nucleus.
  - scale: StarDist rescales the image by this factor before inference.
    2D_versatile_fluo was trained at ~0.5µm/px (nucleus ~14-25px).
    CosMx at 0.12µm/px has nuclei ~57px — 2-4x larger than training.
    scale=0.5 makes nuclei appear ~28px, inside the training range.
    Theoretical optimum: 0.12274/0.5 ≈ 0.245.
  - min_diam_um / max_diam_um: post-inference size filter in µm.
    Removes false positives from background blobs and large artifacts.
    At CosMx, brain nuclei are 4-15µm; defaults conservatively cover 3-20µm.
  - min_intensity_ratio: post-inference intensity filter.
    Each detected cell's mean DAPI intensity (measured on the Gaussian-blurred
    image) must be ≥ image_mean × min_intensity_ratio to be kept.
    0.0 = disabled. 1.5 recommended for CosMx: removes low-intensity
    background blobs while retaining dimmer out-of-focus nuclei.
  - pixel_size_um: physical pixel size for the size filter calculation.
    Default 0.12274 µm/px for CosMx.
"""
from __future__ import annotations

import math

import numpy as np

from .base import BaseSegmenter


class StarDistSegmenter(BaseSegmenter):
    """Wrapper around StarDist2D pretrained models, tuned for CosMx DAPI.

    Args:
        model:               Pretrained model name ('2D_versatile_fluo').
        prob_threshold:      Detection confidence threshold (lower = more cells).
        nms_threshold:       Non-maximum suppression IoU threshold.
        scale:               Rescale factor applied by StarDist before inference.
                             0.5 maps CosMx 57px nuclei to ~28px, matching the
                             training distribution of 2D_versatile_fluo (~14-25px).
        n_tiles:             Tile grid to avoid OOM on large images. None = auto
                             (computed from image size and scale at inference time).
        gaussian_sigma:      Std-dev (px) of Gaussian blur applied before
                             normalization. Merges sub-nuclear chromatin spots into
                             whole-nucleus blobs. 10px works for scale 0.5-1.0.
        min_diam_um:         Minimum nucleus diameter to keep (µm).
        max_diam_um:         Maximum nucleus diameter to keep (µm).
        min_intensity_ratio: Keep cells whose mean blurred-DAPI intensity is ≥
                             image_mean × this value. 0.0 = disabled.
                             1.5 is a good starting point for CosMx DAPI.
        pixel_size_um:       Physical pixel size (µm/px). Used for size filter.
    """

    def __init__(
        self,
        model: str = "2D_versatile_fluo",
        prob_threshold: float = 0.3,
        nms_threshold: float = 0.4,
        scale: float = 0.5,
        n_tiles: tuple | None = None,
        gaussian_sigma: float = 5.0,
        min_diam_um: float = 4.0,
        max_diam_um: float = 30.0,
        min_intensity_ratio: float = 1.0,
        pixel_size_um: float = 0.12274,
    ) -> None:
        self._model_name = model
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        self.scale = scale
        self.n_tiles = n_tiles
        self.gaussian_sigma = gaussian_sigma
        self.min_diam_um = min_diam_um
        self.max_diam_um = max_diam_um
        self.min_intensity_ratio = min_intensity_ratio
        self.pixel_size_um = pixel_size_um
        self._model = None

    def _load_model(self) -> None:
        from pathlib import Path
        from stardist.models import StarDist2D
        p = Path(self._model_name)
        if p.exists() and p.is_dir():
            # Load fine-tuned model from local directory
            self._model = StarDist2D(None, name=p.name, basedir=str(p.parent))
        else:
            self._model = StarDist2D.from_pretrained(self._model_name)

    def _auto_n_tiles(self, h: int, w: int) -> tuple[int, int]:
        # Target ~1100px per tile in the scaled image (consistent with the
        # prior (4,4) default for 4256px at scale=1.0: 4256/4=1064px/tile).
        scaled_h = h * self.scale
        scaled_w = w * self.scale
        nt_h = max(1, math.ceil(scaled_h / 1100))
        nt_w = max(1, math.ceil(scaled_w / 1100))
        return (nt_h, nt_w)

    def _filter_cells(
        self,
        labels: np.ndarray,
        blurred_image: np.ndarray,
    ) -> np.ndarray:
        """Remove detections that fail size or intensity criteria.

        Args:
            labels:        Integer label mask from StarDist.
            blurred_image: Gaussian-blurred DAPI image (float32, pre-normalization).
                           Used for per-cell mean intensity measurement.

        Returns:
            Filtered label mask with consecutive labels starting at 1.
        """
        from skimage.measure import regionprops
        from skimage.segmentation import relabel_sequential

        # intensity_image lets regionprops compute mean_intensity per cell
        props = regionprops(labels, intensity_image=blurred_image)

        img_mean = float(blurred_image.mean())
        intensity_threshold = img_mean * self.min_intensity_ratio  # 0 when ratio=0

        keep = []
        for r in props:
            diam_um = 2.0 * math.sqrt(r.area / math.pi) * self.pixel_size_um
            if not (self.min_diam_um <= diam_um <= self.max_diam_um):
                continue
            mean_int = getattr(r, "intensity_mean", None) or getattr(r, "mean_intensity", 0.0)
            if self.min_intensity_ratio > 0 and mean_int < intensity_threshold:
                continue
            keep.append(r.label)

        filtered = np.where(np.isin(labels, keep), labels, 0)
        filtered, _, _ = relabel_sequential(filtered)
        return filtered.astype(np.int32)

    @property
    def name(self) -> str:
        scale_tag = f"_s{str(self.scale).replace('.', '')}"
        return f"stardist_g{int(self.gaussian_sigma)}{scale_tag}"

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment a 2D grayscale image. Returns integer label mask.

        Preprocessing pipeline:
          1. Gaussian blur (gaussian_sigma px) to merge chromatin spots.
          2. csbdeep percentile normalization clipped to [0, 1].
          3. StarDist predict_instances (with internal rescaling by `scale`).
          4. Combined size + intensity filter on the blurred DAPI image.
        """
        if self._model is None:
            self._load_model()

        from csbdeep.utils import normalize
        from skimage.filters import gaussian

        img_f = image.astype(np.float32)

        # Step 1: Gaussian blur to merge sub-nuclear chromatin foci.
        # The blurred image is retained for intensity filtering in step 4.
        if self.gaussian_sigma > 0:
            img_blurred = gaussian(img_f, sigma=self.gaussian_sigma).astype(np.float32)
        else:
            img_blurred = img_f.copy()

        # Step 2: csbdeep percentile normalization → clip to [0, 1]
        img_norm = normalize(img_blurred, 2, 99.8)
        img_norm = np.clip(img_norm, 0.0, 1.0)

        # Step 3: StarDist inference
        n_tiles = self.n_tiles if self.n_tiles is not None else self._auto_n_tiles(*img_norm.shape[:2])
        labels, _ = self._model.predict_instances(
            img_norm,
            prob_thresh=self.prob_threshold,
            nms_thresh=self.nms_threshold,
            scale=self.scale,
            n_tiles=n_tiles,
        )

        # Step 4: Size + intensity filter using the pre-normalization blurred image.
        labels = self._filter_cells(labels.astype(np.int32), img_blurred)
        return labels
