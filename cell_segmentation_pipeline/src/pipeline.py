"""SegmentationPipeline: orchestrates I/O → preprocess → segment → mask → QC."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import io as _io
from .masks import expand_mask, mask_to_cell_table, save_colored_mask_png
from .preprocessing import (
    build_tophat_comparison,
    preprocess,
    save_comparison_png,
)
from .qc import (
    aggregate_qc,
    compute_fov_qc,
    plot_cellcount_comparison,
    plot_size_distribution,
    save_qc_panel,
)
from .segmentation import get_segmenter
from .visualization import save_overlay_png


class SegmentationPipeline:
    """End-to-end segmentation pipeline for CosMx morphology images.

    Usage::

        cfg = io.load_config("cell_segmentation_pipeline/config/default_config.yaml")
        pipeline = SegmentationPipeline(cfg)
        pipeline.run_all(
            image_dir="raw_data/pilot_4fov/slide1_RNA/morphology_images",
            output_dir="outputs/cell_segmentation_pipeline",
        )
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._manifest: pd.DataFrame | None = None

        manifest_path = cfg.get("input", {}).get("manifest", "config/sample_manifest.csv")
        if Path(manifest_path).exists():
            self._manifest = _io.load_manifest(manifest_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def run_fov(
        self,
        tif_path: str | Path,
        output_dir: str | Path,
        model_name: str | None = None,
        save_comparison: bool = True,
    ) -> dict:
        """Process a single FOV and save all outputs.

        Args:
            tif_path:        Path to multi-channel TIFF.
            output_dir:      Root output directory.
            model_name:      Segmentation model (overrides config default).
            save_comparison: Whether to save TopHat comparison PNG.

        Returns:
            Per-FOV QC dict (from compute_fov_qc).
        """
        tif_path   = Path(tif_path)
        output_dir = Path(output_dir)
        cfg        = self.cfg

        # ── 1. Load TIFF and extract DAPI channel ──────────────────────────
        fov_id  = _io.parse_fov_id(tif_path)
        ch      = cfg.get("input", {}).get("channel", 0)
        img_all = _io.load_tiff(tif_path)
        raw     = _io.extract_channel(img_all, ch)

        fov_meta = {}
        if self._manifest is not None:
            fov_meta = _io.get_fov_meta(self._manifest, fov_id)

        tag = _build_tag(fov_meta, fov_id)
        print(f"[{fov_id}] Loaded: {tif_path.name}  shape={raw.shape}")

        # ── 2. Preprocess ─────────────────────────────────────────────────
        pre_cfg  = cfg.get("preprocessing", {})
        enhanced = preprocess(raw, pre_cfg)

        enhanced_path = output_dir / "enhanced" / f"{tag}_enhanced.tif"
        _io.save_tiff(enhanced, enhanced_path)
        print(f"[{fov_id}] Enhanced → {enhanced_path}")

        if save_comparison:
            radii = pre_cfg.get("comparison_radii", [10, 15, 20, 25])
            cmp_imgs = build_tophat_comparison(raw, radii)
            cmp_path = output_dir / "enhanced" / f"{tag}_tophat_comparison.png"
            save_comparison_png(cmp_imgs, cmp_path, title=f"{fov_id} TopHat comparison")

        # ── 3. Segmentation ────────────────────────────────────────────────
        if model_name is None:
            model_name = cfg.get("segmentation", {}).get("default_model", "cellpose_cpsam")

        segmenter = get_segmenter(model_name, cfg)
        print(f"[{fov_id}] Segmenting with {segmenter.name} …")
        mask = segmenter.segment(enhanced)
        print(f"[{fov_id}] → {int(mask.max())} nuclei detected")

        mask_path = output_dir / "masks" / f"{tag}_{segmenter.name}_nuclear_mask.tif"
        _io.save_mask_tiff(mask, mask_path)

        # ── 4. Mask expansion ──────────────────────────────────────────────
        exp_cfg    = cfg.get("segmentation", {}).get("expansion", {})
        expand_px  = exp_cfg.get("expand_px", 5)
        min_area   = exp_cfg.get("min_area_px", 50)
        exp_mask   = expand_mask(mask, expand_px=expand_px, min_area_px=min_area)

        exp_path = output_dir / "expanded_masks" / f"{tag}_{segmenter.name}_expanded_mask.tif"
        _io.save_mask_tiff(exp_mask, exp_path)

        # ── 5. Cell table ──────────────────────────────────────────────────
        cell_table = mask_to_cell_table(mask, raw, fov_meta)
        csv_path   = output_dir / "cell_tables" / f"{tag}_{segmenter.name}_cells.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cell_table.to_csv(str(csv_path), index=False)
        print(f"[{fov_id}] Cell table → {csv_path}  ({len(cell_table)} rows)")

        # ── 6. Overlays ────────────────────────────────────────────────────
        ov_path = output_dir / "overlays" / f"{tag}_{segmenter.name}_overlay.png"
        save_overlay_png(enhanced, mask, ov_path)

        col_path = output_dir / "overlays" / f"{tag}_{segmenter.name}_colored.png"
        save_colored_mask_png(mask, col_path)

        # ── 7. QC panel ────────────────────────────────────────────────────
        panel_path = output_dir / "qc" / "panels" / f"{tag}_{segmenter.name}_qc_panel.png"
        save_qc_panel(raw, enhanced, mask, exp_mask, panel_path, fov_id, segmenter.name)

        dist_path = output_dir / "qc" / "panels" / f"{tag}_{segmenter.name}_size_dist.png"
        plot_size_distribution(cell_table, dist_path, fov_id, segmenter.name)

        # ── 8. Return QC metrics ────────────────────────────────────────────
        return compute_fov_qc(cell_table, cfg, fov_id=fov_id, model_name=segmenter.name)

    def run_all(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        fov_ids: list[str] | None = None,
        model_name: str | None = None,
        save_comparison: bool = True,
    ) -> list[dict]:
        """Process all (or selected) FOVs and write aggregate QC report.

        Args:
            image_dir:       Directory containing morphology TIFFs.
            output_dir:      Root output directory.
            fov_ids:         Optional list of FOV IDs to process (None = all).
            model_name:      Segmentation model (overrides config default).
            save_comparison: Whether to save TopHat comparison PNGs.

        Returns:
            List of per-FOV QC dicts.
        """
        image_dir  = Path(image_dir)
        output_dir = Path(output_dir)

        tif_paths = _io.list_fov_tiffs(image_dir)
        if not tif_paths:
            raise FileNotFoundError(f"No TIF files found in {image_dir}")

        tif_paths = _io.filter_fovs(tif_paths, fov_ids)
        print(f"Processing {len(tif_paths)} FOV(s) from {image_dir}")

        qc_results = []
        for tif_path in tif_paths:
            qc = self.run_fov(
                tif_path,
                output_dir,
                model_name=model_name,
                save_comparison=save_comparison,
            )
            qc_results.append(qc)

        # ── Aggregate QC report ────────────────────────────────────────────
        summary = aggregate_qc(qc_results)
        qc_dir  = output_dir / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)

        summary_path = qc_dir / "qc_summary.csv"
        summary.to_csv(str(summary_path), index=False)
        print(f"QC summary → {summary_path}")

        cc_path = qc_dir / "cellcount_comparison.png"
        plot_cellcount_comparison(summary, cc_path)

        return qc_results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_tag(fov_meta: dict, fov_id: str) -> str:
    """Build filename tag, e.g. 'FOV00001_AD_F'."""
    parts = [fov_id]
    if fov_meta.get("condition"):
        parts.append(str(fov_meta["condition"]))
    if fov_meta.get("region"):
        parts.append(str(fov_meta["region"]))
    return "_".join(parts)
