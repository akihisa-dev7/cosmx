#!/usr/bin/env python3
"""CLI: generate QC reports from existing mask + image outputs.

Example::

    python cell_segmentation_pipeline/scripts/run_qc.py \\
        --mask_dir  outputs/cell_segmentation_pipeline/masks \\
        --image_dir outputs/cell_segmentation_pipeline/enhanced \\
        --output    outputs/cell_segmentation_pipeline/qc
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tifffile
import numpy as np
import pandas as pd

from cell_segmentation_pipeline.src.masks import mask_to_cell_table
from cell_segmentation_pipeline.src.qc import (
    aggregate_qc,
    compute_fov_qc,
    plot_cellcount_comparison,
    plot_size_distribution,
    save_qc_panel,
)
from cell_segmentation_pipeline.src import io as _io


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate QC reports from mask files.")
    p.add_argument("--mask_dir",  required=True, help="Directory of *_nuclear_mask.tif files")
    p.add_argument("--image_dir", required=True, help="Directory of enhanced TIFFs (for panels)")
    p.add_argument("--output",    required=True, help="Output QC directory")
    p.add_argument("--manifest",  default=None,  help="sample_manifest.csv path")
    p.add_argument("--cosmx_ref", default=None,
                   help="CSV with columns 'fov,cosmx_cells' for comparison bars")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    mask_dir  = Path(args.mask_dir)
    image_dir = Path(args.image_dir)
    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = None
    if args.manifest and Path(args.manifest).exists():
        manifest = _io.load_manifest(args.manifest)
    elif Path("config/sample_manifest.csv").exists():
        manifest = _io.load_manifest("config/sample_manifest.csv")

    cosmx_ref = None
    if args.cosmx_ref and Path(args.cosmx_ref).exists():
        ref_df = pd.read_csv(args.cosmx_ref)
        cosmx_ref = dict(zip(ref_df["fov"], ref_df["cosmx_cells"]))

    mask_paths = sorted(mask_dir.glob("*_nuclear_mask.tif"))
    if not mask_paths:
        print(f"ERROR: No *_nuclear_mask.tif found in {mask_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = {"qc": {"min_cell_area_px": 50, "max_cell_area_px": 10000}}
    qc_results = []

    for mp in mask_paths:
        stem = mp.stem                         # e.g. FOV00001_AD_F_cellpose_cpsam_nuclear_mask
        # parse FOV ID and model name from stem
        fov_id = _io.parse_fov_id(mp)
        model_name = stem.replace("_nuclear_mask", "").split("_")[-2] + "_" + stem.replace("_nuclear_mask", "").split("_")[-1] \
            if "_cellpose_" in stem else stem.split("_")[-2]

        mask = tifffile.imread(str(mp)).astype(np.int32)

        # Find matching image
        img_candidates = list(image_dir.glob(f"{fov_id}*.tif")) + list(image_dir.glob(f"{fov_id}*.TIF"))
        raw = tifffile.imread(str(img_candidates[0])) if img_candidates else np.zeros(mask.shape, dtype=np.uint16)
        if raw.ndim == 3:
            raw = raw[0]

        fov_meta = {}
        if manifest is not None:
            fov_meta = _io.get_fov_meta(manifest, fov_id)

        cell_table = mask_to_cell_table(mask, raw, fov_meta)
        qc         = compute_fov_qc(cell_table, cfg, fov_id=fov_id, model_name=model_name)
        qc_results.append(qc)

        tag = mp.stem.replace("_nuclear_mask", "")
        plot_size_distribution(
            cell_table,
            out_dir / "panels" / f"{tag}_size_dist.png",
            fov_id, model_name,
        )
        print(f"  [{fov_id}] n={qc['n_cells']}  median_diam={qc['median_diam_px']:.1f}px")

    summary = aggregate_qc(qc_results)
    summary.to_csv(str(out_dir / "qc_summary.csv"), index=False)
    plot_cellcount_comparison(summary, out_dir / "cellcount_comparison.png", cosmx_ref)

    print(f"\nQC summary → {out_dir / 'qc_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
