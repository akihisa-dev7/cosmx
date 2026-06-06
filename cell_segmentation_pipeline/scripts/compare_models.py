#!/usr/bin/env python3
"""CLI: run multiple segmentation models on the same FOV(s) and compare.

Example::

    python cell_segmentation_pipeline/scripts/compare_models.py \\
        --input  raw_data/pilot_4fov/slide1_RNA/morphology_images \\
        --models cellpose_cpsam cellpose_nuclei stardist \\
        --output outputs/cell_segmentation_pipeline/model_comparison \\
        --fov    FOV00001
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cell_segmentation_pipeline.src import io as _io
from cell_segmentation_pipeline.src.masks import expand_mask, mask_to_cell_table
from cell_segmentation_pipeline.src.preprocessing import preprocess
from cell_segmentation_pipeline.src.qc import aggregate_qc, compute_fov_qc
from cell_segmentation_pipeline.src.segmentation import get_segmenter
from cell_segmentation_pipeline.src.visualization import (
    make_model_comparison_grid,
    save_overlay_png,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare segmentation models on CosMx images.")
    p.add_argument("--input",   required=True, help="Directory of morphology TIFFs")
    p.add_argument("--models",  required=True, nargs="+",
                   help="Model names (e.g. cellpose_cpsam cellpose_nuclei stardist)")
    p.add_argument("--output",  required=True, help="Output directory for comparison results")
    p.add_argument("--fov",     default=None,  help="Single FOV ID (default: all)")
    p.add_argument("--channel", type=int, default=0, help="DAPI channel index (default: 0)")
    p.add_argument("--diameter",           type=float, default=25.0)
    p.add_argument("--cellprob_threshold", type=float, default=-1.0)
    p.add_argument("--flow_threshold",     type=float, default=0.6)
    p.add_argument("--expand_px",          type=int,   default=5)
    p.add_argument("--manifest", default=None)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    in_dir = Path(args.input)
    out    = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    manifest = None
    if args.manifest and Path(args.manifest).exists():
        manifest = _io.load_manifest(args.manifest)
    elif Path("config/sample_manifest.csv").exists():
        manifest = _io.load_manifest("config/sample_manifest.csv")

    cfg = {
        "input": {"channel": args.channel},
        "segmentation": {
            "cellpose": {
                "cpsam":  {"diameter": args.diameter, "cellprob_threshold": args.cellprob_threshold, "flow_threshold": args.flow_threshold},
                "nuclei": {"diameter": 20, "cellprob_threshold": args.cellprob_threshold, "flow_threshold": args.flow_threshold},
                "cyto3":  {"diameter": 28, "cellprob_threshold": args.cellprob_threshold, "flow_threshold": args.flow_threshold},
            },
            "stardist": {},
            "instanseg": {},
            "expansion": {"expand_px": args.expand_px, "min_area_px": 50},
        },
        "qc": {"min_cell_area_px": 50, "max_cell_area_px": 10000},
        "preprocessing": {"method": "tophat", "tophat_radius_px": 20},
    }

    pre_cfg = {"method": "tophat", "tophat_radius_px": 20}

    tif_paths = _io.list_fov_tiffs(in_dir)
    fov_filter = [args.fov] if args.fov else None
    tif_paths  = _io.filter_fovs(tif_paths, fov_filter)

    if not tif_paths:
        print(f"ERROR: No TIF files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    all_qc: list[dict] = []

    for tif_path in tif_paths:
        fov_id  = _io.parse_fov_id(tif_path)
        img_all = _io.load_tiff(tif_path)
        raw     = _io.extract_channel(img_all, args.channel) if img_all.ndim == 3 else img_all
        enhanced = preprocess(raw, pre_cfg)

        fov_meta = {}
        if manifest is not None:
            fov_meta = _io.get_fov_meta(manifest, fov_id)

        print(f"\n=== {fov_id} ===")
        grid_results = []

        for model_name in args.models:
            try:
                seg = get_segmenter(model_name, cfg)
                print(f"  Running {seg.name} …", end=" ", flush=True)
                mask = seg.segment(enhanced)
                n    = int(mask.max())
                print(f"{n} cells")
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            exp_mask   = expand_mask(mask, expand_px=args.expand_px)
            cell_table = mask_to_cell_table(mask, raw, fov_meta)

            # Save per-model outputs
            tag = f"{fov_id}_{seg.name}"
            _io.save_mask_tiff(mask,     out / "masks"          / f"{tag}_nuclear_mask.tif")
            _io.save_mask_tiff(exp_mask, out / "expanded_masks"  / f"{tag}_expanded_mask.tif")
            cell_table.to_csv(str(out / "cell_tables" / f"{tag}_cells.csv"), index=False)
            csv_path = out / "cell_tables" / f"{tag}_cells.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            cell_table.to_csv(str(csv_path), index=False)

            save_overlay_png(enhanced, mask, out / "overlays" / f"{tag}_overlay.png")

            qc = compute_fov_qc(cell_table, cfg, fov_id=fov_id, model_name=seg.name)
            all_qc.append(qc)
            grid_results.append({"model_name": seg.name, "image": enhanced, "mask": mask, "n_cells": n})

        if grid_results:
            grid_path = out / f"{fov_id}_model_comparison.png"
            make_model_comparison_grid(grid_results, grid_path)
            print(f"  Comparison grid → {grid_path}")

    if all_qc:
        summary = aggregate_qc(all_qc)
        summary.to_csv(str(out / "comparison_summary.csv"), index=False)
        print(f"\nSummary:\n{summary.to_string(index=False)}")
        print(f"\nSaved → {out / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
