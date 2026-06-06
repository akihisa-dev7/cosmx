#!/usr/bin/env python3
"""CLI: segment enhanced CosMx images and generate masks + cell tables.

Example::

    python cell_segmentation_pipeline/scripts/run_segmentation.py \\
        --input  outputs/cell_segmentation_pipeline/enhanced \\
        --output outputs/cell_segmentation_pipeline \\
        --model  cellpose_cpsam \\
        --diameter 25 --cellprob_threshold -1.0 --flow_threshold 0.6 \\
        --expand_px 5

Notes:
    - --input can be either a directory of enhanced TIFFs (from run_preprocess.py)
      OR the original morphology_images directory (raw DAPI used directly).
    - --manifest can point to the project sample_manifest.csv for metadata tagging.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cell_segmentation_pipeline.src import io as _io
from cell_segmentation_pipeline.src.masks import (
    expand_mask,
    expand_mask_membrane_guided,
    mask_to_cell_table,
    save_colored_mask_png,
)
from cell_segmentation_pipeline.src.qc import (
    compute_fov_qc,
    save_qc_panel,
    plot_size_distribution,
    aggregate_qc,
    plot_cellcount_comparison,
)
from cell_segmentation_pipeline.src.segmentation import get_segmenter
from cell_segmentation_pipeline.src.visualization import save_overlay_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment CosMx DAPI images.")
    p.add_argument("--input",   required=True, help="Directory of (enhanced) TIFF images")
    p.add_argument("--output",  required=True, help="Root output directory")
    p.add_argument("--model",   default="cellpose_cpsam",
                   help="Segmentation model (cellpose_cpsam | cellpose_nuclei | "
                        "cellpose_cyto3 | stardist | instanseg)")
    p.add_argument("--diameter",           type=float, default=25.0)
    p.add_argument("--cellprob_threshold", type=float, default=-1.0)
    p.add_argument("--flow_threshold",     type=float, default=0.6)
    p.add_argument("--expand_px",          type=int,   default=5,
                   help="Mask expansion distance in px (default: 5)")
    p.add_argument("--min_area_px",        type=int,   default=50,
                   help="Minimum cell area in px² (default: 50)")
    p.add_argument("--channel",            type=int,   default=0,
                   help="Input channel (default: 0 = DAPI). "
                        "Ignored if --input contains single-channel TIFFs.")
    p.add_argument("--membrane_channel",   type=int,   default=None,
                   help="Membrane channel index for guided expansion (default: None=disabled). "
                        "Use 3 for CosMx 5-channel TIFs. Requires raw morphology_images input.")
    p.add_argument("--manifest", default=None,
                   help="Path to sample_manifest.csv for metadata tagging")
    p.add_argument("--fov", default=None, help="Single FOV ID to process (default: all)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    in_dir = Path(args.input)
    out    = Path(args.output)

    manifest = None
    if args.manifest and Path(args.manifest).exists():
        manifest = _io.load_manifest(args.manifest)
    elif Path("config/sample_manifest.csv").exists():
        manifest = _io.load_manifest("config/sample_manifest.csv")

    # Build a minimal config to reuse get_segmenter
    cfg = {
        "input": {"channel": args.channel},
        "segmentation": {
            "default_model": args.model,
            "cellpose": {
                "cpsam":   {"diameter": args.diameter, "cellprob_threshold": args.cellprob_threshold, "flow_threshold": args.flow_threshold},
                "nuclei":  {"diameter": args.diameter, "cellprob_threshold": args.cellprob_threshold, "flow_threshold": args.flow_threshold},
                "cyto3":   {"diameter": args.diameter, "cellprob_threshold": args.cellprob_threshold, "flow_threshold": args.flow_threshold},
            },
            "stardist": {},
            "instanseg": {},
            "expansion": {"expand_px": args.expand_px, "min_area_px": args.min_area_px},
        },
        "qc": {"min_cell_area_px": args.min_area_px, "max_cell_area_px": 10000},
    }

    segmenter = get_segmenter(args.model, cfg)
    print(f"Segmenter: {segmenter.name}")

    tif_paths = _io.list_fov_tiffs(in_dir)
    fov_filter = [args.fov] if args.fov else None
    tif_paths  = _io.filter_fovs(tif_paths, fov_filter)

    if not tif_paths:
        print(f"ERROR: No TIF files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(tif_paths)} FOV(s) …")
    qc_results = []

    # Multi-channel mode flag: cpsam_membrane uses DAPI(ch0) + Membrane(ch3)
    _MULTICHANNEL_MODELS = {"cellpose_cpsam_membrane"}
    is_multichannel = args.model in _MULTICHANNEL_MODELS

    for tif_path in tif_paths:
        fov_id  = _io.parse_fov_id(tif_path)
        img_all = _io.load_tiff(tif_path)

        if is_multichannel and img_all.ndim == 3:
            # DAPI(ch0) + Membrane(ch3) → TopHat each → normalise → stack (H,W,2)
            from skimage.morphology import disk, white_tophat
            ch_nuc = img_all[0].astype(np.float32)   # DAPI
            ch_mem = img_all[3].astype(np.float32)   # Membrane

            th_nuc = white_tophat(ch_nuc, disk(20))
            th_mem = white_tophat(ch_mem, disk(20))

            def _norm01(x):
                mn, mx = float(x.min()), float(x.max())
                return (x - mn) / (mx - mn + 1e-8)

            # Cellpose channels=[1,2]: ch-index-0=cytoplasm/membrane, ch-index-1=nucleus
            raw = np.stack([_norm01(th_mem), _norm01(th_nuc)], axis=-1)
        elif img_all.ndim == 3:
            raw = _io.extract_channel(img_all, args.channel)
        else:
            raw = img_all

        fov_meta = {}
        if manifest is not None:
            fov_meta = _io.get_fov_meta(manifest, fov_id)

        tag = f"{fov_id}"
        if fov_meta.get("condition"):
            tag += f"_{fov_meta['condition']}"
        if fov_meta.get("region"):
            tag += f"_{fov_meta['region']}"

        print(f"  [{fov_id}] segmenting …")
        mask = segmenter.segment(raw)
        n_cells = int(mask.max())
        print(f"  [{fov_id}] → {n_cells} nuclei")

        _io.save_mask_tiff(mask, out / "masks" / f"{tag}_{segmenter.name}_nuclear_mask.tif")

        # Membrane-guided expansion if --membrane_channel is given and raw TIF has it
        if args.membrane_channel is not None and img_all.ndim == 3 and img_all.shape[0] > args.membrane_channel:
            mem_raw = img_all[args.membrane_channel]
            print(f"  [{fov_id}] expanding with Membrane ch{args.membrane_channel} guidance …")
            exp_mask = expand_mask_membrane_guided(
                mask, mem_raw,
                max_expand_px=args.expand_px * 2,   # allow up to 2× the requested distance
                tophat_radius=20,
                min_area_px=args.min_area_px,
            )
        else:
            exp_mask = expand_mask(mask, expand_px=args.expand_px, min_area_px=args.min_area_px)

        _io.save_mask_tiff(exp_mask, out / "expanded_masks" / f"{tag}_{segmenter.name}_expanded_mask.tif")

        # For cell_table intensity metrics, use raw DAPI (not the stacked multi-ch array)
        raw_for_table = img_all[args.channel] if img_all.ndim == 3 else raw
        cell_table = mask_to_cell_table(mask, raw_for_table, fov_meta)
        csv_path   = out / "cell_tables" / f"{tag}_{segmenter.name}_cells.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cell_table.to_csv(str(csv_path), index=False)

        save_overlay_png(raw_for_table, mask, out / "overlays" / f"{tag}_{segmenter.name}_overlay.png")
        save_colored_mask_png(mask,   out / "overlays" / f"{tag}_{segmenter.name}_colored.png")

        save_qc_panel(
            raw_for_table, raw_for_table, mask, exp_mask,
            out / "qc" / "panels" / f"{tag}_{segmenter.name}_qc_panel.png",
            fov_id, segmenter.name,
        )
        plot_size_distribution(
            cell_table,
            out / "qc" / "panels" / f"{tag}_{segmenter.name}_size_dist.png",
            fov_id, segmenter.name,
        )

        qc = compute_fov_qc(cell_table, cfg, fov_id=fov_id, model_name=segmenter.name)
        qc_results.append(qc)
        med = qc.get("median_diam_px") if qc else None
        med_str = f"{med:.1f}" if med is not None else "N/A"
        print(f"  [{fov_id}] done  n_cells={n_cells}  median_diam={med_str}px")

    # Aggregate summary
    summary = aggregate_qc(qc_results)
    qc_dir  = out / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(str(qc_dir / "qc_summary.csv"), index=False)
    plot_cellcount_comparison(summary, qc_dir / "cellcount_comparison.png")

    print(f"\nSummary:\n{summary.to_string(index=False)}")
    print(f"\nQC summary → {qc_dir / 'qc_summary.csv'}")


if __name__ == "__main__":
    main()
