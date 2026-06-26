#!/usr/bin/env python3
"""Run Baysor on existing DAPI masks for FOV00001 and produce a comparison report.

Usage:
    python cell_segmentation_pipeline/scripts/run_baysor_comparison.py \
        [--fov FOV00001] [--models cpsam stardist_g10_s025]

Compares DAPI-only vs Baysor-refined segmentation for cellpose_cpsam and stardist.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fov", default="FOV00001", help="FOV ID to process")
    p.add_argument("--models", nargs="+",
                   default=["cpsam", "stardist_g10_s025"],
                   help="Base mask model name suffixes (e.g. cpsam stardist_g10_s025)")
    p.add_argument("--output_root", default=str(ROOT / "outputs" / "pilot_4fov"),
                   help="Root output dir containing masks/")
    p.add_argument("--tx_csv_dir", default=str(ROOT / "outputs_ssd" / "proseg_pilot"),
                   help="Directory containing per-FOV tx CSV files")
    p.add_argument("--scale_um", type=float, default=10.0)
    p.add_argument("--min_molecules", type=int, default=15)
    p.add_argument("--prior_confidence", type=float, default=0.5)
    p.add_argument("--baysor_bin", default=os.path.expanduser("~/.local/bin/baysor"))
    p.add_argument("--z_planes", nargs="*", type=int, default=None,
                   help="Z-planes to use (e.g. 1 2 3). None = all z-planes. "
                        "Use a subset to avoid Julia GC crashes on very large datasets.")
    p.add_argument("--n_cells_init", type=int, default=2000,
                   help="Initial EM component count. 2000 avoids Julia GC crashes on "
                        "large datasets. 0 = auto (Baysor default, may OOM/crash).")
    return p.parse_args()


def cell_stats(mask: np.ndarray, pixel_size_um: float = 0.12028) -> dict:
    """Return dict of cell count and diameter statistics from a mask."""
    n = int(mask.max())
    if n == 0:
        return {"n_cells": 0, "median_diam_um": 0, "p10_diam_um": 0, "p90_diam_um": 0}
    areas = []
    for lbl in range(1, n + 1):
        areas.append(float((mask == lbl).sum()))
    diams = [2 * math.sqrt(a / math.pi) * pixel_size_um for a in areas]
    return {
        "n_cells": n,
        "median_diam_um": float(np.median(diams)),
        "p10_diam_um": float(np.percentile(diams, 10)),
        "p90_diam_um": float(np.percentile(diams, 90)),
        "mean_area_px2": float(np.mean(areas)),
    }


def main():
    args = parse_args()
    # Julia GC settings: single thread + single GC thread avoids Julia 1.10 GC segfaults
    os.environ.setdefault("JULIA_NUM_THREADS", "1")
    os.environ.setdefault("JULIA_NUM_GC_THREADS", "1")

    import tifffile
    from cell_segmentation_pipeline.src.segmentation.baysor_segmenter import BaysorSegmenter

    out_root = Path(args.output_root)
    tx_csv_dir = Path(args.tx_csv_dir)
    fov_id = args.fov
    fov_num = int("".join(c for c in fov_id if c.isdigit()))

    tx_csv = tx_csv_dir / f"{fov_id}_tx.csv.gz"
    if not tx_csv.exists():
        print(f"ERROR: tx CSV not found: {tx_csv}")
        sys.exit(1)

    results = {}
    for model_suffix in args.models:
        # Find the mask file
        mask_candidates = list((out_root / "masks").glob(f"{fov_id}_*{model_suffix}_nuclear_mask.tif"))
        if not mask_candidates:
            print(f"WARNING: no mask found for {fov_id} / {model_suffix}, skipping")
            continue
        prior_path = mask_candidates[0]
        print(f"\n{'='*60}")
        print(f"Model: {model_suffix}  |  Prior: {prior_path.name}")
        print(f"{'='*60}")

        # DAPI-only stats
        dapi_mask = tifffile.imread(str(prior_path)).astype(np.int32)
        dapi_stats = cell_stats(dapi_mask)
        print(f"  DAPI-only: {dapi_stats['n_cells']} cells, "
              f"median_diam={dapi_stats['median_diam_um']:.1f}µm "
              f"(p10={dapi_stats['p10_diam_um']:.1f}, p90={dapi_stats['p90_diam_um']:.1f})")

        # Run Baysor
        baysor_out = out_root / "baysor" / fov_id / model_suffix
        z_tag = f"_z{''.join(str(z) for z in args.z_planes)}" if args.z_planes else ""
        seg = BaysorSegmenter(
            scale_um=args.scale_um,
            min_molecules=args.min_molecules,
            prior_confidence=args.prior_confidence,
            baysor_bin=args.baysor_bin,
            z_planes=args.z_planes,
            n_cells_init=args.n_cells_init,
        )
        z_info = f", z_planes={args.z_planes}" if args.z_planes else ""
        print(f"  Running Baysor (scale={args.scale_um}µm, min_mol={args.min_molecules}, "
              f"prior_conf={args.prior_confidence}{z_info}) ...")
        try:
            mask_path = seg.run_fov(
                tx_csv=tx_csv,
                prior_mask=prior_path,
                out_dir=baysor_out,
                fov_col="fov",
                fov_id=fov_num,
                image_shape=dapi_mask.shape[:2],
            )
            baysor_mask = tifffile.imread(str(mask_path)).astype(np.int32)
            baysor_stats = cell_stats(baysor_mask)
            print(f"  Baysor:     {baysor_stats['n_cells']} cells, "
                  f"median_diam={baysor_stats['median_diam_um']:.1f}µm "
                  f"(p10={baysor_stats['p10_diam_um']:.1f}, p90={baysor_stats['p90_diam_um']:.1f})")

            # Save refined mask to standard masks dir
            tag = prior_path.name.replace(f"_{model_suffix}_nuclear_mask.tif", "")
            refined_mask_path = out_root / "masks" / f"{tag}_{model_suffix}_baysor_nuclear_mask.tif"
            tifffile.imwrite(str(refined_mask_path), baysor_mask.astype(np.uint16))
            print(f"  Saved refined mask: {refined_mask_path.name}")

        except Exception as e:
            print(f"  ERROR during Baysor run: {e}")
            import traceback; traceback.print_exc()
            baysor_stats = {"error": str(e)}

        results[model_suffix] = {
            "fov": fov_id,
            "prior_mask": prior_path.name,
            "dapi_only": dapi_stats,
            "baysor": baysor_stats,
        }

    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Metric':<20} {'DAPI-only':>12} {'Baysor':>12} {'Change':>12}")
    print("-" * 82)
    for model, r in results.items():
        d = r["dapi_only"]
        b = r.get("baysor", {})
        if "error" in b:
            print(f"{model:<25}  ERROR: {b['error']}")
            continue
        for metric in ["n_cells", "median_diam_um", "p10_diam_um", "p90_diam_um"]:
            dv = d.get(metric, 0)
            bv = b.get(metric, 0)
            pct = f"{(bv-dv)/max(dv,1)*100:+.1f}%" if dv != 0 else "N/A"
            print(f"{model:<25} {metric:<20} {dv:>12.1f} {bv:>12.1f} {pct:>12}")
        print()

    # Save JSON results
    report_path = out_root / "baysor_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved: {report_path}")


if __name__ == "__main__":
    main()
