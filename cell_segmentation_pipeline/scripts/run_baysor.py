#!/usr/bin/env python3
"""Run Baysor RNA-guided segmentation on CosMx pilot FOVs.

Baysor jointly optimises transcript spatial density and a Cellpose nuclear
prior to produce more accurate cell boundaries than DAPI-only segmentation.

Prerequisites:
    1. Baysor binary installed:
       wget https://github.com/kharchenkolab/Baysor/releases/latest/download/baysor
       chmod +x baysor && sudo mv baysor /usr/local/bin/

    2. Run from the CosMx_2026 project root:
       source cell_segmentation_pipeline/.venv/bin/activate
       pip install anndata  # if not already installed

Examples:
    # Pilot 4 FOVs with Cellpose cpsam as prior
    python cell_segmentation_pipeline/scripts/run_baysor.py \\
        --tx_dir   outputs_ssd/proseg_pilot/ \\
        --prior_dir outputs/pilot_4fov/masks/ \\
        --output   outputs/baysor/ \\
        --fovs FOV00001 FOV00007 FOV00037 FOV00043

    # Single FOV with custom parameters
    python cell_segmentation_pipeline/scripts/run_baysor.py \\
        --tx_dir   outputs_ssd/proseg_pilot/ \\
        --prior_dir outputs/pilot_4fov/masks/ \\
        --output   outputs/baysor/ \\
        --fovs FOV00001 \\
        --scale 10 --min_molecules 20 --prior_confidence 0.6

    # Use CosMx native CellLabels as prior (fallback)
    python cell_segmentation_pipeline/scripts/run_baysor.py \\
        --tx_dir       outputs_ssd/proseg_pilot/ \\
        --prior_dir    raw_data/pilot_4fov/slide1_RNA/per_fov_decoded/ \\
        --prior_glob   "*/CellLabels_F*.tif" \\
        --output       outputs/baysor/
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cell_segmentation_pipeline.src.segmentation.baysor_segmenter import BaysorSegmenter
from cell_segmentation_pipeline.src.qc import compute_fov_qc


# ── FOV ID parsing helpers ─────────────────────────────────────────────────────

def _fov_str_to_int(fov_str: str) -> int:
    """'FOV00001' -> 1, 'F00001' -> 1, '1' -> 1"""
    return int(fov_str.lstrip("FOVf0") or "0")


def _find_prior_mask(fov_str: str, prior_dir: Path, prior_glob: str) -> Path | None:
    """Find the best prior mask for a given FOV ID string."""
    fov_num = str(_fov_str_to_int(fov_str))
    fov_padded = fov_str if fov_str.startswith("FOV") else f"FOV{int(fov_num):05d}"

    candidates = sorted(prior_dir.glob(prior_glob))

    # Prefer cpsam masks, then any nuclear mask, then CellLabels
    for keyword in ["cpsam_nuclear", "cpsam", "nuclear_mask", "CellLabels"]:
        for c in candidates:
            if fov_padded in c.name or f"F{int(fov_num):05d}" in c.name:
                if keyword in c.name:
                    return c

    # Fallback: any mask containing the FOV ID
    for c in candidates:
        if fov_padded in c.name or f"F{int(fov_num):05d}" in c.name:
            return c

    return None


def _find_tx_file(fov_str: str, tx_dir: Path) -> Path | None:
    """Find the per-FOV transcript file in tx_dir."""
    fov_padded = fov_str if fov_str.startswith("FOV") else f"FOV{int(fov_str):05d}"
    candidates = (
        list(tx_dir.glob(f"{fov_padded}_tx.csv.gz"))
        + list(tx_dir.glob(f"{fov_padded}_tx.csv"))
        + list(tx_dir.glob(f"*{fov_padded}*tx*.csv.gz"))
        + list(tx_dir.glob(f"*{fov_padded}*tx*.csv"))
    )
    return candidates[0] if candidates else None


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baysor RNA-guided segmentation for CosMx data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--tx_dir", required=True,
        help="Directory containing per-FOV transcript CSV files (e.g. outputs_ssd/proseg_pilot/)",
    )
    p.add_argument(
        "--prior_dir", required=True,
        help="Directory containing prior segmentation mask TIFs (Cellpose cpsam or CellLabels)",
    )
    p.add_argument(
        "--output", required=True,
        help="Root output directory (e.g. outputs/baysor/)",
    )
    p.add_argument(
        "--fovs", nargs="+", default=None,
        help="FOV IDs to process (e.g. FOV00001 FOV00007). Default: auto-detect from tx_dir",
    )
    p.add_argument(
        "--prior_glob", default="*.tif",
        help="Glob pattern relative to prior_dir for finding masks (default: *.tif)",
    )
    p.add_argument("--scale",             type=float, default=10.0,
                   help="Expected cell radius in micrometres (default: 10)")
    p.add_argument("--min_molecules",     type=int,   default=15,
                   help="Min transcripts per cell (default: 15)")
    p.add_argument("--prior_confidence",  type=float, default=0.5,
                   help="Prior segmentation confidence 0-1 (default: 0.5)")
    p.add_argument("--n_clusters",        type=int,   default=4,
                   help="Expression type clusters (default: 4)")
    p.add_argument("--pixel_size_um",     type=float, default=0.12028,
                   help="CosMx pixel size in um/px (default: 0.12028)")
    p.add_argument("--baysor_bin",        default="baysor",
                   help="Path to baysor binary (default: baysor)")
    p.add_argument("--prior_dilation_px", type=int,   default=15,
                   help="Pixels to dilate prior mask for Voronoi boundary (default: 15)")
    p.add_argument("--use_z",             action="store_true",
                   help="Pass z-coordinates to Baysor (default: off, force_2d=true)")
    p.add_argument("--use_docker",        action="store_true",
                   help="Run Baysor via Docker (vpetukhov/baysor:latest) instead of local binary")
    p.add_argument("--dry_run",           action="store_true",
                   help="Validate setup (find files, check binary) without running Baysor")
    p.add_argument(
        "--mode", default="all",
        choices=["all", "prepare", "postprocess"],
        help=(
            "Execution mode for Docker Compose 3-stage workflow:\n"
            "  all         — full pipeline (prepare + baysor + postprocess) in one call\n"
            "  prepare     — write transcripts CSV + TOML config, then print baysor command\n"
            "  postprocess — read Baysor output, build label mask + AnnData h5ad"
        ),
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    tx_dir    = Path(args.tx_dir)
    prior_dir = Path(args.prior_dir)
    out_root  = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    seg = BaysorSegmenter(
        scale_um=args.scale,
        min_molecules=args.min_molecules,
        prior_confidence=args.prior_confidence,
        n_clusters=args.n_clusters,
        pixel_size_um=args.pixel_size_um,
        baysor_bin=args.baysor_bin,
        prior_dilation_px=args.prior_dilation_px,
        use_z=args.use_z,
        use_docker=args.use_docker,
    )

    # Discover FOVs
    if args.fovs:
        fov_ids = args.fovs
    else:
        tx_files = sorted(tx_dir.glob("FOV*_tx.csv.gz")) + sorted(tx_dir.glob("FOV*_tx.csv"))
        fov_ids = [f.name.split("_tx")[0] for f in tx_files]
        if not fov_ids:
            print(f"ERROR: No FOV tx files found in {tx_dir}", file=sys.stderr)
            sys.exit(1)

    run_mode = args.mode  # "all", "prepare", or "postprocess"
    backend  = "Docker" if args.use_docker else f"binary ({args.baysor_bin})"
    print(f"\nBaysor RNA-guided segmentation  [{backend}]  mode={run_mode}")
    print(f"  FOVs:        {fov_ids}")
    print(f"  Scale:       {args.scale} um  ({args.scale / args.pixel_size_um:.0f} px)")
    print(f"  Prior conf:  {args.prior_confidence}")
    print(f"  Min mol:     {args.min_molecules}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Output root: {out_root}\n")

    results = []

    for fov_str in fov_ids:
        print(f"{'='*60}")
        print(f"  FOV: {fov_str}")

        fov_out = out_root / fov_str
        fov_int = _fov_str_to_int(fov_str)

        # ── postprocess mode: no tx_file/prior_mask needed ────────────────────
        if run_mode == "postprocess":
            try:
                mask_path = seg.postprocess_fov(
                    out_dir=fov_out,
                    fov_id=fov_int,
                )
                import tifffile, numpy as np
                mask = tifffile.imread(str(mask_path))
                n_cells = int(mask.max())
                results.append({"fov": fov_str, "status": "OK", "n_cells": n_cells,
                                 "mask": str(mask_path)})
                print(f"  DONE  n_cells={n_cells}  mask={mask_path.name}")
            except Exception as e:
                print(f"  ERROR in {fov_str}: {e}", file=sys.stderr)
                traceback.print_exc()
                results.append({"fov": fov_str, "status": "ERROR", "n_cells": 0, "error": str(e)})
            continue

        # ── prepare / all: need tx_file and prior_mask ─────────────────────────
        tx_file = _find_tx_file(fov_str, tx_dir)
        if tx_file is None:
            print(f"  WARNING: No tx file found for {fov_str} in {tx_dir} — skipping")
            continue

        prior_mask = _find_prior_mask(fov_str, prior_dir, args.prior_glob)
        if prior_mask is None:
            print(f"  WARNING: No prior mask found for {fov_str} in {prior_dir}")
            print(f"  Using CosMx native CellLabels as fallback ...")
            native_glob = f"**/CellLabels_F{fov_int:05d}.tif"
            native = list(Path("raw_data").rglob(native_glob))
            prior_mask = native[0] if native else None

        if prior_mask:
            print(f"  Prior mask:  {prior_mask.name}")
        else:
            print(f"  WARNING: No prior mask at all — Baysor will run without prior")

        if args.dry_run:
            print(f"  DRY RUN — tx={tx_file.name}, prior={prior_mask.name if prior_mask else 'NONE'}")
            results.append({"fov": fov_str, "status": "DRY_RUN", "n_cells": 0,
                            "tx_file": str(tx_file), "prior": str(prior_mask)})
            continue

        try:
            if run_mode == "prepare":
                seg.prepare_fov(
                    tx_csv=tx_file,
                    prior_mask=prior_mask,
                    out_dir=fov_out,
                    fov_col="fov",
                    fov_id=fov_int,
                )
                results.append({"fov": fov_str, "status": "PREPARED", "n_cells": 0,
                                 "out_dir": str(fov_out)})
            else:
                # mode == "all"
                mask_path = seg.run_fov(
                    tx_csv=tx_file,
                    prior_mask=prior_mask,
                    out_dir=fov_out,
                    fov_col="fov",
                    fov_id=fov_int,
                )
                import tifffile, numpy as np
                mask = tifffile.imread(str(mask_path))
                n_cells = int(mask.max())
                results.append({"fov": fov_str, "status": "OK", "n_cells": n_cells,
                                 "mask": str(mask_path)})
                print(f"  DONE  n_cells={n_cells}  mask={mask_path.name}")

        except Exception as e:
            print(f"  ERROR in {fov_str}: {e}", file=sys.stderr)
            traceback.print_exc()
            results.append({"fov": fov_str, "status": "ERROR", "n_cells": 0, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    summary.to_csv(out_root / "baysor_run_summary.csv", index=False)
    print(f"\nSummary saved -> {out_root / 'baysor_run_summary.csv'}")


if __name__ == "__main__":
    main()
