#!/usr/bin/env python3
"""CLI: preprocess CosMx morphology images.

Example::

    python cell_segmentation_pipeline/scripts/run_preprocess.py \\
        --input  raw_data/pilot_4fov/slide1_RNA/morphology_images \\
        --output outputs/cell_segmentation_pipeline/enhanced \\
        --channel 0 --method tophat --radius 20 --fov FOV00001
"""
import argparse
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cell_segmentation_pipeline.src import io as _io
from cell_segmentation_pipeline.src.preprocessing import (
    build_tophat_comparison,
    preprocess,
    save_comparison_png,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess CosMx DAPI images.")
    p.add_argument("--input",   required=True, help="Directory of morphology TIFFs")
    p.add_argument("--output",  required=True, help="Output directory for enhanced TIFFs")
    p.add_argument("--channel", type=int, default=0, help="Channel index (default: 0 = DAPI)")
    p.add_argument("--method",  default="tophat",
                   choices=["tophat", "clahe", "rolling_ball", "none"],
                   help="Enhancement method (default: tophat)")
    p.add_argument("--radius",  type=int, default=20, help="TopHat radius in px (default: 20)")
    p.add_argument("--fov",     default=None,
                   help="Single FOV ID to process (e.g. FOV00001). Default: all.")
    p.add_argument("--no-comparison", action="store_true",
                   help="Skip TopHat comparison PNG generation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assemble preprocessing config from CLI args
    pre_cfg = {
        "method": args.method,
        "tophat_radius_px": args.radius,
        "comparison_radii": [10, 15, 20, 25],
    }

    tif_paths = _io.list_fov_tiffs(input_dir)
    fov_filter = [args.fov] if args.fov else None
    tif_paths  = _io.filter_fovs(tif_paths, fov_filter)

    if not tif_paths:
        print(f"ERROR: No TIF files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(tif_paths)} FOV(s) → {output_dir}")

    for tif_path in tif_paths:
        fov_id   = _io.parse_fov_id(tif_path)
        img_all  = _io.load_tiff(tif_path)
        raw      = _io.extract_channel(img_all, args.channel)

        enhanced = preprocess(raw, pre_cfg)

        out_path = output_dir / f"{fov_id}_enhanced.tif"
        _io.save_tiff(enhanced, out_path)
        print(f"  [{fov_id}] saved → {out_path}")

        if not args.no_comparison and args.method == "tophat":
            radii    = pre_cfg["comparison_radii"]
            cmp_imgs = build_tophat_comparison(raw, radii)
            cmp_path = output_dir / f"{fov_id}_tophat_comparison.png"
            save_comparison_png(cmp_imgs, cmp_path, title=f"{fov_id} TopHat comparison")
            print(f"  [{fov_id}] comparison → {cmp_path}")

    print("Done.")


if __name__ == "__main__":
    main()
