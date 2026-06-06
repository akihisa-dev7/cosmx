"""
Napari annotation tool for ground truth mask correction.

Opens DAPI image + existing mask side-by-side.
User edits the Labels layer, then saves corrected ground truth.

Usage (from CosMx_2026/):
    /home/fujiyamaakihisa/miniconda3/envs/cosmx_annotation/bin/python \
        annotation_tools/launch_annotator.py

Options:
    --fov   FOV ID to annotate (default: FOV00001)
    --model Which existing mask to start from: stardist_g10_s025 | cpsam | cosmx
            (default: stardist_g10_s025)
    --crop  Crop a subregion for easier annotation: "y0,x0,h,w" (default: full)
            Example: --crop 1000,1000,1024,1024

Keyboard shortcuts in napari:
    Ctrl+S  →  Save current Labels as ground truth
    L       →  Toggle Labels layer visibility
    [/]     →  Decrease/increase brush size
    E       →  Erase mode
    P       →  Paint mode
    F       →  Fill mode (fill entire connected region)

Saved to: annotation_tools/ground_truth/<fov>_gt_mask.tif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT_ROOT = Path(__file__).parent.parent
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "pilot_4fov" / "enhanced"
MASKS_DIR = PROJECT_ROOT / "outputs" / "pilot_4fov" / "masks"
GT_DIR = Path(__file__).parent / "ground_truth"

# FOV -> sample suffix mapping
FOV_SUFFIX = {
    "FOV00001": "AD_F",
    "FOV00007": "AD_H",
    "FOV00037": "Control_F",
    "FOV00043": "Control_H",
}


def resolve_paths(fov: str, model: str) -> tuple[Path, Path]:
    suffix = FOV_SUFFIX.get(fov.upper())
    if suffix is None:
        sys.exit(f"Unknown FOV: {fov}. Available: {list(FOV_SUFFIX)}")

    dapi_path = ENHANCED_DIR / f"{fov.upper()}_{suffix}_tophat_enhanced.tif"
    mask_path = MASKS_DIR / f"{fov.upper()}_{suffix}_{model}_nuclear_mask.tif"

    if not dapi_path.exists():
        sys.exit(f"DAPI image not found: {dapi_path}")
    if not mask_path.exists():
        # fallback: list available masks for this FOV
        available = list(MASKS_DIR.glob(f"{fov.upper()}_{suffix}_*_nuclear_mask.tif"))
        sys.exit(
            f"Mask not found: {mask_path}\n"
            f"Available masks:\n" + "\n".join(f"  {p.name}" for p in available)
        )
    return dapi_path, mask_path


def parse_crop(crop_str: str | None) -> tuple[int, int, int, int] | None:
    if crop_str is None:
        return None
    parts = [int(x) for x in crop_str.split(",")]
    if len(parts) != 4:
        sys.exit("--crop must be 'y0,x0,h,w'")
    return tuple(parts)


def apply_crop(arr: np.ndarray, crop: tuple | None) -> tuple[np.ndarray, tuple | None]:
    if crop is None:
        return arr, None
    y0, x0, h, w = crop
    return arr[y0:y0+h, x0:x0+w], crop


def main():
    parser = argparse.ArgumentParser(description="Napari ground truth annotator")
    parser.add_argument("--fov", default="FOV00001")
    parser.add_argument("--model", default="stardist_g10_s025",
                        help="stardist_g10_s025 | cpsam | cosmx")
    parser.add_argument("--crop", default=None,
                        help="Crop region: y0,x0,h,w (e.g. 1000,1000,1024,1024)")
    args = parser.parse_args()

    fov = args.fov.upper()
    dapi_path, mask_path = resolve_paths(fov, args.model)
    crop = parse_crop(args.crop)

    print(f"Loading DAPI:  {dapi_path.name}")
    print(f"Loading mask:  {mask_path.name}")

    dapi = tifffile.imread(str(dapi_path)).astype(np.float32)
    mask = tifffile.imread(str(mask_path)).astype(np.int32)

    dapi, _ = apply_crop(dapi, crop)
    mask, _ = apply_crop(mask, crop)

    print(f"Image shape:   {dapi.shape}  (cells in mask: {mask.max()})")
    if crop:
        y0, x0, h, w = crop
        print(f"Crop applied:  y={y0}:{y0+h}, x={x0}:{x0+w}")

    GT_DIR.mkdir(parents=True, exist_ok=True)
    crop_tag = f"_crop{crop[0]}-{crop[1]}-{crop[2]}-{crop[3]}" if crop else ""
    save_path = GT_DIR / f"{fov}_gt_mask{crop_tag}.tif"

    import napari

    viewer = napari.Viewer(title=f"Annotator — {fov} [{args.model}]")
    viewer.add_image(dapi, name="DAPI", colormap="gray", contrast_limits=[0, 1])
    labels_layer = viewer.add_labels(mask, name="annotations")

    print("\n--- Napari opened ---")
    print(f"Edit the 'annotations' layer, then press Ctrl+S to save.")
    print(f"Save path: {save_path}")

    @viewer.bind_key("Control-s")
    def save_annotations(viewer):
        data = labels_layer.data.astype(np.uint16)
        tifffile.imwrite(str(save_path), data)
        n_cells = int(data.max())
        print(f"\nSaved {n_cells} cells → {save_path}")

    print("\nTips:")
    print("  P = paint mode  |  E = erase mode  |  F = fill mode")
    print("  [/] = brush size  |  L = toggle label visibility")
    print("  To add new cell: set label ID to new number, paint")
    print("  To delete cell:  erase or set label=0 and fill")

    napari.run()

    # Auto-save on close if not saved yet
    if not save_path.exists():
        data = labels_layer.data.astype(np.uint16)
        tifffile.imwrite(str(save_path), data)
        print(f"Auto-saved on exit → {save_path}")
    else:
        print(f"Annotation complete: {save_path}")


if __name__ == "__main__":
    main()
