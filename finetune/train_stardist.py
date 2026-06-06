"""
StarDist fine-tuning script for brain cell segmentation.

Fine-tunes 2D_versatile_fluo on annotated CosMx brain DAPI images.

Usage (from CosMx_2026/):
    /home/fujiyamaakihisa/miniconda3/envs/cosmx_annotation/bin/python \
        finetune/train_stardist.py

Options:
    --gt_dir    Path to ground truth masks (default: annotation_tools/ground_truth/)
    --img_dir   Path to DAPI enhanced images (default: outputs/pilot_4fov/enhanced/)
    --out_dir   Where to save fine-tuned model (default: finetune/models/)
    --name      Model name (default: stardist_brain_v1)
    --epochs    Training epochs (default: 100)
    --patches   Patches per image per epoch (default: 32)
    --val_frac  Fraction of data for validation (default: 0.2)

Requirements: ≥ 1 annotated image in gt_dir (produced by launch_annotator.py)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from stardist import fill_label_holes
from stardist.models import StarDist2D, Config2D

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_GT_DIR  = PROJECT_ROOT / "annotation_tools" / "ground_truth"
DEFAULT_IMG_DIR = PROJECT_ROOT / "outputs" / "pilot_4fov" / "enhanced"
DEFAULT_OUT_DIR = Path(__file__).parent / "models"

# Preprocessing must match what the pipeline uses at inference time
GAUSSIAN_SIGMA = 5.0   # matches stardist_segmenter.py default


def preprocess_dapi(img: np.ndarray) -> np.ndarray:
    """Blur + percentile normalize to [0,1] float32 — mirrors StarDistSegmenter."""
    blurred = gaussian_filter(img.astype(np.float32), sigma=GAUSSIAN_SIGMA)
    lo, hi = np.percentile(blurred, 1), np.percentile(blurred, 99)
    if hi == lo:
        return np.zeros_like(blurred)
    return np.clip((blurred - lo) / (hi - lo), 0, 1).astype(np.float32)


def load_pairs(gt_dir: Path, img_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load matching (DAPI, mask) pairs from gt_dir and img_dir."""
    gt_paths = sorted(gt_dir.glob("*_gt_mask*.tif"))
    if not gt_paths:
        sys.exit(
            f"No ground truth masks found in {gt_dir}\n"
            f"Run annotation_tools/launch_annotator.py first."
        )

    images, masks = [], []
    for gt_path in gt_paths:
        fov = _parse_fov(gt_path.name)
        crop = _parse_crop_tag(gt_path.name)

        # find matching enhanced DAPI
        dapi_matches = list(img_dir.glob(f"{fov}_*_tophat_enhanced.tif"))
        if not dapi_matches:
            print(f"[SKIP] No DAPI for {gt_path.name}")
            continue
        dapi_path = dapi_matches[0]

        dapi_raw = tifffile.imread(str(dapi_path)).astype(np.float32)
        mask_raw = tifffile.imread(str(gt_path)).astype(np.int32)

        if crop:
            y0, x0, h, w = crop
            dapi_raw = dapi_raw[y0:y0+h, x0:x0+w]

        if dapi_raw.shape != mask_raw.shape:
            print(f"[SKIP] Shape mismatch: DAPI {dapi_raw.shape} vs mask {mask_raw.shape}")
            continue

        dapi_proc = preprocess_dapi(dapi_raw)
        mask_clean = fill_label_holes(mask_raw)

        images.append(dapi_proc)
        masks.append(mask_clean)
        print(f"Loaded: {gt_path.name} — {mask_clean.max()} cells, shape {dapi_proc.shape}")

    if not images:
        sys.exit("No valid image/mask pairs found. Check paths and shapes.")

    return images, masks


def _parse_fov(filename: str) -> str:
    m = re.match(r"(FOV\d{5})", filename)
    return m.group(1) if m else ""


def _parse_crop_tag(filename: str) -> tuple | None:
    m = re.search(r"_crop(\d+)-(\d+)-(\d+)-(\d+)", filename)
    if m:
        return tuple(int(x) for x in m.groups())
    return None


def split_train_val(images, masks, val_frac):
    n = len(images)
    n_val = max(1, int(n * val_frac))
    # use last n_val samples as validation
    return images[:-n_val], masks[:-n_val], images[-n_val:], masks[-n_val:]


def build_config(n_rays: int = 32, patch_size: int = 256) -> Config2D:
    return Config2D(
        n_rays=n_rays,
        grid=(2, 2),           # subsampling grid — 2x gives best speed/accuracy
        use_gpu=False,         # set True if GPU available (TF will auto-detect)
        n_channel_in=1,
        train_patch_size=(patch_size, patch_size),
        train_epochs=100,
        train_steps_per_epoch=100,
    )


def main():
    parser = argparse.ArgumentParser(description="StarDist fine-tuning for brain CosMx")
    parser.add_argument("--gt_dir",   type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--img_dir",  type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--out_dir",  type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name",     default="stardist_brain_v1")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--patches",  type=int, default=32)
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading annotated data ===")
    images, masks = load_pairs(args.gt_dir, args.img_dir)
    print(f"Total pairs: {len(images)}")

    if len(images) == 1:
        # with only 1 image, use 80/20 crop split instead of image split
        print("Only 1 image — using spatial 80/20 split for validation")
        img, msk = images[0], masks[0]
        h = img.shape[0]
        split = int(h * 0.8)
        X_train = [img[:split]]
        Y_train = [msk[:split]]
        X_val   = [img[split:]]
        Y_val   = [msk[split:]]
    else:
        X_train, Y_train, X_val, Y_val = split_train_val(images, masks, args.val_frac)

    print(f"Train: {len(X_train)} images | Val: {len(X_val)} images")

    # ── Load pretrained model and fine-tune ──────────────────────────────────
    print("\n=== Loading pretrained 2D_versatile_fluo ===")
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    # Override training config
    cfg = model.config
    cfg.train_epochs = args.epochs
    cfg.train_steps_per_epoch = args.patches
    cfg.train_patch_size = (256, 256)

    # Update basedir and name so weights are saved to our out_dir
    model.basedir = str(args.out_dir)
    model.name = args.name

    print(f"Fine-tuning for {args.epochs} epochs × {args.patches} steps")
    print(f"Save path: {args.out_dir / args.name}")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n=== Training ===")
    model.train(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        augmenter=None,   # StarDist applies internal augmentation
    )

    # ── Optimize thresholds on validation data ────────────────────────────────
    print("\n=== Optimizing thresholds on validation data ===")
    model.optimize_thresholds(X_val, Y_val)

    print(f"\n=== Done ===")
    print(f"Fine-tuned model saved: {args.out_dir / args.name}")
    print(f"prob_thresh: {model.thresholds.prob:.3f}")
    print(f"nms_thresh:  {model.thresholds.nms:.3f}")
    print("\nTo use in pipeline, run:")
    print(f"  python cell_segmentation_pipeline/scripts/run_segmentation.py \\")
    print(f"    --model stardist_finetuned \\")
    print(f"    --model_path {args.out_dir / args.name}")


if __name__ == "__main__":
    main()
