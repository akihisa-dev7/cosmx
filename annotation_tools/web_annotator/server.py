"""
Web annotation server for CosMx segmentation mask editing.

Run from CosMx_2026/:
    /home/fujiyamaakihisa/miniconda3/envs/cosmx_annotation/bin/pip install -q fastapi uvicorn pillow
    /home/fujiyamaakihisa/miniconda3/envs/cosmx_annotation/bin/python \
        annotation_tools/web_annotator/server.py

Then open http://100.88.113.119:8000 in your browser.
"""
from __future__ import annotations

import base64
import colorsys
import io
import os
from pathlib import Path

import numpy as np
import tifffile
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

app = FastAPI()

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configurable via environment variables (set by 01_Annotate.py when launching)
ENHANCED_DIR = Path(os.environ.get(
    "ANNOTATOR_ENHANCED_DIR",
    str(PROJECT_ROOT / "outputs" / "pilot_4fov" / "enhanced"),
))
MASKS_DIR = Path(os.environ.get(
    "ANNOTATOR_MASKS_DIR",
    str(PROJECT_ROOT / "outputs" / "pilot_4fov" / "masks"),
))
GT_DIR = Path(os.environ.get(
    "ANNOTATOR_GT_DIR",
    str(Path(__file__).parent.parent / "ground_truth"),
))

FOV_SUFFIX = {
    "FOV00001": "AD_F",
    "FOV00007": "AD_H",
    "FOV00037": "Control_F",
    "FOV00043": "Control_H",
}
PIXEL_SIZE_UM = 0.12028  # CosMx calibration µm/pixel

_gt_cache: dict[str, np.ndarray] = {}
_dapi_cache: dict[str, np.ndarray] = {}
_source_cache: dict[str, str] = {}  # fov -> "ground_truth" or model name


def _load_dapi(fov: str) -> np.ndarray:
    if fov not in _dapi_cache:
        # Try enhanced dir with known suffix pattern
        suffix = FOV_SUFFIX.get(fov, "")
        candidates = (
            list(ENHANCED_DIR.glob(f"{fov}*enhanced*.tif"))
            + ([] if not suffix else [ENHANCED_DIR / f"{fov}_{suffix}_tophat_enhanced.tif"])
        )
        candidates = [p for p in candidates if p.exists()]
        if not candidates:
            raise FileNotFoundError(f"No enhanced DAPI found for {fov} in {ENHANCED_DIR}")
        img = tifffile.imread(str(candidates[0])).astype(np.float32)
        _dapi_cache[fov] = img[0] if img.ndim == 3 else img
    return _dapi_cache[fov]


def _load_gt(fov: str) -> np.ndarray:
    if fov not in _gt_cache:
        # 1. Check for previously saved ground truth
        gt_path = GT_DIR / f"{fov}_gt_mask.tif"
        if gt_path.exists():
            mask = tifffile.imread(str(gt_path)).astype(np.int32)
            _source_cache[fov] = "ground_truth (編集済み)"
        else:
            # 2. Discover any nuclear mask in MASKS_DIR for this FOV
            hits = [
                p for p in MASKS_DIR.glob(f"{fov}*nuclear_mask.tif")
                if "baysor" not in p.name
            ]
            if not hits:
                raise FileNotFoundError(f"No mask found for {fov} in {MASKS_DIR}")
            hits.sort()
            mask = tifffile.imread(str(hits[0])).astype(np.int32)
            _source_cache[fov] = hits[0].stem
        _gt_cache[fov] = mask
    return _gt_cache[fov]


def _png_b64(arr: np.ndarray, mode: str) -> str:
    img = Image.fromarray(arr, mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


def _discover_fovs() -> list[str]:
    """Discover FOV IDs that have at least one mask in MASKS_DIR."""
    seen: set[str] = set()
    if MASKS_DIR.exists():
        for p in MASKS_DIR.glob("FOV*nuclear_mask.tif"):
            fov = p.name.split("_")[0]
            if fov.startswith("FOV"):
                seen.add(fov)
    # Also include hardcoded pilot FOVs if DAPI available
    for fov in FOV_SUFFIX:
        if any(ENHANCED_DIR.glob(f"{fov}*enhanced*.tif")):
            seen.add(fov)
    return sorted(seen)


@app.get("/api/info")
def get_info():
    fovs = {}
    for fov in _discover_fovs():
        try:
            dapi = _load_dapi(fov)
            gt = _load_gt(fov)
            fovs[fov] = {
                "h": int(dapi.shape[0]),
                "w": int(dapi.shape[1]),
                "n_cells": int(gt.max()),
                "source": _source_cache.get(fov, "unknown"),
            }
        except FileNotFoundError:
            pass
    return {"fovs": fovs, "pixel_size_um": PIXEL_SIZE_UM}


@app.get("/api/tile")
def get_tile(fov: str, y: int = 0, x: int = 0, size: int = 1024):
    fov = fov.upper()
    if fov not in _discover_fovs():
        raise HTTPException(404, f"Unknown FOV: {fov}")

    dapi_full = _load_dapi(fov)
    gt_full = _load_gt(fov)
    H, W = dapi_full.shape

    y = max(0, min(y, H - size))
    x = max(0, min(x, W - size))

    dapi_crop = dapi_full[y:y+size, x:x+size]
    gt_crop = gt_full[y:y+size, x:x+size]

    if dapi_crop.shape != (size, size):
        ph, pw = dapi_crop.shape
        pad_d = np.zeros((size, size), dtype=np.float32)
        pad_g = np.zeros((size, size), dtype=np.int32)
        pad_d[:ph, :pw] = dapi_crop
        pad_g[:ph, :pw] = gt_crop
        dapi_crop, gt_crop = pad_d, pad_g

    lo, hi = float(dapi_crop.min()), float(dapi_crop.max())
    if hi > lo:
        dapi_u8 = ((dapi_crop - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        dapi_u8 = np.zeros((size, size), dtype=np.uint8)

    dapi_b64 = _png_b64(dapi_u8, "L")
    labels_b64 = base64.b64encode(gt_crop.astype(np.uint16).tobytes()).decode()

    return {
        "fov": fov, "y": y, "x": x, "size": size,
        "image_h": int(H), "image_w": int(W),
        "dapi_b64": dapi_b64,
        "labels_b64": labels_b64,
        "max_label": int(gt_full.max()),
        "source": _source_cache.get(fov, "unknown"),
    }


class SaveRequest(BaseModel):
    fov: str
    y: int
    x: int
    size: int
    labels_b64: str


@app.post("/api/save")
def save_tile(req: SaveRequest):
    fov = req.fov.upper()
    if fov not in _discover_fovs():
        raise HTTPException(404, f"Unknown FOV: {fov}")

    raw = base64.b64decode(req.labels_b64)
    patch = np.frombuffer(raw, dtype=np.uint16).reshape(req.size, req.size).astype(np.int32)

    gt_full = _load_gt(fov)
    H, W = gt_full.shape
    y = max(0, min(req.y, H - req.size))
    x = max(0, min(req.x, W - req.size))
    h = min(req.size, H - y)
    w = min(req.size, W - x)

    gt_full[y:y+h, x:x+w] = patch[:h, :w]
    _gt_cache[fov] = gt_full
    _source_cache[fov] = "ground_truth (編集済み)"

    GT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = GT_DIR / f"{fov}_gt_mask.tif"
    tifffile.imwrite(str(save_path), gt_full.astype(np.uint16))

    return {
        "status": "ok",
        "saved_to": str(save_path),
        "total_cells": int(gt_full.max()),
    }


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",         type=int, default=8000)
    parser.add_argument("--host",         default="0.0.0.0")
    parser.add_argument("--masks-dir",    default=None, help="Override MASKS_DIR")
    parser.add_argument("--enhanced-dir", default=None, help="Override ENHANCED_DIR")
    parser.add_argument("--gt-dir",       default=None, help="Override GT_DIR")
    args = parser.parse_args()

    # Allow CLI overrides (take precedence over env vars)
    if args.masks_dir:
        MASKS_DIR    = Path(args.masks_dir)
    if args.enhanced_dir:
        ENHANCED_DIR = Path(args.enhanced_dir)
    if args.gt_dir:
        GT_DIR       = Path(args.gt_dir)

    print(f"MASKS_DIR:    {MASKS_DIR}")
    print(f"ENHANCED_DIR: {ENHANCED_DIR}")
    print(f"GT_DIR:       {GT_DIR}")
    print(f"Listening on  http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
