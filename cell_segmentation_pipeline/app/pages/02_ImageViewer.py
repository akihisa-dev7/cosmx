"""Image viewer page — zoom/pan inspection of DAPI, seg masks, and overlays."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # cell_segmentation_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

from utils.image_loader import (
    PROJECT_ROOT,
    load_raw_dapi,
    norm_uint8,
    downsample_img,
    downsample_mask,
)
from utils.overlay_utils import colorize_mask, compose_overlay

st.set_page_config(page_title="Image Viewer", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────

PIXEL_SIZE_UM = 0.12028   # µm per full-res pixel (CosMx calibration)
FONT_PATH     = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DISPLAY_PX    = 860        # canvas display size

_ENHANCED_ROOTS = [
    PROJECT_ROOT / "outputs" / "pilot_4fov",
    PROJECT_ROOT / "outputs" / "enhanced_images",
]

MODELS = {
    "cellpose_cpsam":  "Cellpose cpsam",
    "cellpose_nuclei": "Cellpose nuclei",
    "stardist":        "StarDist",
    "instanseg":       "InstanSeg",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _nice_bar_um(target: float) -> float:
    for v in [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500]:
        if v >= target * 0.5:
            return float(v)
    return 500.0


def _draw_scale_bar(img: np.ndarray, crop_h_px: int, bar_um: float) -> np.ndarray:
    from PIL import Image as _PI, ImageDraw as _PD, ImageFont as _PF
    h, w   = img.shape[:2]
    scale  = h / crop_h_px
    bar_px = max(5, int(round(bar_um / PIXEL_SIZE_UM * scale)))
    rgb    = np.stack([img, img, img], axis=-1) if img.ndim == 2 else img.copy()
    pil    = _PI.fromarray(rgb.astype(np.uint8))
    draw   = _PD.Draw(pil)
    mg, th = 10, 4
    x2, yb = w - mg, h - mg - th
    x1     = x2 - bar_px
    draw.rectangle([x1 - 1, yb - 1, x2 + 1, yb + th + 1], fill=(0, 0, 0))
    draw.rectangle([x1, yb, x2, yb + th],                   fill=(255, 255, 255))
    lbl = f"{int(bar_um)} µm"
    try:
        font = _PF.truetype(FONT_PATH, 12)
        bb   = draw.textbbox((0, 0), lbl, font=font)
        tw, fh = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        font = None; tw, fh = len(lbl) * 7, 11
    tx = (x1 + x2) // 2 - tw // 2
    ty = yb - fh - 4
    draw.text((tx + 1, ty + 1), lbl, fill=(0, 0, 0),      font=font)
    draw.text((tx,     ty),     lbl, fill=(255, 255, 255), font=font)
    return np.array(pil)


@st.cache_data(show_spinner=False)
def _load_enhanced_dapi(fov_id: str) -> np.ndarray | None:
    import tifffile
    for root in _ENHANCED_ROOTS:
        d = root / "enhanced"
        if d.exists():
            cands = sorted(d.glob(f"{fov_id}*enhanced*.tif"))
            if cands:
                img = tifffile.imread(str(cands[0]))
                return img[0] if img.ndim == 3 else img
        d2 = root / "tif"
        if d2.exists():
            cands = sorted(d2.glob(f"{fov_id}*enhanced*.tif"))
            if cands:
                img = tifffile.imread(str(cands[0]))
                return img[0] if img.ndim == 3 else img
    return load_raw_dapi(fov_id)


@st.cache_data(show_spinner=False)
def _load_mask(fov_id: str, model_key: str, output_dir: str, baysor: bool) -> np.ndarray | None:
    import tifffile
    root = Path(output_dir) / "masks"
    if not root.exists():
        return None
    for p in root.glob(f"{fov_id}*nuclear_mask.tif"):
        has_model  = (model_key.replace("cellpose_", "") in p.name or model_key in p.name)
        has_baysor = "baysor" in p.name
        if has_model and has_baysor == baysor:
            return tifffile.imread(str(p)).astype(np.int32)
    return None


# ── Check session target ──────────────────────────────────────────────────────

target = st.session_state.get("detail_target")

if target is None:
    st.warning("表示対象が設定されていません。Segmentation Runner で 🔍 詳細 を押してください。")
    if st.button("← Segmentation Runner に戻る"):
        st.switch_page("pages/00_Segmentation_Runner.py")
    st.stop()

fov_id     = target["fov"]
model_key  = target["model"]
output_dir = target["output_dir"]
init_type  = target.get("type", "dapi")   # "dapi" | "mask" | "baysor_mask"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔍 Image Viewer")
    st.caption(f"FOV: **{fov_id}**")
    st.caption(f"Model: **{MODELS.get(model_key, model_key)}**")

    if st.button("← Runner に戻る", use_container_width=True):
        st.switch_page("pages/00_Segmentation_Runner.py")

    if st.button("✏️ Annotate に切り替え", use_container_width=True, type="primary"):
        st.session_state["annotate_target"] = {
            "fov": fov_id, "model": model_key, "output_dir": output_dir,
            "use_baysor": init_type == "baysor_mask",
        }
        st.switch_page("pages/01_Annotate.py")

    st.divider()
    st.subheader("表示する画像")
    view_options = {
        "dapi":        "Enhanced DAPI",
        "mask":        "Seg Mask (DAPI)",
        "baysor_mask": "Baysor Mask",
        "overlay":     "Overlay",
    }
    # Only show baysor_mask if it exists
    _baysor_exists = _load_mask(fov_id, model_key, output_dir, baysor=True) is not None
    shown_options  = {k: v for k, v in view_options.items()
                      if k != "baysor_mask" or _baysor_exists}

    _opt_keys = list(shown_options.keys())
    _default_idx = _opt_keys.index(init_type) if init_type in _opt_keys else 0
    view_type = st.radio(
        "画像タイプ",
        options=_opt_keys,
        format_func=lambda k: shown_options[k],
        index=_default_idx,
        key="iv_view_type",
    )

    st.divider()
    st.subheader("ズーム / 位置")
    zoom = st.select_slider("ズーム", options=[1, 2, 4, 8], value=2, key="iv_zoom")

# ── Load full-res source images ───────────────────────────────────────────────

with st.spinner("画像を読み込み中…"):
    dapi_full = _load_enhanced_dapi(fov_id)
    mask_full = _load_mask(fov_id, model_key, output_dir, baysor=False)
    bays_full = _load_mask(fov_id, model_key, output_dir, baysor=True) if _baysor_exists else None

if dapi_full is None and mask_full is None:
    st.error("画像もマスクも見つかりませんでした。")
    st.stop()

# Determine reference shape
_ref = dapi_full if dapi_full is not None else mask_full
_H, _W = _ref.shape[:2]

# ── Zoom / position controls (needs _H, _W) ──────────────────────────────────

crop_h = max(64, _H // zoom)
crop_w = max(64, _W // zoom)

with st.sidebar:
    _max_cx = max(1, _W - crop_w)
    _max_cy = max(1, _H - crop_h)
    if _max_cx > 0:
        cx = st.slider("X位置 (px)", 0, _max_cx, _max_cx // 2, step=32, key="iv_cx")
    else:
        cx = 0
        st.caption("X: 全幅表示中")
    if _max_cy > 0:
        cy = st.slider("Y位置 (px)", 0, _max_cy, _max_cy // 2, step=32, key="iv_cy")
    else:
        cy = 0
        st.caption("Y: 全高表示中")

    um_per_disp = PIXEL_SIZE_UM * crop_h / DISPLAY_PX
    bar_um = _nice_bar_um(um_per_disp * DISPLAY_PX * 0.15)
    st.caption(f"表示範囲: {crop_w * PIXEL_SIZE_UM:.0f} × {crop_h * PIXEL_SIZE_UM:.0f} µm")
    st.caption(f"スケールバー: **{int(bar_um)} µm**")
    st.caption(f"1表示px ≈ {um_per_disp:.2f} µm")

# ── Build display image ───────────────────────────────────────────────────────

def _crop_ds(arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
    c = arr[cy: cy + crop_h, cx: cx + crop_w]
    if is_mask:
        return downsample_mask(c, DISPLAY_PX)
    return downsample_img(c, DISPLAY_PX)


if view_type == "dapi":
    if dapi_full is None:
        st.error("DAPI画像が見つかりません")
        st.stop()
    base = norm_uint8(_crop_ds(dapi_full))
    disp = _draw_scale_bar(base, crop_h, bar_um)

elif view_type == "mask":
    if mask_full is None:
        st.error("Seg Maskが見つかりません")
        st.stop()
    mask_crop = _crop_ds(mask_full, is_mask=True)
    disp = colorize_mask(mask_crop)
    n_cells = int(np.unique(mask_crop[mask_crop > 0]).shape[0])

elif view_type == "baysor_mask":
    if bays_full is None:
        st.error("Baysor Maskが見つかりません")
        st.stop()
    mask_crop = _crop_ds(bays_full, is_mask=True)
    disp = colorize_mask(mask_crop)
    n_cells = int(np.unique(mask_crop[mask_crop > 0]).shape[0])

else:  # overlay
    if dapi_full is None:
        st.error("DAPI画像が見つかりません")
        st.stop()
    dapi_ds  = norm_uint8(_crop_ds(dapi_full))
    ov_src   = bays_full if bays_full is not None else mask_full
    if ov_src is not None:
        mask_ds = _crop_ds(ov_src, is_mask=True)
        base    = compose_overlay(dapi_ds, mask_ds)
    else:
        base    = np.stack([dapi_ds, dapi_ds, dapi_ds], axis=-1)
    disp = _draw_scale_bar(base, crop_h, bar_um)

# ── Main display ──────────────────────────────────────────────────────────────

_type_label = shown_options.get(view_type, view_type)
st.title(f"🔍 {MODELS.get(model_key, model_key)} — {fov_id}")

img_col, meta_col = st.columns([4, 1])

with img_col:
    st.image(disp, use_container_width=True, clamp=True)
    st.caption(
        f"**{_type_label}**  |  "
        f"表示範囲: {crop_w * PIXEL_SIZE_UM:.0f} × {crop_h * PIXEL_SIZE_UM:.0f} µm  |  "
        f"起点: ({cx} px, {cy} px)  |  "
        f"ズーム: {zoom}×"
    )

with meta_col:
    st.subheader("情報")
    st.metric("FOV", fov_id)
    st.metric("Model", MODELS.get(model_key, model_key))
    st.metric("表示タイプ", _type_label)
    st.metric("ズーム", f"{zoom}×")
    st.metric("表示幅", f"{crop_w * PIXEL_SIZE_UM:.0f} µm")
    st.metric("表示高さ", f"{crop_h * PIXEL_SIZE_UM:.0f} µm")

    if view_type in ("mask", "baysor_mask"):
        st.metric("表示範囲内細胞数", n_cells)

    st.divider()

    # Quick-switch image type buttons
    st.subheader("切り替え")
    for opt_key, opt_label in shown_options.items():
        if opt_key == view_type:
            st.markdown(f"▶ **{opt_label}**")
        else:
            if st.button(opt_label, key=f"sw_{opt_key}", use_container_width=True):
                st.session_state["detail_target"] = {
                    **target,
                    "type": opt_key,
                }
                st.rerun()
