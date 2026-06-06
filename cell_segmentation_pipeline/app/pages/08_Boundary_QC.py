"""Boundary QC — CosMx native label vs custom segmentation comparison.

Four comparison modes:
  A  Current Otsu-based QC
  B  CosMx native label QC
  C  Side-by-side (Otsu | CosMx label)
  D  Overlay (all layers together)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # CosMx_2026/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # app/

import glob
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.image_loader import (
    load_raw_dapi, load_enhanced, load_mask,
    norm_uint8, downsample_img, downsample_mask,
    DEFAULT_OUTPUT_ROOTS,
)
from utils.cell_cropper import find_cell_at_xy, get_cell_crop
from utils.cosmx_label_loader import (
    load_compartment, load_cell_labels_c, load_cosmx_label_data,
    run_comparison_cached, build_rgb_overlay,
    list_available_fovs, PER_FOV_ROOT, LAYER_COLORS,
)

st.set_page_config(page_title="Boundary QC", layout="wide")
st.title("CosMx Correspondence QC  （評価軸 2/3）")
st.caption("CosMx native label と custom segmentation の空間的対応関係を評価します。")

st.info(
    "**評価方針** — CosMx native labels（cell-boundary marker 寄り）と custom cpsam（DAPI nucleus 寄り）は"
    "セグメンテーション定義が異なるため、Population IoU・Dice・Boundary F1 は意味のある比較ができません。"
    "このページでは **Registration status / Centroid shift / Matched-pair IoU / Cell count ratio** のみを評価軸として使用します。"
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PER_FOV_ROOT_STR = str(PER_FOV_ROOT)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    seg_root = st.selectbox(
        "Segmentation output root",
        DEFAULT_OUTPUT_ROOTS,
        format_func=lambda p: Path(p).name,
        key="bqc_seg_root",
    )

    fov_names = list_available_fovs(PER_FOV_ROOT_STR)
    if not fov_names:
        fov_names = ["FOV00001", "FOV00007", "FOV00037", "FOV00043"]
    fov_name = st.selectbox("FOV", fov_names, key="bqc_fov")

    model_options = ["cpsam", "instanseg", "stardist"]
    model_name = st.selectbox("Custom model", model_options, key="bqc_model")

    mode = st.radio(
        "View mode",
        ["A: Otsu QC", "B: CosMx Label QC", "C: Side-by-side", "D: Overlay"],
        key="bqc_mode",
    )

    st.divider()
    st.markdown("**Display**")
    alpha     = st.slider("Overlay opacity", 0.0, 1.0, 0.50, 0.05, key="bqc_alpha")
    boundary_width = st.slider("Boundary dilation (display only)", 0, 5, 1, 1, key="bqc_bw")
    max_dim   = st.select_slider("Resolution", [512, 768, 1024, 1536], value=1024, key="bqc_dim")

    st.divider()
    st.markdown("**Analysis parameters**")
    match_dist = st.slider("Centroid match distance (px)", 5, 100, 30, 5, key="bqc_mdist")
    bnd_tol    = st.slider("Boundary F1 tolerance (px)", 1, 10, 3, 1, key="bqc_btol")

    with st.expander("Layer toggles (Overlay mode)"):
        show_dapi    = st.checkbox("DAPI background", True,  key="bqc_l_dapi")
        show_cnuc    = st.checkbox("CosMx nucleus (blue)",   True,  key="bqc_l_cnuc")
        show_ccell   = st.checkbox("CosMx cell (green)",     True,  key="bqc_l_ccell")
        show_cmem    = st.checkbox("CosMx membrane (purple)",False, key="bqc_l_cmem")
        show_custom  = st.checkbox("Custom mask (red)",      True,  key="bqc_l_cust")
        show_overlap = st.checkbox("Overlap (yellow)",       True,  key="bqc_l_ov")
        show_fn      = st.checkbox("Missed by custom (orange)", False, key="bqc_l_fn")
        show_fp      = st.checkbox("Extra in custom (magenta)", False, key="bqc_l_fp")
        show_otsu    = st.checkbox("Otsu mask (cyan)",       False, key="bqc_l_otsu")
        boundary_only = st.checkbox("Boundary lines only",  True,  key="bqc_boundary")


# ── Load data ──────────────────────────────────────────────────────────────────
raw      = load_raw_dapi(fov_name, 0)
enhanced = load_enhanced(fov_name, seg_root)
custom_mask_img = load_mask(fov_name, model_name, seg_root)

comp_arr    = load_compartment(fov_name, PER_FOV_ROOT_STR)
cell_lbl_arr = load_cell_labels_c(fov_name, PER_FOV_ROOT_STR)

ref = enhanced if enhanced is not None else raw

# ── Sanity check panel ─────────────────────────────────────────────────────────
with st.expander("Coordinate & shape sanity check", expanded=False):
    checks = []
    shapes = {
        "Raw DAPI":           raw.shape if raw is not None else None,
        "Enhanced DAPI":      enhanced.shape if enhanced is not None else None,
        "Custom mask":        custom_mask_img.shape if custom_mask_img is not None else None,
        "CompartmentLabels":  comp_arr.shape if comp_arr is not None else None,
        "CellLabels":         cell_lbl_arr.shape if cell_lbl_arr is not None else None,
    }
    expected = (4256, 4256)
    for name, sh in shapes.items():
        ok  = sh == expected if sh is not None else False
        st.markdown(
            f"{'✅' if ok else '❌' if sh is not None else '⚠️'}  **{name}**: "
            f"`{sh}`" + (" — shape mismatch!" if sh and sh != expected else "")
        )

    # Coordinate system note
    cosmx = load_cosmx_label_data(fov_name, PER_FOV_ROOT_STR)
    if cosmx is not None and custom_mask_img is not None:
        sane = cosmx.sanity_check(custom_mask_img)
        if sane["issues"]:
            for issue in sane["issues"]:
                st.warning(f"⚠️ {issue}")
        else:
            st.success("All shapes consistent ✓")

    st.info(
        "**Note:** Custom (cpsam) and CosMx native segmentations are "
        "independent algorithms — direct pixel IoU is expected to be low (~7-9%). "
        "Cell matching is done by nearest centroid (default 30px)."
    )

    with st.expander("Manual offset correction"):
        ox = st.number_input("X offset (px)", -500, 500, 0, key="bqc_ox")
        oy = st.number_input("Y offset (px)", -500, 500, 0, key="bqc_oy")
        flip_x = st.checkbox("Flip X", False, key="bqc_fx")
        flip_y = st.checkbox("Flip Y", False, key="bqc_fy")
        swap_xy = st.checkbox("Swap X/Y", False, key="bqc_sxy")
        if ox != 0 or oy != 0 or flip_x or flip_y or swap_xy:
            st.caption(
                "Offset/flip applied to CosMx labels for comparison. "
                "Use this to diagnose coordinate misalignment."
            )


def _apply_offset(arr: np.ndarray, ox: int, oy: int,
                  flip_x: bool, flip_y: bool, swap_xy: bool) -> np.ndarray:
    if swap_xy:
        arr = arr.T
    if flip_x:
        arr = np.fliplr(arr)
    if flip_y:
        arr = np.flipud(arr)
    if ox != 0 or oy != 0:
        arr = np.roll(arr, (oy, ox), axis=(0, 1))
    return arr


ox_val  = st.session_state.get("bqc_ox", 0)
oy_val  = st.session_state.get("bqc_oy", 0)
fx_val  = st.session_state.get("bqc_fx", False)
fy_val  = st.session_state.get("bqc_fy", False)
sxy_val = st.session_state.get("bqc_sxy", False)


def get_comp_adjusted():
    if comp_arr is None:
        return None
    return _apply_offset(comp_arr.copy(), ox_val, oy_val, fx_val, fy_val, sxy_val)


def get_cell_lbl_adjusted():
    if cell_lbl_arr is None:
        return None
    return _apply_offset(cell_lbl_arr.copy(), ox_val, oy_val, fx_val, fy_val, sxy_val)


# ── Compute label comparison (cached) ─────────────────────────────────────────
label_result = None
if comp_arr is not None and custom_mask_img is not None:
    label_result = run_comparison_cached(
        fov_name, model_name, PER_FOV_ROOT_STR, seg_root,
        float(match_dist), int(bnd_tol),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helper: build downsampled images for Plotly
# ══════════════════════════════════════════════════════════════════════════════

def _plotly_img(arr_2d_or_rgb: np.ndarray, title: str = "") -> go.Figure:
    fig = px.imshow(arr_2d_or_rgb, aspect="equal",
                    color_continuous_scale="gray" if arr_2d_or_rgb.ndim == 2 else None)
    fig.update_layout(
        margin=dict(l=0, r=0, t=28 if title else 0, b=0),
        title=title, dragmode="pan", coloraxis_showscale=False,
    )
    return fig


def _ds_gray(img: np.ndarray) -> np.ndarray:
    return downsample_img(norm_uint8(img), max_dim) if img is not None else None


def _ds_mask(mask: np.ndarray) -> np.ndarray:
    return downsample_mask(mask, max_dim) if mask is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# MODE A — Otsu-based QC
# ══════════════════════════════════════════════════════════════════════════════
if "A:" in mode:
    st.subheader("Mode A — Otsu-based QC")

    if custom_mask_img is None:
        st.warning(f"No custom mask found for {fov_name} / {model_name}.")
        st.stop()

    # Run Otsu QC
    with st.spinner("Running Otsu QC…"):
        import sys as _sys
        _sys.path.insert(0, str(PROJECT_ROOT))
        from cell_segmentation_pipeline.src.dapi_qc import (
            run_fov_dapi_qc, detect_dapi_objects
        )

        raw_np = raw if raw is not None else np.zeros((4256, 4256), np.uint16)
        enh_np = enhanced if enhanced is not None else raw_np

        otsu_result = run_fov_dapi_qc(
            raw_np, enh_np, custom_mask_img,
            fov_id=fov_name, model_name=model_name,
            dapi_object_min_area=200,
        )
        dapi_lbl = otsu_result.get("dapi_labels", np.zeros_like(custom_mask_img))

    # Metric cards
    cols = st.columns(5)
    metrics = [
        ("Cells",          otsu_result["cell_count"]),
        ("DAPI objects",   otsu_result["dapi_object_count"]),
        ("SNR",            f"{otsu_result['signal_to_background_ratio']:.2f}x"),
        ("DAPI+ fraction", f"{otsu_result['dapi_positive_mask_fraction']*100:.1f}%"),
        ("Mean IoU (matched)", f"{otsu_result['mean_iou']:.3f}"),
    ]
    for col, (label, val) in zip(cols, metrics):
        col.metric(label, val)

    # Overlay figure
    if ref is not None:
        gray_ds   = _ds_gray(ref)
        custom_ds = _ds_mask(custom_mask_img)
        dapi_ds   = _ds_mask(dapi_lbl)

        per_cell = otsu_result.get("per_cell_df", pd.DataFrame())
        good_lbl = set(
            per_cell.loc[per_cell.get("auto_qc_label", pd.Series()) == "good", "label"]
            .astype(int).tolist()
        ) if "auto_qc_label" in per_cell.columns else set()
        low_lbl = set(
            per_cell.loc[per_cell.get("auto_qc_label", pd.Series()) == "low_dapi", "label"]
            .astype(int).tolist()
        ) if "auto_qc_label" in per_cell.columns else set()

        # Build QC-coloured mask overlay
        rgb_mask = np.zeros((*gray_ds.shape, 3), dtype=np.uint8)
        for lbl in np.unique(custom_ds):
            if lbl == 0: continue
            pxl = custom_ds == lbl
            if lbl in good_lbl:    rgb_mask[pxl] = [0, 200, 0]
            elif lbl in low_lbl:   rgb_mask[pxl] = [255, 215, 0]
            else:                  rgb_mask[pxl] = [69, 117, 180]

        base  = np.stack([gray_ds]*3, axis=-1).astype(np.float32)
        fg    = rgb_mask.any(axis=-1)
        blend = base.copy()
        blend[fg] = (1-alpha)*base[fg] + alpha*rgb_mask[fg].astype(np.float32)
        blend = np.clip(blend,0,255).astype(np.uint8)

        col_img, col_hist = st.columns([3, 2])
        with col_img:
            st.plotly_chart(_plotly_img(blend, f"Otsu QC — {fov_name}"),
                            use_container_width=True, key="bqc_a_img")
        with col_hist:
            if "mean_intensity" in per_cell.columns:
                fig_h = px.histogram(per_cell, x="mean_intensity", nbins=40,
                                     color="auto_qc_label" if "auto_qc_label" in per_cell.columns else None,
                                     title="DAPI intensity distribution")
                fig_h.update_layout(height=300, margin=dict(t=40,b=20,l=20,r=20))
                st.plotly_chart(fig_h, use_container_width=True, key="bqc_a_hist")

    # Legend
    st.markdown(
        "🟢 Matched (DAPI+) &nbsp; 🟡 Low-DAPI signal &nbsp; 🔵 Extra mask (no DAPI match)"
    )

    # Per-cell table
    with st.expander("Per-cell data"):
        pc = otsu_result.get("per_cell_df", pd.DataFrame())
        st.dataframe(pc.head(200), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE B — CosMx label QC
# ══════════════════════════════════════════════════════════════════════════════
elif "B:" in mode:
    st.subheader("Mode B — CosMx Native Label QC")

    if comp_arr is None:
        st.error(f"CompartmentLabels not found for {fov_name} at {PER_FOV_ROOT_STR}")
        st.stop()
    if custom_mask_img is None:
        st.warning(f"No custom mask for {fov_name}/{model_name}")
        st.stop()

    comp_adj = get_comp_adjusted()
    cell_adj = get_cell_lbl_adjusted()

    # Metric cards
    if label_result is not None:
        import sys as _sys
        _sys.path.insert(0, str(PROJECT_ROOT))
        from cell_segmentation_pipeline.src.composite_qc_score import (
            compute_extended_match_metrics,
            classify_registration,
            detect_segmentation_issues,
        )

        # ── Compute extended metrics ──────────────────────────────────────────
        cosmx_obj = load_cosmx_label_data(fov_name, PER_FOV_ROOT_STR)
        custom_cents = None
        cosmx_cents  = None
        if custom_mask_img is not None:
            from cell_segmentation_pipeline.src.label_based_qc import compute_centroids
            custom_cents = compute_centroids(custom_mask_img)
        if cosmx_obj is not None:
            from cell_segmentation_pipeline.src.label_based_qc import compute_centroids as _cc
            cosmx_cents = _cc(cosmx_obj.cell_labels)

        ext = compute_extended_match_metrics(
            label_result.get("per_cell_df", pd.DataFrame()),
            custom_cents, cosmx_cents, float(match_dist),
        )

        # ── Registration status ───────────────────────────────────────────────
        reg_status, reg_msg = classify_registration(
            ext["mean_dx"], ext["mean_dy"], ext["std_dx"], ext["std_dy"]
        )
        _reg_color = {"OK": "green", "Minor offset": "orange", "Warning": "red", "Unknown": "gray"}
        _reg_badge_style = (
            f"display:inline-block;padding:3px 12px;border-radius:5px;"
            f"background-color:{'#d4edda' if reg_status=='OK' else '#fff3cd' if reg_status=='Minor offset' else '#f8d7da'};"
            f"color:{'#155724' if reg_status=='OK' else '#856404' if reg_status=='Minor offset' else '#721c24'};"
            f"font-weight:bold;"
        )
        st.markdown(
            f'Registration: <span style="{_reg_badge_style}">{reg_status}</span>　'
            f'Mean shift: dx={ext["mean_dx"]:+.1f}px, dy={ext["mean_dy"]:+.1f}px  '
            f'(|shift|={ext["registration_shift_px"]:.1f}px)',
            unsafe_allow_html=True,
        )
        st.caption(reg_msg)

        # ── Segmentation warnings ─────────────────────────────────────────────
        warnings = detect_segmentation_issues(
            int(ext["n_custom"]), int(ext["n_cosmx"]),
            ext["match_rate_custom"], ext["one_to_one_match_rate"],
            ext["multi_match_cosmx_fraction"],
        )
        for w in warnings:
            sev = w["severity"]
            icon = "🔴" if sev == "high" else "🟡"
            st.warning(f"{icon} **{w['type'].replace('_',' ').title()}** — {w['message']}")

        st.divider()

        # ── Primary metrics: correspondence (not Population IoU) ──────────────
        st.markdown("#### CosMx 対応関係指標（主評価）")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cell count ratio",
                  f"{ext['custom_to_cosmx_ratio']:.2f}×",
                  help="custom / CosMx cell count。1.0 が理想。2.0以上は過分割疑い")
        c2.metric("Matched-pair IoU (median)",
                  f"{ext['matched_pair_iou_median']:.3f}",
                  help="centroid が近いペアの IoU 中央値。Population IoU より意味があります")
        c3.metric("One-to-one match rate",
                  f"{ext['one_to_one_match_rate']:.1%}",
                  help="CosMx 1細胞に対して custom 1細胞が1対1対応している割合")
        c4.metric("Median centroid dist",
                  f"{label_result['median_centroid_dist']:.1f}px",
                  help="centroid 距離の中央値。小さいほど空間的に整合している")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Custom cells",      ext["n_custom"])
        c6.metric("CosMx cells",       ext["n_cosmx"])
        c7.metric("Matched pairs",     ext["n_matched"])
        c8.metric("Match rate (custom)", f"{ext['match_rate_custom']:.1%}")

        # Population IoU は評価軸から外し、デバッグ用に最小限だけ残す
        with st.expander("Population IoU（デバッグ用 — 評価に使わないこと）", expanded=False):
            st.caption(
                "定義が異なる2つのセグメンテーションの pixel-level 比較は意味がありません。"
                "Registration 確認には上記の centroid shift / matched-pair IoU を使ってください。"
            )
            st.write(f"Nucleus IoU: {label_result['nucleus_iou']:.4f} / "
                     f"Dice: {label_result['nucleus_dice']:.4f} / "
                     f"Boundary F1: {label_result['boundary_f1_vs_nucleus']:.4f}")

        # ── Interpretation message ────────────────────────────────────────────
        if (
            label_result["nucleus_iou"] < 0.15
            and ext["registration_shift_px"] < 10
            and ext["matched_pair_iou_median"] > 0.15
        ):
            st.success(
                "✅ **解釈** — Population IoU が低いのは、CosMx native label と custom cpsam mask の "
                "セグメンテーション定義が異なるためと考えられます（CosMx: cell boundary 寄り、"
                "cpsam: DAPI nucleus 寄り）。"
                "Registration shift は小さく（{:.1f}px）、近いペアでは IoU が {:.3f} 出ています。"
                "これは registration 失敗や DAPI segmentation 不良を意味しません。".format(
                    ext["registration_shift_px"], ext["matched_pair_iou_median"]
                )
            )

    # Build overlay
    if ref is not None:
        gray_ds     = _ds_gray(ref)
        nucleus_ds  = _ds_mask((comp_adj == 1).astype(np.uint8))
        cell_ds     = _ds_mask((cell_adj > 0).astype(np.uint8)) if cell_adj is not None else None
        membrane_ds = _ds_mask((comp_adj == 3).astype(np.uint8))
        custom_ds   = _ds_mask(custom_mask_img)

        layers = {}
        if show_cnuc and nucleus_ds is not None:
            layers["cosmx_nucleus"] = nucleus_ds > 0
        if show_ccell and cell_ds is not None:
            layers["cosmx_cell"] = cell_ds > 0
        if show_cmem and membrane_ds is not None:
            layers["cosmx_membrane"] = membrane_ds > 0
        if show_custom and custom_ds is not None:
            layers["custom_mask"] = custom_ds > 0
        if show_overlap and nucleus_ds is not None and custom_ds is not None:
            layers["overlap"] = (nucleus_ds > 0) & (custom_ds > 0)

        rgb = build_rgb_overlay(
            gray_ds, layers, alpha=alpha,
            boundary_only=boundary_only, dilate_px=boundary_width,
        )

        col_img, col_leg = st.columns([4, 1])
        with col_img:
            ev = st.plotly_chart(
                _plotly_img(rgb, f"CosMx Label QC — {fov_name}"),
                use_container_width=True, key="bqc_b_img",
                on_select="rerun", selection_mode=["points"],
            )
        with col_leg:
            st.markdown("**Legend**")
            legend_items = [
                ("cosmx_nucleus",  "CosMx nucleus",  "🔵"),
                ("cosmx_cell",     "CosMx cell",     "🟢"),
                ("cosmx_membrane", "CosMx membrane", "🟣"),
                ("custom_mask",    "Custom mask",    "🔴"),
                ("overlap",        "Overlap",        "🟡"),
            ]
            for key, label, emoji in legend_items:
                c = LAYER_COLORS.get(key, (128,128,128))
                st.markdown(
                    f'<span style="color:rgb{c}">{emoji}</span> {label}',
                    unsafe_allow_html=True,
                )

        # Click-to-inspect (using custom mask)
        if ev and hasattr(ev, "selection") and ev.selection.points:
            pt = ev.selection.points[0]
            H, W = ref.shape[:2]
            sc = max_dim / max(H, W)
            ox2, oy2 = int(pt["x"] / sc), int(pt["y"] / sc)
            clicked_lbl = find_cell_at_xy(custom_mask_img, ox2, oy2)
            if clicked_lbl and ref is not None:
                crop = get_cell_crop(ref, custom_mask_img, clicked_lbl, padding=35)
                if crop:
                    st.image(crop["overlay_crop"], caption=f"Custom cell {clicked_lbl}",
                             width=200)

    # Metrics detail
    if label_result is not None:
        with st.expander("Matched cell details (top 50 by IoU)"):
            pc = label_result.get("per_cell_df", pd.DataFrame())
            if len(pc) > 0:
                matched = pc[pc["matched"]].sort_values("iou", ascending=False).head(50)
                st.dataframe(matched, use_container_width=True, hide_index=True)

        with st.expander("Centroid distance histogram"):
            pc = label_result.get("per_cell_df", pd.DataFrame())
            if len(pc) > 0:
                fig_cd = px.histogram(
                    pc, x="centroid_dist", nbins=40,
                    color="matched",
                    color_discrete_map={True: "#2A9D8F", False: "#E63946"},
                    title="Centroid distances (custom→CosMx nearest)",
                    labels={"centroid_dist": "Distance (px)", "matched": "Matched"},
                )
                fig_cd.update_layout(height=280, margin=dict(t=40,b=20,l=20,r=20))
                st.plotly_chart(fig_cd, use_container_width=True, key="bqc_b_cdhist")


# ══════════════════════════════════════════════════════════════════════════════
# MODE C — Side-by-side
# ══════════════════════════════════════════════════════════════════════════════
elif "C:" in mode:
    st.subheader("Mode C — Side-by-side: Otsu | CosMx Label")

    col_l, col_r = st.columns(2)

    # ── Left: Otsu ─────────────────────────────────────────────────────────
    with col_l:
        st.markdown("**Otsu-based QC**")
        if ref is not None and custom_mask_img is not None:
            from cell_segmentation_pipeline.src.dapi_qc import detect_dapi_objects
            raw_np = raw if raw is not None else np.zeros((4256,4256), np.uint16)
            dapi_lbl_otsu = detect_dapi_objects(raw_np, min_area_px=200, gaussian_sigma=3.0)
            gray_ds  = _ds_gray(ref)
            dapi_ds  = _ds_mask(dapi_lbl_otsu)
            cust_ds  = _ds_mask(custom_mask_img)

            otsu_layers = {}
            if show_otsu and dapi_ds is not None:
                otsu_layers["otsu_mask"] = dapi_ds > 0
            if show_custom and cust_ds is not None:
                otsu_layers["custom_mask"] = cust_ds > 0
            if dapi_ds is not None and cust_ds is not None:
                otsu_layers["overlap"] = (dapi_ds > 0) & (cust_ds > 0)

            rgb_otsu = build_rgb_overlay(
                gray_ds, otsu_layers, alpha=alpha,
                boundary_only=boundary_only, dilate_px=boundary_width,
            )
            st.plotly_chart(_plotly_img(rgb_otsu, "Otsu objects + custom mask"),
                            use_container_width=True, key="bqc_c_l")
            st.caption(
                f"Otsu objects: {int(dapi_lbl_otsu.max())} | "
                f"Custom cells: {int(custom_mask_img.max())}"
            )
        else:
            st.info("DAPI or mask not available.")

    # ── Right: CosMx label ────────────────────────────────────────────────
    with col_r:
        st.markdown("**CosMx native label**")
        comp_adj = get_comp_adjusted()
        if ref is not None and comp_adj is not None and custom_mask_img is not None:
            gray_ds   = _ds_gray(ref)
            nuc_ds    = _ds_mask((comp_adj==1).astype(np.uint8))
            cust_ds   = _ds_mask(custom_mask_img)

            lbl_layers = {}
            if show_cnuc and nuc_ds is not None:
                lbl_layers["cosmx_nucleus"] = nuc_ds > 0
            if show_custom and cust_ds is not None:
                lbl_layers["custom_mask"] = cust_ds > 0
            if nuc_ds is not None and cust_ds is not None:
                lbl_layers["overlap"] = (nuc_ds > 0) & (cust_ds > 0)

            rgb_lbl = build_rgb_overlay(
                gray_ds, lbl_layers, alpha=alpha,
                boundary_only=boundary_only, dilate_px=boundary_width,
            )
            st.plotly_chart(_plotly_img(rgb_lbl, "CosMx nucleus + custom mask"),
                            use_container_width=True, key="bqc_c_r")
            st.caption(
                f"CosMx nucleus px: {int((comp_adj==1).sum()):,} | "
                f"Custom mask px: {int((custom_mask_img>0).sum()):,}"
            )
        else:
            st.info("CompartmentLabels or mask not available.")

    # ── Metrics comparison table ─────────────────────────────────────────
    if label_result is not None:
        st.subheader("対応関係指標")
        st.caption("Population IoU / Dice / Boundary F1 はセグメンテーション定義の違いにより除外。")
        n_custom  = label_result["custom_n_cells"]
        n_cosmx   = label_result["cosmx_n_cells"]
        n_matched = label_result["matched_cells"]
        med_dist  = label_result["median_centroid_dist"]
        mean_dist = label_result["mean_centroid_dist"]
        mean_iou  = label_result["mean_iou_matched"]
        rows = [
            {"指標": "Custom cells",           "値": n_custom,
             "備考": ""},
            {"指標": "CosMx cells",            "値": n_cosmx,
             "備考": ""},
            {"指標": "Cell count ratio",       "値": f"{n_custom/n_cosmx:.2f}×",
             "備考": "1.0 が理想、2.0以上は過分割疑い"},
            {"指標": "Matched pairs",          "値": n_matched,
             "備考": f"match rate {n_matched/n_custom:.1%}"},
            {"指標": "Median centroid dist",   "値": f"{med_dist:.1f} px",
             "備考": "小さいほど空間的整合性が高い"},
            {"指標": "Mean centroid dist",     "値": f"{mean_dist:.1f} px",
             "備考": ""},
            {"指標": "Matched-pair IoU (mean)","値": f"{mean_iou:.3f}",
             "備考": "近いペア同士の IoU（0.3以上が良好）"},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE D — Overlay (all layers)
# ══════════════════════════════════════════════════════════════════════════════
elif "D:" in mode:
    st.subheader("Mode D — Full Layer Overlay")

    if ref is None:
        st.warning("No reference image found.")
        st.stop()

    gray_ds   = _ds_gray(ref)
    comp_adj  = get_comp_adjusted()
    cell_adj  = get_cell_lbl_adjusted()

    # Build all requested layers
    layers = {}
    raw_np = raw if raw is not None else np.zeros((4256,4256), np.uint16)

    if show_cnuc and comp_adj is not None:
        layers["cosmx_nucleus"]  = _ds_mask((comp_adj == 1).astype(np.uint8)) > 0
    if show_ccell and cell_adj is not None:
        layers["cosmx_cell"]     = _ds_mask((cell_adj > 0).astype(np.uint8)) > 0
    if show_cmem and comp_adj is not None:
        layers["cosmx_membrane"] = _ds_mask((comp_adj == 3).astype(np.uint8)) > 0
    if show_custom and custom_mask_img is not None:
        layers["custom_mask"]    = _ds_mask(custom_mask_img) > 0
    if show_overlap and comp_adj is not None and custom_mask_img is not None:
        n_ds = _ds_mask((comp_adj==1).astype(np.uint8))
        c_ds = _ds_mask(custom_mask_img)
        layers["overlap"] = (n_ds > 0) & (c_ds > 0)
    if show_fn and comp_adj is not None and custom_mask_img is not None:
        n_ds = _ds_mask((comp_adj==1).astype(np.uint8))
        c_ds = _ds_mask(custom_mask_img)
        layers["diff_fn"] = (n_ds > 0) & ~(c_ds > 0)  # nucleus but no custom
    if show_fp and comp_adj is not None and custom_mask_img is not None:
        cell_ds = _ds_mask((cell_adj > 0).astype(np.uint8)) if cell_adj is not None else None
        c_ds    = _ds_mask(custom_mask_img)
        if cell_ds is not None:
            layers["diff_fp"] = (c_ds > 0) & ~(cell_ds > 0)  # custom but no CosMx cell
    if show_otsu and custom_mask_img is not None:
        from cell_segmentation_pipeline.src.dapi_qc import detect_dapi_objects
        otsu_lbl = detect_dapi_objects(raw_np, min_area_px=200, gaussian_sigma=3.0)
        layers["otsu_mask"] = _ds_mask(otsu_lbl) > 0

    rgb = build_rgb_overlay(
        gray_ds, layers, alpha=alpha,
        boundary_only=boundary_only, dilate_px=boundary_width,
    )

    # Interactive Plotly figure with click
    ev_d = st.plotly_chart(
        _plotly_img(rgb, f"Full overlay — {fov_name} ({model_name})"),
        use_container_width=True, key="bqc_d_img",
        on_select="rerun", selection_mode=["points"],
    )

    # Legend panel
    st.markdown("**Active layers:**")
    legend_cols = st.columns(4)
    for i, (key, name, emoji) in enumerate([
        ("cosmx_nucleus",  "CosMx Nucleus",  "🔵"),
        ("cosmx_cell",     "CosMx Cell",     "🟢"),
        ("cosmx_membrane", "CosMx Membrane", "🟣"),
        ("custom_mask",    "Custom mask",    "🔴"),
        ("overlap",        "Overlap",        "🟡"),
        ("diff_fn",        "Missed (FN)",    "🟠"),
        ("diff_fp",        "Extra (FP)",     "🔮"),
        ("otsu_mask",      "Otsu objects",   "🩵"),
    ]):
        if key in layers:
            legend_cols[i % 4].markdown(f"{emoji} {name}")

    # Click-to-inspect
    if ev_d and hasattr(ev_d, "selection") and ev_d.selection.points:
        pt = ev_d.selection.points[0]
        H, W = ref.shape[:2]
        sc   = max_dim / max(H, W)
        ox2, oy2 = int(pt["x"] / sc), int(pt["y"] / sc)

        with st.expander("Clicked pixel info", expanded=True):
            st.markdown(f"Pixel: x={ox2}, y={oy2}")
            if comp_arr is not None:
                comp_val = int(comp_arr[oy2, ox2]) if 0<=oy2<comp_arr.shape[0] and 0<=ox2<comp_arr.shape[1] else -1
                comp_name = {0:"background", 1:"nucleus", 2:"cytoplasm", 3:"membrane"}.get(comp_val, "?")
                st.markdown(f"CompartmentLabels: **{comp_val}** ({comp_name})")
            if cell_lbl_arr is not None:
                cell_id = int(cell_lbl_arr[oy2, ox2]) if 0<=oy2<cell_lbl_arr.shape[0] and 0<=ox2<cell_lbl_arr.shape[1] else 0
                st.markdown(f"CellLabels: **{cell_id}** ({'cell' if cell_id > 0 else 'background'})")
            if custom_mask_img is not None:
                cust_lbl = find_cell_at_xy(custom_mask_img, ox2, oy2)
                st.markdown(f"Custom mask label: **{cust_lbl or 0}**")
                if cust_lbl and ref is not None:
                    crop = get_cell_crop(ref, custom_mask_img, cust_lbl, padding=35)
                    if crop:
                        st.image(crop["overlay_crop"], caption=f"Custom cell {cust_lbl}", width=180)

    # FOV summary — meaningful metrics only
    if label_result is not None:
        with st.expander("このFOVのサマリー指標"):
            # Show only the meaningful correspondence metrics
            _SKIP = {
                "nucleus_iou", "nucleus_dice", "cell_iou", "cell_dice",
                "boundary_f1_vs_nucleus", "boundary_precision_vs_nucleus",
                "boundary_recall_vs_nucleus", "boundary_f1_vs_cell",
                "mean_area_diff_frac", "median_area_diff_frac",
            }
            metrics_disp = {
                k: round(v, 4)
                for k, v in label_result.items()
                if isinstance(v, float)
                and k not in _SKIP
                and not k.startswith("false_")
                and "per_cell" not in k
            }
            st.dataframe(
                pd.DataFrame([{"指標": k, "値": v} for k, v in metrics_disp.items()]),
                use_container_width=True, hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Export section (always visible)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Export")
exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    if label_result is not None:
        pc = label_result.get("per_cell_df", pd.DataFrame())
        if not pc.empty:
            csv_bytes = pc.to_csv(index=False).encode()
            st.download_button(
                "⬇️ per-cell metrics CSV",
                data=csv_bytes,
                file_name=f"{fov_name}_{model_name}_boundary_qc.csv",
                mime="text/csv",
                key="bqc_dl_pc",
            )

with exp_col2:
    if label_result is not None:
        row = {
            k: v for k, v in label_result.items()
            if not isinstance(v, (pd.DataFrame, np.ndarray, dict))
        }
        summary_bytes = pd.DataFrame([row]).to_csv(index=False).encode()
        st.download_button(
            "⬇️ summary metrics CSV",
            data=summary_bytes,
            file_name=f"{fov_name}_{model_name}_summary.csv",
            mime="text/csv",
            key="bqc_dl_summary",
        )

with exp_col3:
    if st.button("💾 Save overlay PNG", key="bqc_save_png"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path as _Path

        qc_dir = PROJECT_ROOT / "outputs" / "cell_segmentation_pipeline" / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)

        if ref is not None:
            comp_a = get_comp_adjusted()
            cell_a = get_cell_lbl_adjusted()
            ly = {}
            if comp_a is not None:
                ly["cosmx_nucleus"] = (comp_a == 1)
            if custom_mask_img is not None:
                ly["custom_mask"] = custom_mask_img > 0
            if comp_a is not None and custom_mask_img is not None:
                ly["overlap"] = (comp_a == 1) & (custom_mask_img > 0)
            rgb_save = build_rgb_overlay(
                norm_uint8(ref), ly, alpha=0.55,
                boundary_only=True, dilate_px=1,
            )
            fig_s, ax_s = plt.subplots(figsize=(10, 10))
            ax_s.imshow(rgb_save)
            ax_s.axis("off")
            ax_s.set_title(f"{fov_name} {model_name} — boundary overlay")
            save_path = qc_dir / f"{fov_name}_{model_name}_boundary_overlay.png"
            fig_s.tight_layout()
            fig_s.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig_s)
            st.success(f"Saved: {save_path}")
