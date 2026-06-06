"""DAPI vs Mask Segmentation Accuracy QC — Streamlit page.

Tabs:
  1. Overview    — metric cards, stacked bar, download
  2. Visual QC   — interactive Plotly overlay, click-to-inspect
  3. Manual Review — random cell sampling, label radio buttons, save CSV
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
# 07_DAPI_Mask_QC.py lives at:
#   CosMx_2026/cell_segmentation_pipeline/app/pages/07_DAPI_Mask_QC.py
# parents[0]=pages  [1]=app  [2]=cell_segmentation_pipeline  [3]=CosMx_2026
_PAGE_PATH   = Path(__file__).resolve()
_APP_DIR     = _PAGE_PATH.parents[1]          # app/
_PIPELINE_DIR = _PAGE_PATH.parents[2]         # cell_segmentation_pipeline/
_PROJECT_ROOT = _PAGE_PATH.parents[3]         # CosMx_2026/

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_APP_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.image_loader import (
    DEFAULT_OUTPUT_ROOTS,
    discover_fovs,
    discover_models,
    load_enhanced,
    load_mask,
    load_raw_dapi,
    norm_uint8,
    downsample_img,
    downsample_mask,
)
from utils.cell_cropper import get_cell_crop, find_cell_at_xy
from utils.overlay_utils import compose_overlay
from utils.dapi_qc_loader import (
    load_qc_summary,
    load_qc_per_cell,
    run_fov_qc_cached,
    list_vis_images,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DAPI Mask QC", layout="wide")
st.title("DAPI Nucleus Quality QC  （評価軸 1/3）")
st.caption("custom mask が DAPI 核の上に乗っているかを評価します。")

st.info(
    "**cpsam mask についての注意** — cpsam は DAPI 核の中心だけでなく周辺領域も含むため、"
    "mask 全体の DAPI-positive fraction は低く出ることがあります。"
    "この指標は segmentation の失敗判定ではなく、"
    "**極端なノイズ mask や DAPI から大きく外れた mask の検出** に使います。"
    "\n\n"
    "SNR が 2.0 以上であれば、mask は概ね DAPI 核の上に乗っていると判断してください。"
)

# ── Constants ──────────────────────────────────────────────────────────────────
_QC_ROOT_DEFAULT = str(_PROJECT_ROOT / "outputs" / "cell_segmentation_pipeline" / "qc")
_REVIEW_CSV_NAME = "manual_segmentation_review.csv"

_REVIEW_LABELS = ["Good", "Missed", "Noise", "Over-segmented", "Merged", "Unclear"]
_LABEL_COLOR   = {
    "Good":           "green",
    "Missed":         "red",
    "Noise":          "gray",
    "Over-segmented": "orange",
    "Merged":         "purple",
    "Unclear":        "#888888",
    "":               "#cccccc",
}

# ── Session state ──────────────────────────────────────────────────────────────
def _init_state() -> None:
    if "dqc_result"        not in st.session_state: st.session_state.dqc_result       = {}
    if "dqc_fov"           not in st.session_state: st.session_state.dqc_fov          = None
    if "dqc_model"         not in st.session_state: st.session_state.dqc_model        = None
    if "dqc_root"          not in st.session_state: st.session_state.dqc_root         = None
    if "rev_crops"         not in st.session_state: st.session_state.rev_crops        = []
    if "rev_labels"        not in st.session_state: st.session_state.rev_labels       = {}
    if "rev_comments"      not in st.session_state: st.session_state.rev_comments     = {}
    if "rev_seed"          not in st.session_state: st.session_state.rev_seed         = 0
    if "rev_reviewer"      not in st.session_state: st.session_state.rev_reviewer     = ""
    if "show_matched"      not in st.session_state: st.session_state.show_matched     = True
    if "show_missed"       not in st.session_state: st.session_state.show_missed      = True
    if "show_low_dapi"     not in st.session_state: st.session_state.show_low_dapi    = True
    if "show_extra"        not in st.session_state: st.session_state.show_extra       = True

_init_state()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("DAPI QC Controls")

    # Output root selector
    output_root = st.selectbox(
        "Output root",
        DEFAULT_OUTPUT_ROOTS,
        format_func=lambda p: Path(p).name,
        key="dqc_output_root",
    )

    # FOV selector
    fovs = discover_fovs(output_root)
    if not fovs:
        st.warning("No masks found in selected output root.")
        st.stop()

    fov_id = st.selectbox("FOV", fovs, key="dqc_fov_sel")

    # Model selector
    models = discover_models(fov_id, output_root)
    if not models:
        st.warning(f"No models found for {fov_id}.")
        st.stop()
    model = st.selectbox("Model", models, key="dqc_model_sel")

    st.divider()
    st.markdown("**QC Thresholds**")
    min_area  = st.number_input("Min cell area (px²)", 10,  1000, 50,    key="dqc_min_area")
    max_area  = st.number_input("Max cell area (px²)", 500, 50000, 10000, key="dqc_max_area")
    iou_thresh = st.slider("IoU threshold", 0.1, 0.8, 0.3, 0.05, key="dqc_iou")
    max_dim   = st.select_slider("Display resolution", [512, 768, 1024, 1536], value=1024,
                                  key="dqc_maxdim")

    st.divider()
    qc_root = st.text_input("Pre-computed QC root", value=_QC_ROOT_DEFAULT, key="dqc_qcroot")

    if st.button("Run / Refresh QC", type="primary", key="dqc_run"):
        # Clear cached result to force re-run
        st.session_state.dqc_result = {}
        st.session_state.dqc_fov    = None

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA + RUN QC
# ═══════════════════════════════════════════════════════════════════════════════

# Load images
raw      = load_raw_dapi(fov_id, 0)
enhanced = load_enhanced(fov_id, output_root)
mask     = load_mask(fov_id, model, output_root)

if raw is None or mask is None:
    st.error(
        f"Could not load images for {fov_id} / {model}. "
        "Check the output root and model name."
    )
    st.stop()

ref = enhanced if enhanced is not None else raw

# Check if we need to (re-)run QC
_need_run = (
    st.session_state.dqc_fov   != fov_id
    or st.session_state.dqc_model != model
    or st.session_state.dqc_root  != output_root
    or not st.session_state.dqc_result
)

if _need_run:
    # Try loading from pre-computed CSV first
    _summary_df   = load_qc_summary(qc_root)
    _per_cell_df  = load_qc_per_cell(qc_root)

    _has_csv = (
        _summary_df  is not None
        and _per_cell_df is not None
        and "fov_id"     in _summary_df.columns
        and "model_name" in _summary_df.columns
    )

    if _has_csv:
        _row = _summary_df[
            (_summary_df["fov_id"] == fov_id) & (_summary_df["model_name"] == model)
        ]
        _pcd = _per_cell_df[
            (_per_cell_df["fov_id"] == fov_id) & (_per_cell_df["model_name"] == model)
        ] if "fov_id" in _per_cell_df.columns else pd.DataFrame()
    else:
        _row = pd.DataFrame()
        _pcd = pd.DataFrame()

    if not _row.empty and not _pcd.empty:
        # Reconstruct qc_result dict from CSV (scalar-only; no dapi_labels or per_dapi_df)
        _scalar = _row.iloc[0].to_dict()
        st.session_state.dqc_result = {**_scalar, "per_cell_df": _pcd, "dapi_labels": None}
        st.session_state.dqc_fov   = fov_id
        st.session_state.dqc_model = model
        st.session_state.dqc_root  = output_root
    else:
        # Run QC on-the-fly
        with st.spinner(f"Running DAPI QC for {fov_id} / {model}…"):
            _enh = enhanced if enhanced is not None else raw.astype(np.float32)
            _qc  = run_fov_qc_cached(
                raw      = raw,
                enhanced = _enh,
                mask     = mask,
                fov_id   = fov_id,
                model_name = model,
                min_area_px  = int(min_area),
                max_area_px  = int(max_area),
                iou_threshold = float(iou_thresh),
            )
        st.session_state.dqc_result = _qc
        st.session_state.dqc_fov   = fov_id
        st.session_state.dqc_model = model
        st.session_state.dqc_root  = output_root

qc = st.session_state.dqc_result
per_cell_df = qc.get("per_cell_df", pd.DataFrame())

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_visual, tab_manual = st.tabs([
    "Overview",
    "Visual QC",
    "Manual Review",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader(f"DAPI Nucleus Quality — {fov_id} / {model}")

    # ── SNR badge ─────────────────────────────────────────────────────────────
    import sys as _sys
    _sys.path.insert(0, str(_PROJECT_ROOT))
    from cell_segmentation_pipeline.src.composite_qc_score import classify_dapi_quality

    snr  = qc.get("signal_to_background_ratio", float("nan"))
    dapi_status, dapi_color = classify_dapi_quality(float(snr) if snr == snr else float("nan"))
    _badge_style = (
        f"display:inline-block;padding:4px 14px;border-radius:6px;"
        f"background-color:{'#d4edda' if dapi_color=='green' else '#fff3cd' if dapi_color=='orange' else '#f8d7da'};"
        f"color:{'#155724' if dapi_color=='green' else '#856404' if dapi_color=='orange' else '#721c24'};"
        f"font-weight:bold;font-size:16px;"
    )
    st.markdown(
        f'DAPI Quality: <span style="{_badge_style}">{dapi_status}</span>'
        f'　SNR = {snr:.2f}x',
        unsafe_allow_html=True,
    )
    _snr_guide = {
        "Good":       "SNR ≥ 3.0 — mask は確実に DAPI 核の上に乗っています。",
        "Acceptable": "SNR 2.0〜3.0 — 許容範囲。一部の mask が核からはみ出している可能性があります。",
        "Warning":    "SNR < 2.0 — DAPI 信号が弱い FOV です。取得条件や前処理を確認してください。",
    }
    st.caption(_snr_guide.get(dapi_status, ""))

    st.divider()

    # ── Primary metric cards ───────────────────────────────────────────────────
    ldf  = qc.get("low_dapi_mask_fraction", float("nan"))
    dpf  = qc.get("dapi_positive_mask_fraction", float("nan"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "mask_dapi_snr",
        f"{snr:.2f}x" if snr == snr else "N/A",
        help="mask 内 DAPI 平均 / 背景 DAPI 平均",
    )
    col2.metric(
        "dapi_positive_fraction",
        f"{dpf:.1%}" if dpf == dpf else "N/A",
        help="mask 内で Otsu 閾値以上の pixel 割合（cpsam では低くなりやすい）",
    )
    col3.metric(
        "low_dapi_cell_fraction",
        f"{ldf:.1%}" if ldf == ldf else "N/A",
        help="DAPI 信号が背景 +1.5σ を下回る細胞の割合",
    )
    col4.metric(
        "cell_count",
        qc.get("cell_count", "N/A"),
        help="検出された細胞数",
    )

    # ── Secondary metrics row ─────────────────────────────────────────────────
    med_a = qc.get("size_median_area_px", float("nan"))
    mea_a = qc.get("size_mean_area_px",   float("nan"))
    sf    = qc.get("small_mask_fraction", float("nan"))
    lf    = qc.get("large_mask_fraction", float("nan"))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("median_mask_area", f"{med_a:.0f} px²" if med_a == med_a else "N/A")
    c6.metric("mean_mask_area",   f"{mea_a:.0f} px²" if mea_a == mea_a else "N/A")
    c7.metric("small_object_fraction", f"{sf:.1%}" if sf == sf else "N/A",
              help=f"< {qc.get('small_area_threshold', 50)} px²")
    c8.metric("large_object_fraction", f"{lf:.1%}" if lf == lf else "N/A",
              help=f"> {qc.get('large_area_threshold', 10000)} px²")

    # ── Otsu-based DAPI comparison (secondary) ────────────────────────────────
    with st.expander("Otsu DAPI comparison metrics（参考値）", expanded=False):
        st.caption(
            "以下は Otsu 閾値で検出した DAPI オブジェクトと mask を比較した参考値です。"
            "cpsam は CosMx native より多くの細胞を検出することがあるため、"
            "Recall / Precision が低くても直ちに失敗ではありません。"
        )
        prec = qc.get("estimated_precision", float("nan"))
        rec  = qc.get("estimated_recall",    float("nan"))
        miou = qc.get("mean_iou",            float("nan"))
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        oc1.metric("DAPI objects",    qc.get("dapi_object_count", "N/A"))
        oc2.metric("Matched",         qc.get("matched_count", "N/A"))
        oc3.metric("Coverage (precision)", f"{prec:.3f}" if prec==prec else "N/A")
        oc4.metric("Recall",          f"{rec:.3f}"  if rec==rec   else "N/A")
        oc5.metric("Mean IoU (matched)", f"{miou:.3f}" if miou==miou else "N/A")

    st.divider()

    # ── Stacked bar (if summary CSV has multiple FOVs) ─────────────────────
    _summary_df = load_qc_summary(qc_root)
    if _summary_df is not None and len(_summary_df) > 1:
        st.markdown("#### All FOVs — segmentation breakdown")

        bar_data = []
        for _, r in _summary_df.iterrows():
            fov_lbl = f"{r.get('fov_id','?')} / {r.get('model_name','?')}"
            matched  = int(r.get("matched_count",      0))
            missed   = int(r.get("missed_dapi_count",  0))
            low_d    = int(r.get("low_dapi_mask_count", 0))
            extra    = int(r.get("extra_mask_count",   0))
            bar_data.extend([
                {"FOV": fov_lbl, "Category": "Matched",   "Count": matched},
                {"FOV": fov_lbl, "Category": "Missed DAPI", "Count": missed},
                {"FOV": fov_lbl, "Category": "Low DAPI",  "Count": low_d},
                {"FOV": fov_lbl, "Category": "Extra Mask", "Count": extra},
            ])

        if bar_data:
            bar_df  = pd.DataFrame(bar_data)
            _cmap   = {
                "Matched":    "#00C800",
                "Missed DAPI": "#E63946",
                "Low DAPI":   "#FFD700",
                "Extra Mask": "#4575B4",
            }
            fig_bar = px.bar(
                bar_df, x="FOV", y="Count", color="Category",
                color_discrete_map=_cmap,
                barmode="stack",
                title="Segmentation Category Breakdown by FOV",
            )
            fig_bar.update_layout(
                height=350, margin=dict(t=40, b=60, l=20, r=20),
                legend=dict(orientation="h", y=-0.3),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="dqc_bar_all")

    # ── Per-cell table preview ─────────────────────────────────────────────
    with st.expander("Per-cell QC data (preview)"):
        if len(per_cell_df) > 0:
            disp_cols = [c for c in [
                "label", "mean_intensity", "area", "is_low_dapi",
                "matched_dapi_label", "iou", "matched", "auto_qc_label",
            ] if c in per_cell_df.columns]
            st.dataframe(per_cell_df[disp_cols].head(200),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No per-cell data available.")

    # ── Download QC summary CSV ────────────────────────────────────────────
    st.divider()
    _dl_col1, _dl_col2 = st.columns(2)

    if _summary_df is not None:
        _dl_col1.download_button(
            "Download QC summary CSV",
            data=_summary_df.to_csv(index=False).encode(),
            file_name="dapi_mask_qc_summary.csv",
            mime="text/csv",
            key="dqc_dl_summary",
        )

    if len(per_cell_df) > 0:
        _dl_col2.download_button(
            "Download per-cell CSV",
            data=per_cell_df.to_csv(index=False).encode(),
            file_name="dapi_mask_qc_per_cell.csv",
            mime="text/csv",
            key="dqc_dl_percell",
        )

    # ── Pre-computed visualizations ────────────────────────────────────────
    vis_paths = list_vis_images(qc_root, fov_id, model)
    existing_vis = {desc: p for desc, p in vis_paths.items() if p.exists()}
    if existing_vis:
        with st.expander("Pre-computed visualization PNGs"):
            for desc, p in existing_vis.items():
                st.markdown(f"**{desc}**")
                st.image(str(p), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VISUAL QC
# ══════════════════════════════════════════════════════════════════════════════
with tab_visual:
    st.subheader("Visual QC — Click to Inspect")

    # ── Toggle buttons row ─────────────────────────────────────────────────
    toggle_col1, toggle_col2, toggle_col3, toggle_col4 = st.columns(4)
    with toggle_col1:
        show_matched  = st.checkbox("Show matched",  value=st.session_state.show_matched,
                                     key="dqc_tog_matched")
    with toggle_col2:
        show_missed   = st.checkbox("Show missed",   value=st.session_state.show_missed,
                                     key="dqc_tog_missed")
    with toggle_col3:
        show_low_dapi = st.checkbox("Show low DAPI", value=st.session_state.show_low_dapi,
                                     key="dqc_tog_low")
    with toggle_col4:
        show_extra    = st.checkbox("Show extra",    value=st.session_state.show_extra,
                                     key="dqc_tog_extra")

    # ── Build colored overlay ──────────────────────────────────────────────

    _C_MATCHED   = np.array([0, 200, 0],    dtype=np.uint8)
    _C_LOW_DAPI  = np.array([255, 215, 0],  dtype=np.uint8)
    _C_EXTRA     = np.array([69, 117, 180], dtype=np.uint8)

    # Build label category sets from per_cell_df
    _matched_labels:  set[int] = set()
    _low_dapi_labels: set[int] = set()
    _extra_labels:    set[int] = set()

    if len(per_cell_df) > 0:
        for _, _r in per_cell_df.iterrows():
            _lbl = int(_r.get("label", 0))
            _ql  = _r.get("auto_qc_label", "extra")
            if _ql == "good":
                _matched_labels.add(_lbl)
            elif _ql == "low_dapi":
                _low_dapi_labels.add(_lbl)
            else:
                _extra_labels.add(_lbl)

    # Downsample
    _gray_ds  = downsample_img(norm_uint8(ref), max_dim)
    _mask_ds  = downsample_mask(mask, max_dim)

    # Build RGB color mask
    _rgb_ov = np.zeros((*_mask_ds.shape, 3), dtype=np.uint8)
    for _lbl in np.unique(_mask_ds):
        if _lbl == 0:
            continue
        _px = _mask_ds == _lbl
        if _lbl in _matched_labels   and show_matched:
            _rgb_ov[_px] = _C_MATCHED
        elif _lbl in _low_dapi_labels and show_low_dapi:
            _rgb_ov[_px] = _C_LOW_DAPI
        elif _lbl in _extra_labels    and show_extra:
            _rgb_ov[_px] = _C_EXTRA

    # Blend
    _base = np.stack([_gray_ds, _gray_ds, _gray_ds], axis=-1).astype(np.float32)
    _fg   = _rgb_ov.any(axis=-1)
    _blend = _base.copy()
    _blend[_fg] = (0.55 * _base[_fg] + 0.45 * _rgb_ov[_fg].astype(np.float32))
    _blend = np.clip(_blend, 0, 255).astype(np.uint8)

    # ── Two-column layout ─────────────────────────────────────────────────
    col_img, col_info = st.columns([3, 1])

    with col_img:
        fig_ov = px.imshow(_blend, aspect="equal")
        fig_ov.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            title="DAPI + Mask QC Overlay (click a cell)",
            dragmode="pan",
        )
        click_ev = st.plotly_chart(
            fig_ov,
            use_container_width=True,
            key="dqc_overlay_fig",
            on_select="rerun",
            selection_mode=["points"],
        )

    with col_info:
        st.markdown("**Color legend**")
        _legend_html = """
<div style="font-size:13px; line-height:2">
  <span style="background:#00C800;padding:2px 10px;border-radius:4px;color:white">Matched</span><br>
  <span style="background:#FFD700;padding:2px 10px;border-radius:4px;color:black">Low DAPI</span><br>
  <span style="background:#4575B4;padding:2px 10px;border-radius:4px;color:white">Extra mask</span><br>
  <span style="background:#E63946;padding:2px 10px;border-radius:4px;color:white">Missed DAPI</span>
</div>
"""
        st.markdown(_legend_html, unsafe_allow_html=True)
        st.divider()

        # Resolve clicked cell
        _selected_label: int | None = None
        if click_ev and hasattr(click_ev, "selection") and click_ev.selection.points:
            _pt = click_ev.selection.points[0]
            _H, _W = ref.shape[:2]
            _sc = max_dim / max(_H, _W)
            _ox = int(_pt["x"] / _sc)
            _oy = int(_pt["y"] / _sc)
            _selected_label = find_cell_at_xy(mask, _ox, _oy)

        # Manual label input
        _max_label = max(int(mask.max()), 1)
        _manual_label = st.number_input(
            "Cell label",
            min_value=1,
            max_value=_max_label,
            value=_selected_label if _selected_label else 1,
            step=1,
            key="dqc_cell_label_input",
        )
        if _selected_label is None:
            _selected_label = int(_manual_label)
        else:
            st.success(f"Clicked: label **{_selected_label}**")

        # Cell crop
        _crop = get_cell_crop(ref, mask, _selected_label, padding=30)
        if _crop:
            st.image(_crop["overlay_crop"], use_container_width=True,
                     caption=f"Cell {_selected_label}")
        else:
            st.info("Click a cell or enter a label above.")

        # Cell metrics
        if len(per_cell_df) > 0 and "label" in per_cell_df.columns:
            _cell_row = per_cell_df[per_cell_df["label"] == _selected_label]
            if not _cell_row.empty:
                _cr = _cell_row.iloc[0]
                st.markdown("**Cell metrics**")
                st.markdown(f"- Area: **{_cr.get('area', 'N/A')} px²**")
                _di = _cr.get("mean_intensity", None)
                if _di is not None:
                    st.markdown(f"- DAPI intensity: **{_di:.1f}**")
                _iou = _cr.get("iou", None)
                if _iou is not None:
                    st.markdown(f"- IoU: **{_iou:.3f}**")
                _ql = _cr.get("auto_qc_label", None)
                if _ql:
                    _ql_colors = {
                        "good":     "green",
                        "low_dapi": "orange",
                        "extra":    "blue",
                        "missed":   "red",
                    }
                    _color = _ql_colors.get(_ql, "gray")
                    st.markdown(
                        f'- Auto QC label: <span style="color:{_color};'
                        f'font-weight:bold">{_ql}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(f"No QC data for label {_selected_label}.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MANUAL REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Manual Segmentation Review")
    st.caption(
        "Sample random cells, label each one, and save the review CSV. "
        "Existing reviews are loaded automatically if the file exists."
    )

    # ── Reviewer info ──────────────────────────────────────────────────────
    rev_col_a, rev_col_b, rev_col_c = st.columns(3)
    with rev_col_a:
        reviewer_name = st.text_input(
            "Reviewer name",
            value=st.session_state.rev_reviewer,
            key="dqc_reviewer_name",
        )
        st.session_state.rev_reviewer = reviewer_name
    with rev_col_b:
        n_sample = st.slider("Cells per sample", 3, 30, 9, 3, key="dqc_n_sample")
    with rev_col_c:
        padding = st.slider("Crop padding (px)", 10, 60, 25, 5, key="dqc_padding")

    # ── Load existing review file ─────────────────────────────────────────
    _review_path = Path(qc_root) / _REVIEW_CSV_NAME
    if _review_path.exists() and "rev_loaded" not in st.session_state:
        try:
            _existing = pd.read_csv(str(_review_path))
            for _, _rrow in _existing.iterrows():
                _k = f"{_rrow.get('fov_id','?')}_{_rrow.get('model_name','?')}_{int(_rrow.get('cell_label',0))}"
                st.session_state.rev_labels[_k]   = str(_rrow.get("label", ""))
                st.session_state.rev_comments[_k] = str(_rrow.get("comment", ""))
            st.session_state.rev_loaded = True
            st.info(f"Loaded {len(_existing)} existing reviews from {_review_path}")
        except Exception as _e:
            st.warning(f"Could not load existing review: {_e}")

    # ── Sample random cells ───────────────────────────────────────────────
    if st.button("Sample random cells", key="dqc_sample_btn"):
        st.session_state.rev_seed  += 1
        st.session_state.rev_crops  = []

    # Build crop list if empty or FOV/model changed
    _need_crops = (
        not st.session_state.rev_crops
        or st.session_state.dqc_fov   != st.session_state.get("_rev_fov_loaded")
        or st.session_state.dqc_model != st.session_state.get("_rev_model_loaded")
    )

    if _need_crops:
        # Sample from per_cell_df if available, otherwise from mask directly
        if len(per_cell_df) > 0 and "label" in per_cell_df.columns:
            _available = per_cell_df["label"].dropna().astype(int).tolist()
            rng = np.random.default_rng(st.session_state.rev_seed)
            _chosen = rng.choice(
                _available, size=min(n_sample, len(_available)), replace=False
            ).tolist()
        else:
            _uniq = np.unique(mask)
            _uniq = _uniq[_uniq > 0]
            rng   = np.random.default_rng(st.session_state.rev_seed)
            _chosen = rng.choice(_uniq, size=min(n_sample, len(_uniq)), replace=False).tolist()

        with st.spinner("Extracting cell crops…"):
            _crops = []
            for _cid in _chosen:
                _c = get_cell_crop(ref, mask, int(_cid), padding=padding)
                if _c is not None:
                    _crops.append(_c)
        st.session_state.rev_crops = _crops
        st.session_state["_rev_fov_loaded"]   = st.session_state.dqc_fov
        st.session_state["_rev_model_loaded"] = st.session_state.dqc_model

    crops = st.session_state.rev_crops

    if not crops:
        st.warning("No cell crops available. Check that the mask has detected cells.")
    else:
        # ── Cell grid ───────────────────────────────────────────────────────
        N_COLS = 3
        grid_rows = [crops[i:i + N_COLS] for i in range(0, len(crops), N_COLS)]

        for grid_row in grid_rows:
            gcols = st.columns(N_COLS)
            for gcol, crop in zip(gcols, grid_row):
                cid = crop["cell_id"]
                _key = f"{fov_id}_{model}_{cid}"
                _cur_label   = st.session_state.rev_labels.get(_key, "")
                _cur_comment = st.session_state.rev_comments.get(_key, "")
                _color = _LABEL_COLOR.get(_cur_label, "#cccccc")

                with gcol:
                    # Coloured border
                    _border = f"border:3px solid {_color};padding:4px;border-radius:6px;"
                    st.markdown(f'<div style="{_border}">', unsafe_allow_html=True)
                    st.image(
                        crop["overlay_crop"],
                        use_container_width=True,
                        caption=(
                            f"Cell {cid} | {crop['area']} px² | "
                            f"⌀{crop['est_diam_px']:.0f}px"
                        ),
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    # QC info from per_cell_df
                    if len(per_cell_df) > 0 and "label" in per_cell_df.columns:
                        _pr = per_cell_df[per_cell_df["label"] == cid]
                        if not _pr.empty:
                            _auto = _pr.iloc[0].get("auto_qc_label", "")
                            _iou  = _pr.iloc[0].get("iou", None)
                            _info = f"auto: **{_auto}**"
                            if _iou is not None:
                                _info += f" | IoU: {_iou:.2f}"
                            st.caption(_info)

                    # Label radio
                    _sel = st.radio(
                        f"Label — cell {cid}",
                        [""] + _REVIEW_LABELS,
                        index=([""] + _REVIEW_LABELS).index(_cur_label)
                              if _cur_label in _REVIEW_LABELS else 0,
                        key=f"dqc_rev_lbl_{fov_id}_{model}_{cid}_{st.session_state.rev_seed}",
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                    if _sel != _cur_label:
                        st.session_state.rev_labels[_key] = _sel
                        st.rerun()

                    # Comment
                    _comm = st.text_input(
                        "Comment",
                        value=_cur_comment,
                        key=f"dqc_rev_comm_{fov_id}_{model}_{cid}_{st.session_state.rev_seed}",
                        label_visibility="collapsed",
                        placeholder="Optional comment…",
                    )
                    if _comm != _cur_comment:
                        st.session_state.rev_comments[_key] = _comm

        # ── Save review ────────────────────────────────────────────────────
        st.divider()
        _labeled = [k for k, v in st.session_state.rev_labels.items() if v]
        st.markdown(f"**{len(_labeled)}** cells labeled.")

        if _labeled:
            _save_rows = []
            for _k, _lbl in st.session_state.rev_labels.items():
                if not _lbl:
                    continue
                _parts = _k.split("_")
                # key format: {fov_id}_{model}_{cid}  — cid is last, fov is first 2, model is middle
                _cell_lbl = _parts[-1]
                _fov_part = _parts[0]
                _model_part = "_".join(_parts[1:-1])
                _save_rows.append({
                    "fov_id":      _fov_part,
                    "model_name":  _model_part,
                    "cell_label":  _cell_lbl,
                    "label":       _lbl,
                    "comment":     st.session_state.rev_comments.get(_k, ""),
                    "reviewer":    reviewer_name,
                })
            _review_df = pd.DataFrame(_save_rows)

            _sc1, _sc2 = st.columns(2)

            # Download button
            _sc1.download_button(
                "Download review CSV",
                data=_review_df.to_csv(index=False).encode(),
                file_name=_REVIEW_CSV_NAME,
                mime="text/csv",
                key="dqc_dl_review",
            )

            # Save to disk button
            _save_p = st.text_input(
                "Save path",
                value=str(_review_path),
                key="dqc_rev_savepath",
            )
            if _sc2.button("Save review to disk", key="dqc_rev_save_btn"):
                _sp = Path(_save_p)
                _sp.parent.mkdir(parents=True, exist_ok=True)
                # Merge with existing if present
                if _sp.exists():
                    try:
                        _old = pd.read_csv(str(_sp))
                        _merged = pd.concat([_old, _review_df], ignore_index=True)
                        _merged = _merged.drop_duplicates(
                            subset=["fov_id", "model_name", "cell_label"], keep="last"
                        )
                        _merged.to_csv(str(_sp), index=False)
                        st.success(f"Merged {len(_review_df)} rows → {_sp} ({len(_merged)} total)")
                    except Exception as _merge_err:
                        _review_df.to_csv(str(_sp), index=False)
                        st.success(f"Saved {len(_review_df)} rows → {_sp}")
                else:
                    _review_df.to_csv(str(_sp), index=False)
                    st.success(f"Saved {len(_review_df)} rows → {_sp}")

            # Preview
            with st.expander("Review table preview"):
                st.dataframe(_review_df, use_container_width=True, hide_index=True)
