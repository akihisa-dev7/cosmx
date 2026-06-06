"""Random Cell Review — manual QC labelling workbench."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from utils.cell_cropper import get_random_cells
from utils.image_loader import (
    DEFAULT_OUTPUT_ROOTS, discover_fovs, discover_models,
    get_fov_meta, load_enhanced, load_mask, load_raw_dapi,
)

st.set_page_config(page_title="Random Cell Review", layout="wide")
st.title("Random Cell Review")
st.caption("Manually label random cells — Good / Over-seg / Under-seg / Merged / Rejected")

# ── Labels and colours ────────────────────────────────────────────────────────
LABELS      = ["Good", "Over-segmented", "Under-segmented", "Merged", "Rejected"]
LABEL_EMOJI = {"Good": "✅", "Over-segmented": "⚠️", "Under-segmented": "⚠️",
               "Merged": "🔀", "Rejected": "❌", "": "—"}
LABEL_COLOR = {"Good": "green", "Over-segmented": "orange", "Under-segmented": "orange",
               "Merged": "purple", "Rejected": "red", "": "gray"}

# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    if "review_crops"  not in st.session_state: st.session_state.review_crops  = []
    if "review_labels" not in st.session_state: st.session_state.review_labels = {}
    if "review_seed"   not in st.session_state: st.session_state.review_seed   = 0
    if "review_fov"    not in st.session_state: st.session_state.review_fov    = None
    if "review_model"  not in st.session_state: st.session_state.review_model  = None
    if "all_labels"    not in st.session_state: st.session_state.all_labels    = {}  # cumulative

_init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    output_root = st.selectbox(
        "Output root", DEFAULT_OUTPUT_ROOTS,
        format_func=lambda p: Path(p).name,
        key="review_root",
    )

    fovs = discover_fovs(output_root)
    if not fovs:
        st.warning("No masks found.")
        st.stop()

    fov_id = st.selectbox("FOV", fovs, key="review_fov_sel")
    models = discover_models(fov_id, output_root)
    model  = st.selectbox("Model", models, key="review_model_sel") if models else None

    st.divider()
    n_cells  = st.slider("Cells per review set", 4, 40, 20, 4)
    min_area = st.number_input("Min area filter (px²)", 10, 500, 50)
    padding  = st.slider("Crop padding (px)", 10, 80, 25, 5)

    st.divider()
    # New random set
    if st.button("🎲 New random set", type="primary"):
        st.session_state.review_seed += 1
        st.session_state.review_crops = []   # force reload

# ── Load data ─────────────────────────────────────────────────────────────────
raw      = load_raw_dapi(fov_id)
enhanced = load_enhanced(fov_id, output_root)
mask     = load_mask(fov_id, model, output_root) if model else None
ref      = enhanced if enhanced is not None else raw

if mask is None:
    st.warning("No mask found. Run the segmentation pipeline first.")
    st.stop()

# Reload crops if FOV / model / seed changed
reload_needed = (
    not st.session_state.review_crops
    or st.session_state.review_fov   != fov_id
    or st.session_state.review_model != model
)

if reload_needed:
    with st.spinner("Sampling random cells…"):
        crops = get_random_cells(
            ref, mask,
            n=n_cells,
            seed=st.session_state.review_seed,
            min_area=min_area,
            padding=padding,
        )
    st.session_state.review_crops  = crops
    st.session_state.review_labels = {c["cell_id"]: "" for c in crops}
    st.session_state.review_fov    = fov_id
    st.session_state.review_model  = model

crops = st.session_state.review_crops

if not crops:
    st.error("No cells found with the current filters.")
    st.stop()

# ── Progress bar ─────────────────────────────────────────────────────────────
meta = get_fov_meta(fov_id)
n_labeled = sum(1 for v in st.session_state.review_labels.values() if v)
n_total   = len(crops)
prog_frac = n_labeled / n_total if n_total else 0

st.markdown(
    f"**{fov_id}** — {meta.get('condition','?')} | {meta.get('region','?')} | "
    f"model: `{model}` | seed: {st.session_state.review_seed}"
)
st.progress(prog_frac, text=f"Labelled {n_labeled} / {n_total} cells")

# ── Cell grid ─────────────────────────────────────────────────────────────────
N_COLS = 4
rows   = [crops[i:i + N_COLS] for i in range(0, len(crops), N_COLS)]

for row in rows:
    cols = st.columns(N_COLS)
    for col, crop in zip(cols, row):
        cid   = crop["cell_id"]
        label = st.session_state.review_labels.get(cid, "")
        color = LABEL_COLOR.get(label, "gray")
        emoji = LABEL_EMOJI.get(label, "—")

        with col:
            # Coloured border via caption
            border_style = f"border: 3px solid {color}; padding: 4px; border-radius: 6px;"
            st.markdown(f'<div style="{border_style}">', unsafe_allow_html=True)
            st.image(
                crop["overlay_crop"],
                use_container_width=True,
                caption=f"Cell {cid} | {crop['area']} px² | ⌀{crop['est_diam_px']:.0f}px",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Label selector
            sel = st.selectbox(
                f"Label — cell {cid}",
                [""] + LABELS,
                index=([""] + LABELS).index(label) if label in LABELS else 0,
                key=f"lbl_{fov_id}_{model}_{cid}_{st.session_state.review_seed}",
                label_visibility="collapsed",
            )
            if sel != label:
                st.session_state.review_labels[cid] = sel
                # Add to cumulative store
                key = f"{fov_id}_{model}_{cid}"
                st.session_state.all_labels[key] = {
                    "fov": fov_id, "model": model, "cell_id": cid,
                    "area": crop["area"], "est_diam_px": crop["est_diam_px"],
                    "label": sel,
                }
                st.rerun()

            # Shortcut buttons
            bcols = st.columns(3)
            _labels_short = [("✅","Good"), ("⚠️","Over-segmented"), ("❌","Rejected")]
            for bc, (em, lb) in zip(bcols, _labels_short):
                if bc.button(em, key=f"btn_{fov_id}_{model}_{cid}_{lb}_{st.session_state.review_seed}", use_container_width=True):
                    st.session_state.review_labels[cid] = lb
                    key = f"{fov_id}_{model}_{cid}"
                    st.session_state.all_labels[key] = {
                        "fov": fov_id, "model": model, "cell_id": cid,
                        "area": crop["area"], "est_diam_px": crop["est_diam_px"],
                        "label": lb,
                    }
                    st.rerun()

# ── Summary of current set ────────────────────────────────────────────────────
st.divider()
st.subheader("Current set summary")

label_counts = pd.Series(list(st.session_state.review_labels.values())).value_counts()
label_counts = label_counts[label_counts.index != ""]
if len(label_counts) > 0:
    import plotly.express as px
    fig = px.bar(
        x=label_counts.index, y=label_counts.values,
        labels={"x": "Label", "y": "Count"},
        color=label_counts.index,
        color_discrete_map={l: c for l, c in LABEL_COLOR.items()},
        title="Labels in current review set",
    )
    fig.update_layout(showlegend=False, height=280, margin=dict(t=40,b=20,l=20,r=20))
    st.plotly_chart(fig, use_container_width=True)

# ── Cumulative labels (all sessions) ─────────────────────────────────────────
if st.session_state.all_labels:
    st.subheader("All labels (this session)")
    all_df = pd.DataFrame(list(st.session_state.all_labels.values()))
    if len(all_df):
        labeled_df = all_df[all_df["label"] != ""]
        st.dataframe(labeled_df, use_container_width=True, hide_index=True)

        # ── Export ────────────────────────────────────────────────────────────
        st.subheader("Save labels")
        col1, col2 = st.columns(2)

        csv_bytes = labeled_df.to_csv(index=False).encode()
        col1.download_button(
            "Download manual_qc_labels.csv",
            data=csv_bytes,
            file_name="manual_qc_labels.csv",
            mime="text/csv",
        )

        save_path = st.text_input(
            "Or save to file",
            value=f"outputs/cell_segmentation_pipeline/qc/manual_qc_labels.csv",
        )
        if col2.button("Save to disk"):
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            labeled_df.to_csv(str(p), index=False)
            st.success(f"Saved {len(labeled_df)} labels → {p}")
