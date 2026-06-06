"""Cell Inspector — click/select a cell to view crop, metadata, neighbours."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.cell_cropper import find_cell_at_xy, get_cell_crop
from utils.image_loader import (
    DEFAULT_OUTPUT_ROOTS, discover_fovs, discover_models,
    get_fov_meta, load_cell_table, load_enhanced, load_mask,
    load_raw_dapi, norm_uint8, downsample_img, downsample_mask,
)
from utils.overlay_utils import compose_overlay, make_overlay_fig

st.set_page_config(page_title="Cell Inspector", layout="wide")
st.title("Cell Inspector")
st.caption("Inspect individual cells: zoom into the overlay and select a cell by ID or table click")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    output_root = st.selectbox(
        "Output root", DEFAULT_OUTPUT_ROOTS,
        format_func=lambda p: Path(p).name,
        key="insp_root",
    )
    fovs   = discover_fovs(output_root)
    if not fovs:
        st.warning("No masks found.")
        st.stop()

    fov_id = st.selectbox("FOV", fovs, key="insp_fov")
    models = discover_models(fov_id, output_root)
    model  = st.selectbox("Model", models, key="insp_model") if models else None

    alpha   = st.slider("Overlay opacity", 0.0, 1.0, 0.4, 0.05)
    padding = st.slider("Crop padding (px)", 5, 80, 25, 5)
    max_dim = st.select_slider("Viewer resolution", [512, 768, 1024], value=768)

# ── Load data ─────────────────────────────────────────────────────────────────
raw      = load_raw_dapi(fov_id)
enhanced = load_enhanced(fov_id, output_root)
mask     = load_mask(fov_id, model, output_root) if model else None
ct       = load_cell_table(fov_id, model, output_root) if model else None
ref      = enhanced if enhanced is not None else raw

if mask is None:
    st.warning("No mask found. Run the segmentation pipeline first.")
    st.stop()

n_cells = int(mask.max())
meta    = get_fov_meta(fov_id)
st.markdown(
    f"**{fov_id}** | {meta.get('condition','?')} | {meta.get('region','?')} | "
    f"**{n_cells}** cells | model: `{model}`"
)

# ── Layout: main viewer | right panel ─────────────────────────────────────────
col_main, col_right = st.columns([3, 1])

with col_main:
    st.subheader("Overlay viewer  (zoom / pan with toolbar)")
    fig = make_overlay_fig(ref, mask, alpha=alpha, title="", max_dim=max_dim)
    # Enable click events
    event = st.plotly_chart(
        fig, use_container_width=True,
        key="inspector_chart",
        on_select="rerun",
        selection_mode=["points"],
    )

# ── Click-to-inspect ──────────────────────────────────────────────────────────
selected_cell_id: int | None = None

# Try to read clicked point from plotly event
if event and hasattr(event, "selection") and event.selection.points:
    pt  = event.selection.points[0]
    # Plotly imshow returns x (col) and y (row) in display coordinates.
    # Map back to original image coordinates.
    H, W = ref.shape[:2]
    scale = max_dim / max(H, W)
    orig_x = int(pt["x"] / scale)
    orig_y = int(pt["y"] / scale)
    cell_at = find_cell_at_xy(mask, orig_x, orig_y)
    if cell_at is not None:
        selected_cell_id = cell_at

with col_right:
    st.subheader("Select cell")

    # Manual cell ID selector
    manual_id = st.number_input(
        "Cell ID (1 – N)", min_value=1, max_value=n_cells, value=1, step=1,
        key="insp_cell_id",
    )
    if selected_cell_id is None:
        selected_cell_id = manual_id
    else:
        st.info(f"Clicked cell: **{selected_cell_id}**")

    # Extract crop at full resolution
    if ref is not None and selected_cell_id is not None:
        crop = get_cell_crop(ref, mask, selected_cell_id, padding=padding)
        if crop:
            st.image(crop["overlay_crop"], use_container_width=True, caption="Cell crop (overlay)")

            # Metadata card
            st.markdown("**Cell metadata**")
            st.markdown(f"""
| Field | Value |
|---|---|
| Cell ID | `{crop['cell_id']}` |
| Area | `{crop['area']} px²` |
| Centroid X | `{crop['centroid_x']:.1f}` |
| Centroid Y | `{crop['centroid_y']:.1f}` |
| Est. diameter | `{crop['est_diam_px']:.1f} px` |
| Bbox (y0,x0,y1,x1) | `{crop['bbox']}` |
""")
        else:
            st.warning(f"Cell {selected_cell_id} not found in mask.")

# ── Cell table (searchable / selectable) ─────────────────────────────────────
if ct is not None and len(ct) > 0:
    st.subheader("Cell table (select a row to inspect)")

    display_cols = [c for c in ["label","area","centroid_y","centroid_x","est_diam_px","eccentricity","solidity"] if c in ct.columns]
    ct_display = ct[display_cols].copy()
    ct_display.index = ct_display["label"].astype(int) if "label" in ct_display.columns else ct_display.index

    event2 = st.dataframe(
        ct_display,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="insp_table",
    )
    if event2 and event2.selection.rows:
        row_idx = event2.selection.rows[0]
        table_cell_id = int(ct_display.iloc[row_idx]["label"])
        st.info(f"Table selected: Cell {table_cell_id}")
        crop2 = get_cell_crop(ref, mask, table_cell_id, padding=padding)
        if crop2:
            c1, c2 = st.columns(2)
            c1.image(crop2["raw_crop"],     caption="Raw crop", use_container_width=True)
            c2.image(crop2["overlay_crop"], caption="Overlay",  use_container_width=True)
