"""FOV Viewer — browse images with overlay, zoom/pan, and layer controls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # CosMx_2026/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # app/

import streamlit as st

from utils.image_loader import (
    DEFAULT_OUTPUT_ROOTS, discover_fovs, discover_models,
    get_fov_meta, load_enhanced, load_expanded_mask, load_mask,
    load_raw_dapi, norm_uint8, downsample_img, downsample_mask,
)
from utils.overlay_utils import compose_overlay, make_overlay_fig, make_single_fig

st.set_page_config(page_title="FOV Viewer", layout="wide")
st.title("FOV Viewer")
st.caption("Browse segmentation results with zoom / pan overlay (scroll to zoom, drag to pan)")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    output_root = st.selectbox(
        "Output root",
        DEFAULT_OUTPUT_ROOTS,
        format_func=lambda p: Path(p).name,
        key="fov_output_root",
    )

    fovs = discover_fovs(output_root)
    if not fovs:
        st.warning("No masks found in selected output root.")
        st.stop()

    fov_id = st.selectbox("FOV", fovs)
    models = discover_models(fov_id, output_root)
    model  = st.selectbox("Model", models) if models else None

    st.divider()

    show_raw      = st.checkbox("Raw DAPI",     value=True)
    show_enhanced = st.checkbox("Enhanced",     value=True)
    show_overlay  = st.checkbox("Overlay",      value=True)
    show_mask     = st.checkbox("Mask only",    value=False)
    show_expanded = st.checkbox("Expanded mask overlay", value=False)

    st.divider()
    alpha    = st.slider("Overlay opacity", 0.0, 1.0, 0.40, 0.05)
    max_dim  = st.select_slider("Display resolution", [512, 768, 1024, 1536], value=1024)
    channel  = st.selectbox("Raw channel", [0, 1, 2, 3, 4],
                             format_func=lambda k: f"Ch{k}: {['DAPI','PanCK','G','Membrane','CD45'][k]}")

# ── Load data ─────────────────────────────────────────────────────────────────
meta = get_fov_meta(fov_id)
cond = meta.get("condition", "?")
reg  = meta.get("region", "?")
age  = meta.get("age", "?")

st.markdown(f"**{fov_id}** — condition: `{cond}` | region: `{reg}` | age: `{age}` | model: `{model}`")

raw      = load_raw_dapi(fov_id, channel)
enhanced = load_enhanced(fov_id, output_root)
mask     = load_mask(fov_id, model, output_root)      if model else None
exp_mask = load_expanded_mask(fov_id, model, output_root) if model else None

# Choose reference image for overlays
ref = enhanced if enhanced is not None else raw

# ── Active layers → columns ───────────────────────────────────────────────────
panels = []
if show_raw and raw is not None:
    panels.append(("Raw DAPI", make_single_fig(raw, title=f"Raw DAPI — {fov_id}", max_dim=max_dim)))
if show_enhanced and enhanced is not None:
    panels.append(("Enhanced", make_single_fig(enhanced, title=f"Enhanced — {fov_id}", max_dim=max_dim)))
if show_overlay and ref is not None and mask is not None:
    panels.append(("Overlay", make_overlay_fig(ref, mask, alpha=alpha, title=f"Overlay (α={alpha:.0%}) — {model}", max_dim=max_dim)))
if show_mask and mask is not None:
    from utils.overlay_utils import colorize_mask, downsample_mask as _ds
    col_mask = colorize_mask(_ds(mask, max_dim))
    import plotly.express as px
    fig = px.imshow(col_mask, aspect="equal")
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), title="Mask colours", dragmode="pan")
    panels.append(("Mask", fig))
if show_expanded and ref is not None and exp_mask is not None:
    panels.append(("Expanded overlay", make_overlay_fig(ref, exp_mask, alpha=alpha, title="Expanded mask", max_dim=max_dim)))

if not panels:
    st.info("Enable at least one layer in the sidebar to display images.")
else:
    n = len(panels)
    cols = st.columns(n)
    for col, (label, fig) in zip(cols, panels):
        with col:
            st.subheader(label)
            st.plotly_chart(fig, use_container_width=True)

# ── Cell count info ───────────────────────────────────────────────────────────
if mask is not None:
    n_nuc = int(mask.max())
    n_exp = int(exp_mask.max()) if exp_mask is not None else "—"
    st.markdown(f"**Nuclear cells:** {n_nuc} | **Expanded cells:** {n_exp}")

st.caption("Tip: use Plotly toolbar (top-right of chart) to zoom, pan, reset, or download PNG.")
