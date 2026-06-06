"""Model Comparison — side-by-side segmentation comparison with stats."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.image_loader import (
    DEFAULT_OUTPUT_ROOTS, discover_fovs, discover_models,
    get_fov_meta, load_cell_table, load_enhanced, load_mask,
    load_raw_dapi, norm_uint8, downsample_img, downsample_mask,
)
from utils.overlay_utils import compose_overlay, colorize_mask
from utils.qc_utils import compute_stats, model_comparison_bar

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")
st.caption("Compare segmentation models side-by-side on the same FOV")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    output_roots = st.multiselect(
        "Output roots to search",
        DEFAULT_OUTPUT_ROOTS,
        default=DEFAULT_OUTPUT_ROOTS,
        format_func=lambda p: Path(p).name,
    )
    if not output_roots:
        output_roots = DEFAULT_OUTPUT_ROOTS

    # Collect all FOVs across selected roots
    all_fovs: list[str] = []
    for root in output_roots:
        for fov in discover_fovs(root):
            if fov not in all_fovs:
                all_fovs.append(fov)
    all_fovs = sorted(all_fovs)

    if not all_fovs:
        st.warning("No masks found in selected output roots.")
        st.stop()

    fov_id = st.selectbox("FOV", all_fovs)
    alpha  = st.slider("Overlay opacity", 0.0, 1.0, 0.4, 0.05)
    max_dim = st.select_slider("Panel resolution", [384, 512, 768], value=512)
    channel = st.number_input("Raw channel (0=DAPI)", 0, 4, 0)

# ── Collect available model/root combos ───────────────────────────────────────
available: list[dict] = []
for root in output_roots:
    for model in discover_models(fov_id, root):
        available.append({"root": root, "model": model, "label": f"{Path(root).name} / {model}"})

if not available:
    st.info("No segmentation results found for this FOV. Run the pipeline first.")
    st.stop()

selected_labels = st.multiselect(
    "Select models to compare",
    [a["label"] for a in available],
    default=[a["label"] for a in available],
)
selected = [a for a in available if a["label"] in selected_labels]

if not selected:
    st.warning("Select at least one model.")
    st.stop()

# ── Load reference image ──────────────────────────────────────────────────────
raw      = load_raw_dapi(fov_id, channel)
enhanced = load_enhanced(fov_id, output_roots[0])
ref      = enhanced if enhanced is not None else raw

meta = get_fov_meta(fov_id)
st.markdown(
    f"**{fov_id}** — {meta.get('condition','?')} | {meta.get('region','?')} | "
    f"age {meta.get('age','?')}"
)

# ── Build comparison panels ───────────────────────────────────────────────────
results = []
for sel in selected:
    mask = load_mask(fov_id, sel["model"], sel["root"])
    ct   = load_cell_table(fov_id, sel["model"], sel["root"])
    if mask is None:
        continue

    ref_ds   = downsample_img(norm_uint8(ref), max_dim)
    mask_ds  = downsample_mask(mask, max_dim)
    overlay  = compose_overlay(ref_ds, mask_ds, alpha)
    stats    = compute_stats(ct) if ct is not None else {}

    results.append({
        "label":    sel["label"],
        "model":    sel["model"],
        "overlay":  overlay,
        "mask":     mask,
        "n_cells":  int(mask.max()),
        "stats":    stats,
        "ct":       ct,
    })

if not results:
    st.error("Could not load any masks.")
    st.stop()

# ── Grid display ──────────────────────────────────────────────────────────────
ncols = min(len(results), 4)
cols  = st.columns(ncols)

for i, res in enumerate(results):
    with cols[i % ncols]:
        n = res["n_cells"]
        s = res["stats"]
        med_d = f"{s.get('median_diam_px', 0):.1f}" if s else "—"

        st.markdown(f"**{res['model']}**")
        st.image(res["overlay"], use_container_width=True)
        st.caption(
            f"Cells: **{n}** | Median diam: {med_d} px | "
            f"Median area: {s.get('median_area_px', '—'):.0f} px²" if s else f"Cells: {n}"
        )

# ── Stats table ───────────────────────────────────────────────────────────────
st.subheader("Summary statistics")
rows = []
for res in results:
    row = {"model": res["label"], "n_cells": res["n_cells"]}
    row.update(res["stats"])
    rows.append(row)

summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Bar chart ─────────────────────────────────────────────────────────────────
bar_data = [{"label": r["label"], "n_cells": r["n_cells"]} for r in results]
st.plotly_chart(model_comparison_bar(bar_data), use_container_width=True)

# ── Download comparison panel ─────────────────────────────────────────────────
import io, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _build_panel_png(results: list[dict]) -> bytes:
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), dpi=100)
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        ax.imshow(res["overlay"])
        ax.set_title(f"{res['model']}\n{res['n_cells']} cells", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

if st.button("Download comparison PNG"):
    png = _build_panel_png(results)
    st.download_button("Save PNG", data=png, file_name=f"{fov_id}_model_comparison.png", mime="image/png")
