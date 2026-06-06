"""Registration QC — alignment mismatch verification between custom and CosMx masks.

Checks for:
  - X/Y pixel offset
  - Axis flip (flipX, flipY)
  - Axis swap (X↔Y)
  - Coordinate system inconsistencies
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # CosMx_2026/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # app/

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.image_loader import (
    load_raw_dapi, load_mask, norm_uint8,
    DEFAULT_OUTPUT_ROOTS, discover_fovs, discover_models,
)
from utils.cosmx_label_loader import (
    PER_FOV_ROOT, list_available_fovs,
    load_compartment, load_cell_labels_c,
    DEFAULT_SEG_ROOT,
)
from utils.registration_utils import (
    apply_spatial_transform,
    build_alignment_overlay,
    grid_search_best_offset,
    compute_coordinate_log,
    compute_centroid_displacement,
)

st.set_page_config(page_title="Registration QC", layout="wide")
st.title("Registration QC — Alignment Mismatch Verification")
st.caption("Diagnose coordinate offset / flip / axis-swap between custom segmentation and CosMx native labels")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Selection")

    output_root = st.selectbox(
        "Custom seg output root",
        DEFAULT_OUTPUT_ROOTS,
        format_func=lambda p: Path(p).name,
        key="reg_root",
    )
    per_fov_root = str(PER_FOV_ROOT)

    fovs = list_available_fovs(per_fov_root)
    if not fovs:
        st.error("No per_fov_decoded FOVs found.")
        st.stop()

    fov_id = st.selectbox("FOV", fovs, key="reg_fov")
    models = discover_models(fov_id, output_root)
    model  = st.selectbox("Custom model", models, key="reg_model") if models else None

    st.divider()
    st.header("Transform Controls")

    dx = st.slider("X offset (px)", -50, 50, 0, 1, key="reg_dx",
                   help="Shift custom mask right (+) or left (-)")
    dy = st.slider("Y offset (px)", -50, 50, 0, 1, key="reg_dy",
                   help="Shift custom mask down (+) or up (-)")

    st.markdown("**Flip / Swap**")
    flip_x  = st.checkbox("Flip X (left ↔ right)", key="reg_fx")
    flip_y  = st.checkbox("Flip Y (top ↔ bottom)", key="reg_fy")
    swap_xy = st.checkbox("Swap X↔Y axes (transpose)", key="reg_swap")

    st.divider()
    st.header("Display")
    alpha    = st.slider("Boundary opacity", 0.3, 1.0, 0.7, 0.05)
    dilate   = st.slider("Boundary dilation (px)", 0, 4, 1, 1)
    max_dim  = st.select_slider("Viewer resolution", [512, 768, 1024], value=768)

# ── Load data ─────────────────────────────────────────────────────────────────
raw          = load_raw_dapi(fov_id, channel=0)
custom_mask  = load_mask(fov_id, model, output_root) if model else None
compartment  = load_compartment(fov_id, per_fov_root)
cell_labels  = load_cell_labels_c(fov_id, per_fov_root)

if custom_mask is None:
    st.warning("No custom segmentation mask found. Run the pipeline first.")
    st.stop()
if compartment is None or cell_labels is None:
    st.warning("CosMx native labels (CompartmentLabels / CellLabels) not found.")
    st.stop()

cosmx_nuc_mask  = (compartment == 1).astype(np.uint8)
cosmx_cell_mask = (cell_labels  > 0).astype(np.uint8)

# Apply user transforms to custom mask
transformed = apply_spatial_transform(
    custom_mask, dx=dx, dy=dy, flip_x=flip_x, flip_y=flip_y, swap_xy=swap_xy
)

# Quick metrics at current transform
def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0

def _dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = int((a & b).sum())
    denom = int(a.sum()) + int(b.sum())
    return 2 * inter / denom if denom > 0 else 0.0

nuc_iou  = _iou(cosmx_nuc_mask,  transformed > 0)
nuc_dice = _dice(cosmx_nuc_mask, transformed > 0)
cell_iou = _iou(cosmx_cell_mask, transformed > 0)

st.markdown(f"**{fov_id}** | custom: `{model}` | transform: dx={dx} dy={dy} "
            f"flipX={flip_x} flipY={flip_y} swapXY={swap_xy}")

# Metric bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nucleus IoU",  f"{nuc_iou:.3f}")
col2.metric("Nucleus Dice", f"{nuc_dice:.3f}")
col3.metric("Cell IoU",     f"{cell_iou:.3f}")
col4.metric("Custom cells", int(transformed.max()))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overlay, tab_search, tab_scatter, tab_log = st.tabs([
    "🔵 Alignment Overlay",
    "🔍 Auto Offset Search",
    "📊 Centroid Scatter",
    "📋 Coordinate Log",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Alignment debug overlay
# ────────────────────────────────────────────────────────────────────────────
with tab_overlay:
    st.subheader("Alignment Debug Overlay (Mode E)")
    st.markdown(
        "**Blue** = CosMx nucleus boundary · "
        "**Green** = CosMx cell boundary · "
        "**Red** = custom mask boundary · "
        "**Yellow** = overlap"
    )

    if raw is None:
        st.warning("Raw DAPI not found.")
    else:
        overlay = build_alignment_overlay(
            raw, cosmx_nuc_mask, cosmx_cell_mask, transformed,
            alpha=alpha, dilate=dilate, max_dim=max_dim,
        )
        fig = px.imshow(overlay, aspect="equal")
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode="pan",
            height=700,
        )
        fig.update_traces(
            hovertemplate="x: %{x}<br>y: %{y}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Side-by-side: original vs transformed
    with st.expander("Side-by-side: original transform OFF vs current transform"):
        orig_overlay = build_alignment_overlay(
            raw, cosmx_nuc_mask, cosmx_cell_mask, custom_mask,
            alpha=alpha, dilate=dilate, max_dim=512,
        )
        xfm_overlay = build_alignment_overlay(
            raw, cosmx_nuc_mask, cosmx_cell_mask, transformed,
            alpha=alpha, dilate=dilate, max_dim=512,
        )
        c1, c2 = st.columns(2)
        c1.image(orig_overlay, caption="Original (no transform)", use_container_width=True)
        c2.image(xfm_overlay,  caption=f"Transformed dx={dx} dy={dy}", use_container_width=True)

    # Download overlay PNG
    import io as _io
    from PIL import Image as _PIL
    buf = _io.BytesIO()
    _PIL.fromarray(overlay).save(buf, format="PNG")
    st.download_button(
        "Download overlay PNG",
        data=buf.getvalue(),
        file_name=f"{fov_id}_{model}_registration_overlay_dx{dx}_dy{dy}.png",
        mime="image/png",
    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Auto offset search
# ────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Automatic Offset Search")
    st.markdown(
        "Brute-force grid search over (dx, dy) to find the offset that "
        "maximises IoU / Dice / Boundary F1. Uses 4× downsampled images for speed."
    )

    col_a, col_b, col_c = st.columns(3)
    search_range = col_a.slider("Search range (px)", 10, 50, 30, 5, key="reg_range")
    search_ds    = col_b.selectbox("Downsample factor", [4, 8, 2], key="reg_ds")
    search_metric= col_c.selectbox("Metric", ["iou", "dice", "boundary_f1"], key="reg_metric")

    apply_flip_search = st.checkbox(
        "Apply current flip/swap before searching", value=True, key="reg_apply_xfm"
    )

    if st.button("🔍 Run grid search", type="primary"):
        search_mask = apply_spatial_transform(
            custom_mask,
            flip_x=flip_x if apply_flip_search else False,
            flip_y=flip_y if apply_flip_search else False,
            swap_xy=swap_xy if apply_flip_search else False,
        )

        prog = st.progress(0.0, text="Searching…")
        with st.spinner("Running grid search…"):
            result = grid_search_best_offset(
                search_mask,
                cosmx_nuc_mask,
                range_px=search_range,
                downsample=search_ds,
                metric=search_metric,
                progress_cb=lambda f: prog.progress(f),
            )
        prog.empty()

        best_dx = result["best_dx"]
        best_dy = result["best_dy"]
        best_score = result["best_score"]
        st.session_state["reg_best"] = result

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best X offset", f"{best_dx:+d} px")
        m2.metric("Best Y offset", f"{best_dy:+d} px")
        m3.metric(f"Best {search_metric.upper()}", f"{best_score:.4f}")
        m4.metric("Baseline (no shift)", f"{_iou(cosmx_nuc_mask, custom_mask > 0):.4f}")

        if best_score > _iou(cosmx_nuc_mask, custom_mask > 0) * 1.5:
            st.success(
                f"Large improvement found! The {search_metric} improves by "
                f"{best_score / max(_iou(cosmx_nuc_mask, custom_mask > 0), 1e-6):.1f}× "
                f"at offset ({best_dx:+d}, {best_dy:+d}) px. "
                "This strongly suggests a **registration mismatch** rather than segmentation quality issues."
            )
        elif best_score > 0.4:
            st.info(f"Reasonable alignment found at ({best_dx:+d}, {best_dy:+d}) px.")
        else:
            st.warning(
                f"Best score {best_score:.3f} still low even after offset correction. "
                "May indicate axis flip/swap — try Flip X/Y or Swap axes in the sidebar."
            )

        # Heatmap of score grid
        grid = result["score_grid"]
        steps = result["steps"]
        fig_heat = px.imshow(
            grid,
            x=steps, y=steps,
            labels={"x": "dx (px)", "y": "dy (px)", "color": search_metric},
            color_continuous_scale="Viridis",
            aspect="equal",
            title=f"Grid search — {search_metric} (range ±{search_range} px)",
        )
        fig_heat.add_scatter(
            x=[best_dx], y=[best_dy], mode="markers",
            marker=dict(color="red", size=12, symbol="x"),
            name=f"Best ({best_dx:+d}, {best_dy:+d})",
        )
        fig_heat.update_layout(height=500, margin=dict(t=40, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Apply best offset button
        st.info(
            f"To apply the best offset, set **X offset = {best_dx:+d}** and "
            f"**Y offset = {best_dy:+d}** in the sidebar sliders."
        )

    elif "reg_best" in st.session_state:
        r = st.session_state["reg_best"]
        st.info(
            f"Last search result: best offset ({r['best_dx']:+d}, {r['best_dy']:+d}) px "
            f"| {r['metric']} = {r['best_score']:.4f}. Re-run to update."
        )

    # Flip/swap comparison table
    st.markdown("---")
    st.subheader("Quick flip / swap comparison")
    st.markdown("IoU of custom mask against CosMx nucleus for each transform:")

    flips = {
        "Original":         custom_mask,
        "Flip X":           np.fliplr(custom_mask),
        "Flip Y":           np.flipud(custom_mask),
        "Flip X+Y":         np.flipud(np.fliplr(custom_mask)),
        "Swap XY":          custom_mask.T,
        "Swap XY + Flip X": np.fliplr(custom_mask.T),
        "Swap XY + Flip Y": np.flipud(custom_mask.T),
    }
    rows = []
    for name, m in flips.items():
        if m.shape != cosmx_nuc_mask.shape:
            rows.append({"transform": name, "nucleus_iou": None, "cell_iou": None})
            continue
        rows.append({
            "transform": name,
            "nucleus_iou": round(_iou(cosmx_nuc_mask, m > 0), 4),
            "cell_iou":    round(_iou(cosmx_cell_mask, m > 0), 4),
        })

    flip_df = pd.DataFrame(rows).sort_values("nucleus_iou", ascending=False)
    st.dataframe(
        flip_df.style.background_gradient(subset=["nucleus_iou", "cell_iou"],
                                          cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Centroid scatter
# ────────────────────────────────────────────────────────────────────────────
with tab_scatter:
    st.subheader("Centroid Scatter — custom vs CosMx")
    st.markdown(
        "Each matched pair is connected by a line. "
        "A consistent displacement direction indicates a rigid offset."
    )

    max_dist_scatter = st.slider(
        "Max matching distance (px)", 10, 100, 50, 5, key="reg_scatter_dist"
    )
    show_arrows = st.checkbox("Show displacement vectors", value=True)

    with st.spinner("Computing centroid displacements…"):
        disp_df = compute_centroid_displacement(
            transformed, cell_labels, max_dist_px=max_dist_scatter
        )

    if len(disp_df) == 0:
        st.warning("No matched cell pairs found within the distance threshold.")
    else:
        mean_dx = disp_df["dx"].mean()
        mean_dy = disp_df["dy"].mean()
        med_dx  = disp_df["dx"].median()
        med_dy  = disp_df["dy"].median()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matched pairs", len(disp_df))
        m2.metric("Mean dx", f"{mean_dx:+.1f} px")
        m3.metric("Mean dy", f"{mean_dy:+.1f} px")
        m4.metric("Median dist", f"{disp_df['dist'].median():.1f} px")

        if abs(mean_dx) > 5 or abs(mean_dy) > 5:
            st.warning(
                f"Systematic displacement detected: mean (dx={mean_dx:+.1f}, dy={mean_dy:+.1f}) px. "
                "This is consistent with a registration offset."
            )
        else:
            st.success(f"No large systematic offset detected (mean dx={mean_dx:+.1f}, dy={mean_dy:+.1f} px).")

        # Scatter: custom centroids (red) vs CosMx centroids (blue) + lines
        fig_sc = go.Figure()

        # CosMx centroids (blue)
        fig_sc.add_scatter(
            x=disp_df["cosmx_cx"], y=disp_df["cosmx_cy"],
            mode="markers",
            marker=dict(color="royalblue", size=6, opacity=0.6),
            name="CosMx centroid",
        )
        # Custom centroids (red)
        fig_sc.add_scatter(
            x=disp_df["custom_cx"], y=disp_df["custom_cy"],
            mode="markers",
            marker=dict(color="crimson", size=6, opacity=0.6),
            name="Custom centroid",
        )

        # Displacement lines
        if show_arrows:
            sample = disp_df.sample(min(300, len(disp_df)), random_state=42)
            for _, row in sample.iterrows():
                fig_sc.add_shape(
                    type="line",
                    x0=row["cosmx_cx"], y0=row["cosmx_cy"],
                    x1=row["custom_cx"],  y1=row["custom_cy"],
                    line=dict(color="orange", width=1),
                )

        # Mean displacement arrow (large, prominent)
        H = cosmx_nuc_mask.shape[0]
        cx_mean = disp_df["cosmx_cx"].mean()
        cy_mean = disp_df["cosmx_cy"].mean()
        fig_sc.add_annotation(
            x=cx_mean + mean_dx, y=cy_mean + mean_dy,
            ax=cx_mean, ay=cy_mean,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=2,
            arrowcolor="gold", arrowwidth=3,
            text=f"mean Δ({mean_dx:+.1f}, {mean_dy:+.1f})",
            font=dict(color="gold", size=12),
        )

        fig_sc.update_layout(
            xaxis_title="X (px)", yaxis_title="Y (px)",
            yaxis_autorange="reversed",
            height=600,
            margin=dict(t=20, b=40),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # dx/dy histogram
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig_dx = px.histogram(disp_df, x="dx", nbins=40,
                                  title=f"dx distribution (mean={mean_dx:+.1f}, median={med_dx:+.1f})",
                                  color_discrete_sequence=["#e15759"])
            fig_dx.add_vline(x=0, line_dash="dash", line_color="black")
            fig_dx.add_vline(x=mean_dx, line_color="red",
                             annotation_text=f"mean={mean_dx:+.1f}")
            fig_dx.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_dx, use_container_width=True)

        with col_h2:
            fig_dy = px.histogram(disp_df, x="dy", nbins=40,
                                  title=f"dy distribution (mean={mean_dy:+.1f}, median={med_dy:+.1f})",
                                  color_discrete_sequence=["#4e79a7"])
            fig_dy.add_vline(x=0, line_dash="dash", line_color="black")
            fig_dy.add_vline(x=mean_dy, line_color="royalblue",
                             annotation_text=f"mean={mean_dy:+.1f}")
            fig_dy.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_dy, use_container_width=True)

        # Download CSV
        csv = disp_df.to_csv(index=False).encode()
        st.download_button(
            "Download centroid displacement CSV",
            data=csv,
            file_name=f"{fov_id}_{model}_centroid_displacement.csv",
            mime="text/csv",
        )

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — Coordinate system log
# ────────────────────────────────────────────────────────────────────────────
with tab_log:
    st.subheader("Coordinate System Log")
    st.markdown(
        "Checks shape consistency, index conventions, and centroid statistics "
        "to rule out coordinate-system confusion."
    )

    log = compute_coordinate_log(
        fov_id, transformed, cosmx_nuc_mask, cell_labels,
        dapi_shape=raw.shape if raw is not None else None,
    )

    # Shape check
    shape_ok = log.get("shapes_match", False)
    if shape_ok:
        st.success("✅ All shapes match.")
    else:
        st.error(f"⚠️ Shape mismatch detected! custom={log['custom_mask_shape']} "
                 f"vs CosMx nucleus={log['cosmx_nuc_shape']}")

    # Main log table
    scalar_rows = [
        ("FOV ID",               log.get("fov_id", "")),
        ("DAPI shape",           log.get("dapi_shape", "")),
        ("Custom mask shape",    log.get("custom_mask_shape", "")),
        ("CosMx nuc shape",      log.get("cosmx_nuc_shape", "")),
        ("CosMx cell shape",     log.get("cosmx_cell_shape", "")),
        ("Shapes match",         str(shape_ok)),
        ("Custom n_cells",       log.get("custom_n_cells", "")),
        ("CosMx n_cells",        log.get("cosmx_n_cells", "")),
        ("Custom mean X",        f"{log.get('custom_mean_x', float('nan')):.1f}"),
        ("Custom mean Y",        f"{log.get('custom_mean_y', float('nan')):.1f}"),
        ("Custom X range",       log.get("custom_x_range", "")),
        ("Custom Y range",       log.get("custom_y_range", "")),
        ("CosMx mean X",         f"{log.get('cosmx_mean_x', float('nan')):.1f}"),
        ("CosMx mean Y",         f"{log.get('cosmx_mean_y', float('nan')):.1f}"),
        ("Index convention",     log.get("index_convention", "")),
        ("Centroid convention",  log.get("centroid_convention", "")),
        ("Zero-indexed",         str(log.get("zero_indexed", True))),
    ]

    st.dataframe(
        pd.DataFrame(scalar_rows, columns=["Parameter", "Value"]),
        use_container_width=True, hide_index=True,
    )

    # Displacement stats from centroid scatter (reuse if available)
    if len(disp_df) > 0:
        st.markdown("**Centroid displacement statistics (matched pairs):**")
        disp_stats = pd.DataFrame([{
            "mean dx":   f"{disp_df['dx'].mean():+.2f} px",
            "mean dy":   f"{disp_df['dy'].mean():+.2f} px",
            "median dx": f"{disp_df['dx'].median():+.2f} px",
            "median dy": f"{disp_df['dy'].median():+.2f} px",
            "std dx":    f"{disp_df['dx'].std():.2f} px",
            "std dy":    f"{disp_df['dy'].std():.2f} px",
            "max dist":  f"{disp_df['dist'].max():.1f} px",
            "n matched": len(disp_df),
        }])
        st.dataframe(disp_stats.T.rename(columns={0: "Value"}),
                     use_container_width=True)

    # Interpretation guide
    st.markdown("---")
    st.markdown("""
**Interpretation guide**

| Finding | Most likely cause |
|---|---|
| Large systematic dx/dy (>10 px) | Registration offset between DAPI and label coordinate systems |
| IoU improves dramatically after shift | Pure registration mismatch — not segmentation quality |
| Best flip is `Flip Y` or `Swap XY` | Coordinate axis convention mismatch |
| Low IoU even after all corrections | Genuine segmentation quality difference |
| Shape mismatch | Wrong FOV file loaded, or coordinate system scaling issue |

**Coordinate system notes:**
- CosMx `CompartmentLabels` / `CellLabels` are in **local FOV pixel coordinates** (0-indexed, row=y, col=x)
- The custom segmentation mask should use the same convention
- `image[row, col]` = `image[y, x]` — **not** `image[x, y]`
- CosMx DAPI TIF and label TIFs should be perfectly registered to each other
""")

    # Raw log download
    import json as _json
    log_clean = {k: str(v) for k, v in log.items()}
    st.download_button(
        "Download coordinate log (JSON)",
        data=_json.dumps(log_clean, indent=2),
        file_name=f"{fov_id}_{model}_coordinate_log.json",
        mime="application/json",
    )
