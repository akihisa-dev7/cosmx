"""Cell Type Validation — RNA/Protein cell type QC on custom segmentation.

Loads cosmx_molecular_analysis outputs and overlays cell type annotations
on the segmentation mask for manual validation.

Usage:
    streamlit run cell_segmentation_pipeline/app/streamlit_app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # CosMx_2026/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # app/

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.cell_cropper import find_cell_at_xy, get_cell_crop
from utils.celltype_loader import (
    CELL_TYPE_PALETTE,
    CELL_TYPE_ORDER,
    MOL_ROOT,
    build_cell_type_colored_mask,
    build_expression_mask,
    get_available_fovs_mol,
    get_cell_info,
    get_fov_cells,
    label_to_cell_id,
    load_anndata,
    load_assignment_summary,
    load_cell_type_table,
    load_composition,
    load_joint_fov_summary,
    save_manual_review,
)
from utils.image_loader import (
    load_raw_dapi,
    load_enhanced,
    load_expanded_mask,
    norm_uint8,
    downsample_img,
    downsample_mask,
    get_fov_meta,
)
from utils.overlay_utils import compose_overlay

st.set_page_config(page_title="Cell Type Validation", layout="wide")
st.title("Cell Type Validation")
st.caption(
    "RNA/Protein-based cell type QC — overlays cosmx_molecular_analysis outputs "
    "on custom segmentation masks"
)

# ── Paths ──────────────────────────────────────────────────────────────────────
mol_root = str(MOL_ROOT)
seg_root = str(MOL_ROOT.parents[0] / "pilot_4fov")  # outputs/pilot_4fov


# ── Check data availability ────────────────────────────────────────────────────
ct_table = load_cell_type_table(mol_root)
if ct_table is None:
    st.error(
        "cell_type_table.csv not found. "
        f"Expected at: {mol_root}/cell_type_table.csv\n\n"
        "Run the molecular analysis pipeline first:\n"
        "```bash\n"
        "python cosmx_molecular_analysis/scripts/run_full_pipeline.py "
        "--config cosmx_molecular_analysis/config/default_config.yaml\n"
        "```"
    )
    st.stop()

# Session state for manual overrides
if "manual_overrides" not in st.session_state:
    st.session_state.manual_overrides = {}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    fov_names = get_available_fovs_mol(mol_root)
    if not fov_names:
        st.error("No FOVs found in cell_type_table.")
        st.stop()

    fov_name = st.selectbox("FOV", fov_names, key="ctv_fov")

    st.divider()
    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05, key="ctv_alpha")
    max_dim = st.select_slider("Display resolution", [512, 768, 1024, 1536], value=1024, key="ctv_dim")

    st.divider()
    st.markdown("**Cell type filter**")
    all_types = sorted(ct_table["predicted_cell_type"].unique()) if "predicted_cell_type" in ct_table.columns else []
    selected_types = st.multiselect(
        "Show cell types",
        all_types,
        default=all_types,
        key="ctv_types",
    )

    st.divider()
    st.markdown("**Manual overrides**")
    n_manual = len(st.session_state.manual_overrides)
    st.caption(f"{n_manual} cell(s) manually reviewed")
    if n_manual > 0 and st.button("Clear all overrides"):
        st.session_state.manual_overrides = {}
        st.rerun()

# ── Load mask and images ───────────────────────────────────────────────────────
# Try pilot_4fov expanded masks
import glob as _glob
mask_files = _glob.glob(str(Path(seg_root) / "expanded_masks" / f"{fov_name}_*expanded*.tif"))
mask = None
if mask_files:
    import tifffile as _tiff
    mask = _tiff.imread(mask_files[0]).astype(np.int32)

raw = load_raw_dapi(fov_name, 0)
enhanced = load_enhanced(fov_name, seg_root)
ref = enhanced if enhanced is not None else raw

fov_cells = get_fov_cells(ct_table, fov_name)
adata = load_anndata(mol_root)

if mask is None:
    st.warning(f"No expanded mask found for {fov_name} in {seg_root}/expanded_masks/")

meta = get_fov_meta(fov_name)

st.markdown(
    f"**{fov_name}** | condition: `{meta.get('condition','?')}` | "
    f"region: `{meta.get('region','?')}` | "
    f"cells (annotated): **{len(fov_cells)}**"
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_map, tab_expr, tab_insp, tab_qc, tab_review = st.tabs([
    "🎨 Cell Type Map",
    "🔥 Marker Expression",
    "🔍 Cell Inspector",
    "📊 QC Dashboard",
    "✏️ Manual Review",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Cell Type Map
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("Cell Type Color Map")

    if mask is None:
        st.info("Load a segmentation mask to display the cell type map.")
    else:
        ct_rgb, _lbl_ct = build_cell_type_colored_mask(
            mask, ct_table, fov_name,
            manual_overrides=st.session_state.manual_overrides,
        )

        # Filter: zero-out non-selected types
        if selected_types and len(selected_types) < len(all_types):
            keep_types = set(selected_types)
            # Re-build only with selected types
            ct_rgb_filtered = np.zeros_like(ct_rgb)
            for _, row in fov_cells.iterrows():
                cid = str(row.get("cell_id", ""))
                ct = st.session_state.manual_overrides.get(
                    cid, str(row.get("predicted_cell_type", "Unknown"))
                )
                if ct not in keep_types:
                    continue
                from utils.celltype_loader import cell_id_to_label, _hex_to_rgb, CELL_TYPE_PALETTE
                lbl = cell_id_to_label(cid)
                if lbl is not None:
                    color = CELL_TYPE_PALETTE.get(ct, "#888888")
                    r, g, b = _hex_to_rgb(color)
                    pix = mask == lbl
                    ct_rgb_filtered[pix] = [r, g, b]
            ct_rgb = ct_rgb_filtered

        col_img, col_legend = st.columns([4, 1])

        with col_img:
            # Compose overlay: DAPI (gray) + cell type colors
            if ref is not None:
                gray_ds = downsample_img(norm_uint8(ref), max_dim)
                h0, w0 = ref.shape[:2]
                scale = max_dim / max(h0, w0)
                new_h = int(h0 * scale)
                new_w = int(w0 * scale)
                from PIL import Image as _PIL
                ct_pil = _PIL.fromarray(ct_rgb).resize((new_w, new_h), _PIL.NEAREST)
                ct_ds = np.array(ct_pil)

                # Blend
                base = np.stack([gray_ds, gray_ds, gray_ds], axis=-1).astype(np.float32)
                fg = ct_ds.any(axis=-1)
                blended = base.copy()
                blended[fg] = (1 - alpha) * base[fg] + alpha * ct_ds[fg].astype(np.float32)
                blended = np.clip(blended, 0, 255).astype(np.uint8)

                fig_map = px.imshow(blended, aspect="equal")
                fig_map.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    dragmode="pan",
                )
                st.plotly_chart(fig_map, use_container_width=True, key="ctv_map_fig")
            else:
                # No background image — show mask directly
                ct_ds_sm = downsample_img(ct_rgb, max_dim)
                fig_map = px.imshow(ct_ds_sm, aspect="equal")
                fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode="pan")
                st.plotly_chart(fig_map, use_container_width=True, key="ctv_map_fig_nomask")

        with col_legend:
            st.markdown("**Cell types**")
            counts = fov_cells["predicted_cell_type"].value_counts() if "predicted_cell_type" in fov_cells.columns else {}
            for ct_name in CELL_TYPE_ORDER + [t for t in all_types if t not in CELL_TYPE_ORDER]:
                if ct_name not in all_types:
                    continue
                color = CELL_TYPE_PALETTE.get(ct_name, "#888888")
                cnt = int(counts.get(ct_name, 0))
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
                    f'<div style="width:16px;height:16px;background:{color};border-radius:3px;flex-shrink:0"></div>'
                    f'<span style="font-size:13px">{ct_name} ({cnt})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Manual override indicator
            n_ov = len([c for c in st.session_state.manual_overrides
                        if c.startswith(fov_name + "_")])
            if n_ov > 0:
                st.divider()
                st.markdown(f'<span style="color:#FF6B35">✏️ {n_ov} manually overridden</span>',
                            unsafe_allow_html=True)

        # Export static PNG
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save cell_type_map.png", key="save_ctmap"):
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                save_path = Path(mol_root) / "cell_type_map.png"
                fig_s, ax_s = plt.subplots(figsize=(8, 8))
                if ref is not None:
                    ax_s.imshow(norm_uint8(ref), cmap="gray")
                    ax_s.imshow(ct_rgb, alpha=0.5)
                else:
                    ax_s.imshow(ct_rgb)
                ax_s.axis("off")
                ax_s.set_title(f"Cell type map — {fov_name}")
                fig_s.tight_layout()
                fig_s.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig_s)
                st.success(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Marker Expression
# ══════════════════════════════════════════════════════════════════════════════
with tab_expr:
    st.subheader("Marker Expression Overlay")

    if adata is None:
        st.warning("rna_annotated.h5ad not found. Marker expression overlay unavailable.")
    elif mask is None:
        st.warning("No mask loaded. Expression overlay unavailable.")
    else:
        c_sel, c_plot = st.columns([1, 3])

        with c_sel:
            expr_type = st.radio("Expression source", ["RNA gene", "Obs column"], key="ctv_etype")

            if expr_type == "RNA gene":
                # Common marker genes first
                marker_genes_common = [
                    "SNAP25", "MAP2", "GFAP", "AQP4", "AIF1", "C1QA", "TYROBP",
                    "MBP", "PLP1", "OLIG2", "PDGFRA", "CLDN5", "PECAM1", "ACTA2", "PDGFRB",
                ]
                available_markers = [g for g in marker_genes_common if g in adata.var_names]
                other_genes = [g for g in adata.var_names if g not in marker_genes_common]
                all_gene_options = available_markers + sorted(other_genes)
                gene = st.selectbox(
                    "Gene",
                    all_gene_options,
                    key="ctv_gene",
                )
                use_obs = False
                feature_label = gene
            else:
                num_obs_cols = [c for c in adata.obs.columns if adata.obs[c].dtype in [float, np.float64, np.float32, np.int64, np.int32, int]]
                obs_col = st.selectbox("Obs column", num_obs_cols, key="ctv_obscol")
                gene = obs_col
                use_obs = True
                feature_label = obs_col

            colorscale = st.selectbox(
                "Color scale",
                ["Plasma", "Viridis", "Hot", "RdBu_r", "Turbo"],
                key="ctv_cscale",
            )

        with c_plot:
            expr_mask = build_expression_mask(mask, adata, gene, fov_name, use_obs=use_obs)
            if expr_mask is None:
                st.info(f"'{gene}' not found in AnnData for {fov_name}.")
            else:
                vmax = float(np.percentile(expr_mask[expr_mask > 0], 95)) if (expr_mask > 0).any() else 1.0
                vmin = 0.0

                if ref is not None:
                    gray_ds = downsample_img(norm_uint8(ref), max_dim)
                    h0, w0 = ref.shape[:2]
                    scale = max_dim / max(h0, w0)
                    new_h, new_w = int(h0 * scale), int(w0 * scale)
                    from PIL import Image as _PIL
                    expr_ds = np.array(
                        _PIL.fromarray(expr_mask).resize((new_w, new_h), _PIL.NEAREST)
                    )
                    fig_expr = go.Figure()
                    fig_expr.add_trace(go.Image(z=np.stack([gray_ds]*3, axis=-1)))
                    fig_expr.add_trace(go.Heatmap(
                        z=expr_ds,
                        colorscale=colorscale,
                        zmin=vmin,
                        zmax=vmax,
                        opacity=0.65,
                        showscale=True,
                        colorbar=dict(title=feature_label, len=0.6),
                    ))
                else:
                    from PIL import Image as _PIL
                    expr_ds = np.array(
                        _PIL.fromarray(expr_mask).resize((max_dim, max_dim), _PIL.NEAREST)
                    )
                    fig_expr = go.Figure()
                    fig_expr.add_trace(go.Heatmap(
                        z=expr_ds,
                        colorscale=colorscale,
                        zmin=vmin,
                        zmax=vmax,
                        showscale=True,
                        colorbar=dict(title=feature_label),
                    ))

                fig_expr.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=f"{feature_label} expression — {fov_name}",
                    dragmode="pan",
                    yaxis=dict(scaleanchor="x"),
                )
                st.plotly_chart(fig_expr, use_container_width=True, key="ctv_expr_fig")

                # Stats
                expressed = expr_mask[expr_mask > 0]
                if len(expressed) > 0:
                    st.caption(
                        f"Non-zero pixels: {len(expressed):,} | "
                        f"Mean: {expressed.mean():.3f} | "
                        f"95th pct: {np.percentile(expressed, 95):.3f}"
                    )

        # Protein panel from joint_fov_summary
        with st.expander("FOV-level protein summary (from joint_fov_summary.csv)"):
            joint = load_joint_fov_summary(mol_root)
            if joint is None:
                st.info("joint_fov_summary.csv not found.")
            else:
                prot_cols = [c for c in joint.columns if c not in {
                    "fov_name", "n_cells_rna", "n_cells_protein", "fov",
                    "condition", "region"
                } and not c.startswith(("frac_", "score_"))]
                if prot_cols:
                    fov_row = joint[joint["fov_name"] == fov_name]
                    if not fov_row.empty:
                        prot_vals = fov_row[prot_cols].iloc[0].sort_values(ascending=False)
                        fig_prot = px.bar(
                            x=prot_vals.index,
                            y=prot_vals.values,
                            labels={"x": "Protein", "y": "Mean intensity"},
                            title=f"Protein panel — {fov_name}",
                            color=prot_vals.values,
                            color_continuous_scale="Teal",
                        )
                        fig_prot.update_layout(
                            height=320,
                            margin=dict(t=40, b=60, l=20, r=20),
                            xaxis_tickangle=-45,
                            coloraxis_showscale=False,
                        )
                        st.plotly_chart(fig_prot, use_container_width=True)

        # Save expression overlay
        if adata is not None and mask is not None:
            if st.button("💾 Save marker_expression_overlay.png", key="save_expr"):
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm

                gene_save = st.session_state.get("ctv_gene", list(adata.var_names)[0] if expr_type == "RNA gene" else "")
                expr_save = build_expression_mask(mask, adata, gene_save, fov_name, use_obs=False)
                if expr_save is not None:
                    save_path = Path(mol_root) / "marker_expression_overlay.png"
                    vmax_s = float(np.percentile(expr_save[expr_save > 0], 95)) if (expr_save > 0).any() else 1.0
                    fig_s, ax_s = plt.subplots(figsize=(8, 8))
                    if ref is not None:
                        ax_s.imshow(norm_uint8(ref), cmap="gray")
                    im = ax_s.imshow(expr_save, cmap="plasma", alpha=0.7, vmin=0, vmax=vmax_s)
                    plt.colorbar(im, ax=ax_s, label=gene_save, fraction=0.03)
                    ax_s.axis("off")
                    ax_s.set_title(f"{gene_save} — {fov_name}")
                    fig_s.tight_layout()
                    fig_s.savefig(save_path, dpi=150, bbox_inches="tight")
                    plt.close(fig_s)
                    st.success(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Cell Inspector
# ══════════════════════════════════════════════════════════════════════════════
with tab_insp:
    st.subheader("Cell Inspector")
    st.caption("Click the map or enter a cell ID to inspect RNA markers, cell type scores, and morphology")

    if mask is None:
        st.info("No mask loaded.")
    else:
        col_view, col_info = st.columns([3, 2])

        with col_view:
            # Show cell type colored mask with click detection
            if ref is not None:
                gray_ds = downsample_img(norm_uint8(ref), max_dim)
                h0, w0 = ref.shape[:2]
                scale = max_dim / max(h0, w0)
                new_h, new_w = int(h0 * scale), int(w0 * scale)
                ct_rgb_insp, _ = build_cell_type_colored_mask(
                    mask, ct_table, fov_name,
                    manual_overrides=st.session_state.manual_overrides,
                )
                from PIL import Image as _PIL
                ct_pil = _PIL.fromarray(ct_rgb_insp).resize((new_w, new_h), _PIL.NEAREST)
                ct_ds = np.array(ct_pil)
                base = np.stack([gray_ds]*3, axis=-1).astype(np.float32)
                fg = ct_ds.any(axis=-1)
                blended = base.copy()
                blended[fg] = (1 - alpha) * base[fg] + alpha * ct_ds[fg].astype(np.float32)
                blended = np.clip(blended, 0, 255).astype(np.uint8)

                fig_insp = px.imshow(blended, aspect="equal")
                fig_insp.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    dragmode="select",
                    title="Click a cell to inspect",
                )
                click_ev = st.plotly_chart(
                    fig_insp,
                    use_container_width=True,
                    key="ctv_insp_map",
                    on_select="rerun",
                    selection_mode=["points"],
                )
            else:
                click_ev = None
                st.info("No reference image available for display.")

        with col_info:
            # Resolve cell from click or manual input
            selected_label: int | None = None

            if click_ev and hasattr(click_ev, "selection") and click_ev.selection.points:
                pt = click_ev.selection.points[0]
                H, W = ref.shape[:2] if ref is not None else mask.shape
                sc = max_dim / max(H, W)
                ox, oy = int(pt["x"] / sc), int(pt["y"] / sc)
                selected_label = find_cell_at_xy(mask, ox, oy)

            n_cells_fov = int(fov_cells["cell_id"].str.rsplit("_", n=1).str[-1].astype(int).max()) if "cell_id" in fov_cells.columns and len(fov_cells) > 0 else int(mask.max())

            manual_label = st.number_input(
                "Cell label (1–N)",
                min_value=1,
                max_value=max(int(mask.max()), 1),
                value=selected_label if selected_label else 1,
                step=1,
                key="ctv_insp_id",
            )

            if selected_label is None:
                selected_label = manual_label
            else:
                st.success(f"Clicked mask label: **{selected_label}**")

            cell_id = label_to_cell_id(fov_name, selected_label)
            st.markdown(f"**Cell ID:** `{cell_id}`")

            # Crop
            if ref is not None:
                crop = get_cell_crop(ref, mask, selected_label, padding=30)
                if crop:
                    st.image(crop["overlay_crop"], use_container_width=True, caption="Cell crop")

            # Info card
            info = get_cell_info(ct_table, adata, cell_id)

            # Cell type with override control
            current_ct = st.session_state.manual_overrides.get(
                cell_id,
                info.get("predicted_cell_type", "Unknown")
            )
            st.markdown(f"**Predicted type:** `{current_ct}`")

            score = info.get("cell_type_score", None)
            margin = info.get("cell_type_margin", None)
            if score is not None:
                confidence = "✅ High" if (margin or 0) >= 0.15 else "⚠️ Low"
                st.markdown(
                    f"**Score:** `{score:.4f}` | **Margin:** `{margin:.4f}` | {confidence}"
                )

            # Morphology
            with st.expander("Morphology", expanded=False):
                morph_cols = ["cell_area_um2", "eccentricity", "solidity", "est_diam_px",
                              "n_transcripts", "n_genes", "leiden"]
                for col in morph_cols:
                    val = info.get(col)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        st.markdown(f"- **{col}:** `{val}`")

            # Marker scores bar chart
            scores_dict = info.get("cell_type_scores_dict", {})
            if scores_dict:
                with st.expander("Cell type marker scores", expanded=True):
                    score_df = pd.DataFrame(
                        list(scores_dict.items()), columns=["Cell type", "Score"]
                    ).sort_values("Score", ascending=True)
                    fig_scores = px.bar(
                        score_df, x="Score", y="Cell type",
                        orientation="h",
                        color="Score",
                        color_continuous_scale="Teal",
                        labels={"Score": "Marker score"},
                    )
                    fig_scores.update_layout(
                        height=260, margin=dict(l=0, r=0, t=0, b=0),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_scores, use_container_width=True, key="ctv_scores_bar")

            # Top genes
            top_genes = info.get("top_genes", [])
            if top_genes:
                with st.expander("Top expressed genes", expanded=False):
                    tg_df = pd.DataFrame(top_genes, columns=["Gene", "Expression"])
                    st.dataframe(tg_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — QC Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_qc:
    st.subheader("QC Dashboard")

    # ── All-FOV summary ────────────────────────────────────────────────────────
    st.markdown("#### Cell type composition")

    composition = load_composition(mol_root)
    if composition is not None:
        frac_cols = [c for c in composition.columns
                     if not c.startswith("n_") and c not in {"fov_name", "condition", "region"}
                     and pd.api.types.is_float_dtype(composition[c])]
        if frac_cols:
            comp_plot = composition.set_index("fov_name")[frac_cols]
            fig_comp = px.bar(
                comp_plot,
                barmode="stack",
                color_discrete_map={
                    ct: CELL_TYPE_PALETTE.get(ct, "#888888")
                    for ct in frac_cols
                },
                labels={"value": "Fraction", "variable": "Cell type", "fov_name": "FOV"},
                title="Cell type composition by FOV",
            )
            fig_comp.update_layout(
                height=350, margin=dict(t=40, b=40, l=20, r=20),
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_comp, use_container_width=True, key="ctv_qc_comp")

    # ── Current FOV ────────────────────────────────────────────────────────────
    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.markdown(f"#### {fov_name} — cell type distribution")
        if "predicted_cell_type" in fov_cells.columns:
            ct_counts = fov_cells["predicted_cell_type"].value_counts().reset_index()
            ct_counts.columns = ["Cell type", "Count"]
            ct_counts["Color"] = ct_counts["Cell type"].map(
                lambda x: CELL_TYPE_PALETTE.get(x, "#888888")
            )
            fig_pie = px.pie(
                ct_counts,
                names="Cell type",
                values="Count",
                color="Cell type",
                color_discrete_map={
                    row["Cell type"]: row["Color"] for _, row in ct_counts.iterrows()
                },
                hole=0.35,
            )
            fig_pie.update_layout(
                height=320, margin=dict(t=20, b=20, l=0, r=0),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="ctv_qc_pie")

    with col_q2:
        st.markdown("#### Marker score distributions")
        if adata is not None and "cell_type_scores" in adata.obsm:
            score_cols = adata.uns.get("cell_type_score_columns", [])
            score_df = pd.DataFrame(
                adata.obsm["cell_type_scores"],
                index=adata.obs_names,
                columns=score_cols,
            )
            fov_idx = [c for c in adata.obs_names if c.startswith(fov_name + "_")]
            if fov_idx:
                fov_scores = score_df.loc[fov_idx].melt(
                    var_name="Cell type", value_name="Score"
                )
                fig_violin = px.violin(
                    fov_scores,
                    x="Cell type",
                    y="Score",
                    color="Cell type",
                    color_discrete_map={
                        ct: CELL_TYPE_PALETTE.get(ct, "#888888") for ct in score_cols
                    },
                    box=True,
                    title="Marker score distributions",
                )
                fig_violin.update_layout(
                    height=320,
                    margin=dict(t=40, b=60, l=20, r=20),
                    showlegend=False,
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_violin, use_container_width=True, key="ctv_qc_violin")

    # ── Low-confidence cells ────────────────────────────────────────────────────
    with st.expander("Low-confidence cells", expanded=False):
        if "predicted_cell_type" in fov_cells.columns:
            lc = fov_cells[fov_cells["predicted_cell_type"] == "LowConfidence"]
            n_lc = len(lc)
            n_total = len(fov_cells)
            st.metric("Low-confidence", f"{n_lc} / {n_total} ({100*n_lc/n_total:.1f}%)")
            if n_lc > 0:
                display_lc_cols = [c for c in ["cell_id", "predicted_cell_type",
                                                "cell_type_score", "cell_type_margin",
                                                "n_transcripts", "n_genes"] if c in lc.columns]
                st.dataframe(lc[display_lc_cols], use_container_width=True, hide_index=True)

    # ── Transcript QC ───────────────────────────────────────────────────────────
    with st.expander("Transcript & assignment QC", expanded=False):
        assign_summary = load_assignment_summary(mol_root)
        if assign_summary is not None:
            st.dataframe(assign_summary, use_container_width=True, hide_index=True)

        if "n_transcripts" in fov_cells.columns:
            col_tx, col_g = st.columns(2)
            with col_tx:
                fig_tx = px.histogram(
                    fov_cells,
                    x="n_transcripts",
                    nbins=40,
                    labels={"n_transcripts": "Transcripts per cell"},
                    title="Transcripts per cell",
                    color_discrete_sequence=["#4C72B0"],
                )
                fig_tx.update_layout(height=260, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_tx, use_container_width=True, key="ctv_qc_tx")
            with col_g:
                if "n_genes" in fov_cells.columns:
                    fig_g = px.histogram(
                        fov_cells,
                        x="n_genes",
                        nbins=40,
                        labels={"n_genes": "Genes per cell"},
                        title="Genes per cell",
                        color_discrete_sequence=["#2A9D8F"],
                    )
                    fig_g.update_layout(height=260, margin=dict(t=40, b=20, l=20, r=20))
                    st.plotly_chart(fig_g, use_container_width=True, key="ctv_qc_genes")

    # ── Joint FOV summary ──────────────────────────────────────────────────────
    with st.expander("RNA–Protein FOV summary", expanded=False):
        joint = load_joint_fov_summary(mol_root)
        if joint is not None:
            score_cols_j = [c for c in joint.columns if c.startswith("score_")]
            frac_cols_j = [c for c in joint.columns if c.startswith("frac_")]
            display_j = joint[["fov_name"] + score_cols_j[:5] + frac_cols_j[:4]].copy() if score_cols_j else joint
            st.dataframe(display_j, use_container_width=True, hide_index=True)

    # ── QC export ──────────────────────────────────────────────────────────────
    with st.expander("Export QC summary"):
        qc_rows = []
        for fov_n in fov_names:
            fov_c = get_fov_cells(ct_table, fov_n)
            row = {"fov_name": fov_n, "n_cells": len(fov_c)}
            if "predicted_cell_type" in fov_c.columns:
                ct_cnt = fov_c["predicted_cell_type"].value_counts()
                for ct_name, cnt in ct_cnt.items():
                    row[f"n_{ct_name}"] = cnt
                    row[f"frac_{ct_name}"] = round(cnt / len(fov_c), 4)
            if "n_transcripts" in fov_c.columns:
                row["median_transcripts"] = fov_c["n_transcripts"].median()
                row["median_genes"] = fov_c.get("n_genes", pd.Series(dtype=float)).median()
            qc_rows.append(row)
        qc_export = pd.DataFrame(qc_rows)
        csv_bytes = qc_export.to_csv(index=False).encode()
        st.download_button(
            "Download qc_summary.csv",
            data=csv_bytes,
            file_name="qc_summary.csv",
            mime="text/csv",
        )
        if st.button("💾 Save qc_summary.csv to output dir"):
            save_path = Path(mol_root) / "qc_summary.csv"
            qc_export.to_csv(save_path, index=False)
            st.success(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Manual Review
# ══════════════════════════════════════════════════════════════════════════════
with tab_review:
    st.subheader("Manual Cell Type Review")
    st.caption(
        "Review and correct cell type labels. Changes are stored in session state "
        "and exported as manual_celltype_review.csv."
    )

    if "predicted_cell_type" not in fov_cells.columns:
        st.warning("No predicted_cell_type column found.")
    else:
        cell_type_options = sorted(set(CELL_TYPE_ORDER) | set(all_types))

        # ── Filter controls ────────────────────────────────────────────────────
        rev_col1, rev_col2, rev_col3 = st.columns(3)
        with rev_col1:
            filter_ct = st.multiselect(
                "Filter by predicted type",
                all_types,
                default=all_types,
                key="ctv_rev_filter",
            )
        with rev_col2:
            show_only_lc = st.checkbox("Show only LowConfidence", key="ctv_rev_lc")
        with rev_col3:
            show_only_changed = st.checkbox("Show only changed cells", key="ctv_rev_changed")

        # ── Build review table ─────────────────────────────────────────────────
        rev_df = fov_cells.copy()
        if filter_ct:
            rev_df = rev_df[rev_df["predicted_cell_type"].isin(filter_ct)]
        if show_only_lc:
            rev_df = rev_df[rev_df["predicted_cell_type"] == "LowConfidence"]
        if show_only_changed:
            changed_ids = set(st.session_state.manual_overrides.keys())
            if "cell_id" in rev_df.columns:
                rev_df = rev_df[rev_df["cell_id"].isin(changed_ids)]

        # Add current (possibly overridden) type
        if "cell_id" in rev_df.columns:
            rev_df = rev_df.copy()
            rev_df["current_type"] = rev_df["cell_id"].map(
                lambda cid: st.session_state.manual_overrides.get(cid, "")
            )
            rev_df["override"] = rev_df["current_type"] != ""

        # Display columns
        show_review_cols = [c for c in [
            "cell_id", "predicted_cell_type", "current_type", "override",
            "cell_type_score", "cell_type_margin",
            "n_transcripts", "n_genes", "leiden",
        ] if c in rev_df.columns]

        st.markdown(f"**Showing {len(rev_df)} cells**")
        ev_rev = st.dataframe(
            rev_df[show_review_cols].reset_index(drop=True),
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="ctv_rev_table",
            hide_index=True,
        )

        # ── Edit selected cell ─────────────────────────────────────────────────
        selected_row_idx = None
        if ev_rev and ev_rev.selection.rows:
            selected_row_idx = ev_rev.selection.rows[0]

        st.divider()
        edit_col1, edit_col2 = st.columns([2, 2])

        with edit_col1:
            st.markdown("**Edit selected cell**")
            if selected_row_idx is not None and "cell_id" in rev_df.columns:
                sel_cell_id = str(rev_df.iloc[selected_row_idx]["cell_id"])
                orig_type = str(rev_df.iloc[selected_row_idx]["predicted_cell_type"])
                current_override = st.session_state.manual_overrides.get(sel_cell_id, orig_type)

                st.markdown(f"Cell: `{sel_cell_id}`")
                st.markdown(f"Original: `{orig_type}`")

                new_type = st.selectbox(
                    "New cell type",
                    cell_type_options,
                    index=cell_type_options.index(current_override) if current_override in cell_type_options else 0,
                    key="ctv_rev_newtype",
                )

                c_apply, c_reset = st.columns(2)
                with c_apply:
                    if st.button("Apply", key="ctv_rev_apply", type="primary"):
                        st.session_state.manual_overrides[sel_cell_id] = new_type
                        st.rerun()
                with c_reset:
                    if sel_cell_id in st.session_state.manual_overrides:
                        if st.button("Reset to original", key="ctv_rev_reset"):
                            del st.session_state.manual_overrides[sel_cell_id]
                            st.rerun()
            else:
                st.info("Select a row in the table above to edit.")

        with edit_col2:
            # Show cell crop for selected cell
            if selected_row_idx is not None and mask is not None and ref is not None and "cell_id" in rev_df.columns:
                sel_cid = str(rev_df.iloc[selected_row_idx]["cell_id"])
                sel_lbl = int(sel_cid.rsplit("_", 1)[-1]) if "_" in sel_cid else None
                if sel_lbl is not None:
                    crop_rev = get_cell_crop(ref, mask, sel_lbl, padding=35)
                    if crop_rev:
                        st.image(crop_rev["overlay_crop"], use_container_width=True,
                                caption=f"Cell {sel_lbl}")

        # ── Batch operations ───────────────────────────────────────────────────
        with st.expander("Batch relabel (advanced)"):
            st.markdown("Relabel **all LowConfidence** cells in this FOV to a chosen type.")
            batch_type = st.selectbox("Set all LowConfidence to:", cell_type_options, key="ctv_batch_type")
            if st.button("Apply batch relabel", key="ctv_batch_apply"):
                lc_cells = fov_cells[fov_cells["predicted_cell_type"] == "LowConfidence"]
                if "cell_id" in lc_cells.columns:
                    for cid in lc_cells["cell_id"]:
                        st.session_state.manual_overrides[str(cid)] = batch_type
                st.success(f"Relabelled {len(lc_cells)} cells to {batch_type}")
                st.rerun()

        # ── Save ───────────────────────────────────────────────────────────────
        st.divider()
        c_save1, c_save2 = st.columns(2)

        with c_save1:
            if st.button("💾 Save manual_celltype_review.csv", type="primary", key="ctv_save_manual"):
                save_path = Path(mol_root) / "manual_celltype_review.csv"
                result_df = save_manual_review(
                    ct_table,
                    st.session_state.manual_overrides,
                    str(save_path),
                )
                st.success(f"Saved {len(st.session_state.manual_overrides)} overrides → {save_path}")

        with c_save2:
            # Download button (in-browser)
            if st.session_state.manual_overrides:
                ov_df = pd.DataFrame(
                    [{"cell_id": k, "manual_cell_type": v}
                     for k, v in st.session_state.manual_overrides.items()]
                )
                dl_bytes = ov_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download overrides CSV",
                    data=dl_bytes,
                    file_name="manual_overrides.csv",
                    mime="text/csv",
                    key="ctv_dl_overrides",
                )

        # ── Override summary ───────────────────────────────────────────────────
        if st.session_state.manual_overrides:
            with st.expander(f"Override summary ({len(st.session_state.manual_overrides)} cells)"):
                ov_items = list(st.session_state.manual_overrides.items())
                ov_summary = pd.DataFrame(ov_items, columns=["cell_id", "manual_cell_type"])
                # Join original type
                if "cell_id" in ct_table.columns and "predicted_cell_type" in ct_table.columns:
                    orig_map = ct_table.set_index("cell_id")["predicted_cell_type"].to_dict()
                    ov_summary["original_type"] = ov_summary["cell_id"].map(orig_map)
                st.dataframe(ov_summary, use_container_width=True, hide_index=True)
