"""QC Dashboard — population-level statistics, histograms, flagged cells."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.image_loader import (
    DEFAULT_OUTPUT_ROOTS, discover_fovs, discover_models, load_cell_table,
)
from utils.qc_utils import (
    area_histogram, compute_stats, diameter_histogram,
    flag_cells, model_comparison_bar, scatter_area_vs_diam,
)

st.set_page_config(page_title="QC Dashboard", layout="wide")
st.title("QC Dashboard")
st.caption("Population-level segmentation quality metrics and flagged-cell analysis")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_QC_OVERVIEW_PATH = _PROJECT_ROOT / "outputs" / "cell_segmentation_pipeline" / "qc" / "fov_qc_overview.csv"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    output_roots = st.multiselect(
        "Output roots",
        DEFAULT_OUTPUT_ROOTS,
        default=DEFAULT_OUTPUT_ROOTS,
        format_func=lambda p: Path(p).name,
    )
    if not output_roots:
        output_roots = DEFAULT_OUTPUT_ROOTS

    # Collect all FOV+model combos
    combos: list[dict] = []
    for root in output_roots:
        for fov in discover_fovs(root):
            for model in discover_models(fov, root):
                combos.append({"root": root, "fov": fov, "model": model,
                                "label": f"{Path(root).name} / {fov} / {model}"})

    if not combos:
        st.warning("No cell tables found.")
        st.stop()

    selected_labels = st.multiselect(
        "FOV × Model",
        [c["label"] for c in combos],
        default=[c["label"] for c in combos],
    )
    selected = [c for c in combos if c["label"] in selected_labels]

    st.divider()
    min_area = st.number_input("Min cell area (px²)", 10, 500, 50)
    max_area = st.number_input("Max cell area (px²)", 1000, 50000, 10000)

if not selected:
    st.warning("Select at least one FOV × Model combination.")
    st.stop()

# ── Load cell tables ──────────────────────────────────────────────────────────
all_ct: list[pd.DataFrame] = []
summaries: list[dict] = []

for sel in selected:
    ct = load_cell_table(sel["fov"], sel["model"], sel["root"])
    if ct is None or len(ct) == 0:
        continue
    ct = ct.copy()
    ct["_combo"] = sel["label"]
    ct["_fov"]   = sel["fov"]
    ct["_model"] = sel["model"]
    all_ct.append(ct)
    s = compute_stats(ct, min_area, max_area)
    s["combo"] = sel["label"]
    summaries.append(s)

if not all_ct:
    st.error("No cell tables could be loaded.")
    st.stop()

combined = pd.concat(all_ct, ignore_index=True)

# ── Metric cards ──────────────────────────────────────────────────────────────
st.subheader("Summary metrics")
if len(summaries) == 1:
    s = summaries[0]
    cols = st.columns(5)
    cols[0].metric("Cells detected",   s["n_cells"])
    cols[1].metric("Median area (px²)", f"{s['median_area_px']:.0f}")
    cols[2].metric("Median diam (px)",  f"{s['median_diam_px']:.1f}")
    cols[3].metric("Too small (%)",     f"{s['pct_too_small']:.1f}%")
    cols[4].metric("Too large (%)",     f"{s['pct_too_large']:.1f}%")
else:
    summary_df = pd.DataFrame(summaries)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Multi-combo cell count bar chart ─────────────────────────────────────────
bar_data = [{"label": s["combo"], "n_cells": s["n_cells"]} for s in summaries]
st.plotly_chart(model_comparison_bar(bar_data), use_container_width=True)

# ── Parameter comparison sub-section ─────────────────────────────────────────
with st.expander("Preprocessing parameter comparison (TopHat radius)"):
    st.markdown("""
Run the comparison from the CLI and reload:

```bash
python cell_segmentation_pipeline/scripts/run_preprocess.py \\
    --input  raw_data/pilot_4fov/slide1_RNA/morphology_images \\
    --output outputs/cell_segmentation_pipeline/enhanced \\
    --method tophat --radius 10   # repeat for 15, 20, 25
```

Then re-run segmentation at each enhanced output and add it to the output root selector above.
""")

# ── Expansion distance sub-section ───────────────────────────────────────────
with st.expander("Expansion distance comparison"):
    st.markdown("""
Re-run segmentation with different `--expand_px` values to compare.

Typical options: 0, 3, 5, 8, 10 px.
The expanded mask overlay on the **FOV Viewer** page shows collision risk with neighbours.
""")

# ── Histograms ────────────────────────────────────────────────────────────────
st.subheader("Cell size distributions")
col1, col2 = st.columns(2)

if len(selected) == 1:
    with col1:
        st.plotly_chart(area_histogram(combined, min_area, max_area), use_container_width=True)
    with col2:
        st.plotly_chart(diameter_histogram(combined), use_container_width=True)
else:
    # Multi-combo: colour by combo
    with col1:
        fig = px.histogram(
            combined, x="area", color="_combo", nbins=60, barmode="overlay",
            opacity=0.6,
            labels={"area": "Cell area (px²)", "_combo": "Dataset"},
            title="Cell area distribution",
        )
        fig.add_vline(x=min_area, line_dash="dash", line_color="red")
        fig.add_vline(x=max_area, line_dash="dash", line_color="orange")
        fig.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "est_diam_px" in combined.columns:
            fig2 = px.histogram(
                combined, x="est_diam_px", color="_combo", nbins=50, barmode="overlay",
                opacity=0.6,
                labels={"est_diam_px": "Est. diameter (px)", "_combo": "Dataset"},
                title="Cell diameter distribution",
            )
            fig2.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig2, use_container_width=True)

# ── Area vs diameter scatter ──────────────────────────────────────────────────
with st.expander("Area vs diameter scatter (circularity sanity check)"):
    sample = combined.sample(min(2000, len(combined)), random_state=0)
    if len(selected) == 1:
        st.plotly_chart(scatter_area_vs_diam(sample), use_container_width=True)
    else:
        if "est_diam_px" in sample.columns:
            fig3 = px.scatter(
                sample, x="area", y="est_diam_px", color="_combo",
                opacity=0.4, labels={"area": "Area (px²)", "est_diam_px": "Est. diameter (px)"},
                title="Area vs diameter",
            )
            fig3.update_layout(height=350, margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig3, use_container_width=True)

# ── Flagged cells table ───────────────────────────────────────────────────────
st.subheader("Flagged cells")
flagged = flag_cells(combined, min_area, max_area)
flagged_bad = flagged[flagged["qc_flag"] != "pass"]

if len(flagged_bad) == 0:
    st.success("No cells flagged outside the thresholds.")
else:
    st.warning(f"{len(flagged_bad)} flagged cells ({len(flagged_bad)/len(combined)*100:.1f}%)")
    show_cols = [c for c in ["_combo","_fov","label","area","est_diam_px","qc_flag"] if c in flagged_bad.columns]
    st.dataframe(flagged_bad[show_cols].head(200), use_container_width=True, hide_index=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("Export")
csv_bytes = combined.to_csv(index=False).encode()
st.download_button("Download combined cell table (CSV)", data=csv_bytes,
                   file_name="combined_cell_table.csv", mime="text/csv")

summary_csv = pd.DataFrame(summaries).to_csv(index=False).encode()
st.download_button("Download QC summary (CSV)", data=summary_csv,
                   file_name="qc_summary.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# FOV 総合 QC テーブル（3軸評価）
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("FOV 総合 QC テーブル（3軸評価）")
st.caption(
    "3軸で segmentation 品質を評価します。"
    "**Population IoU だけでは判断しないでください。**"
)

if _QC_OVERVIEW_PATH.exists():
    _overview = pd.read_csv(_QC_OVERVIEW_PATH)

    # ── カラーバッジ付きテーブル ──────────────────────────────────────────────
    def _color_dapi(val):
        if val == "Good":       return "background-color: #d4edda"
        if val == "Acceptable": return "background-color: #fff3cd"
        if val == "Warning":    return "background-color: #f8d7da"
        return ""

    def _color_flag(val):
        return "background-color: #f8d7da; color: #721c24" if val is True else ""

    def _color_reg(val):
        if val == "OK":           return "color: #155724; font-weight: bold"
        if val == "Minor offset": return "color: #856404"
        if val == "Warning":      return "color: #721c24; font-weight: bold"
        return ""

    def _color_score(val):
        try:
            v = float(val)
            if v >= 0.60: return "background-color: #d4edda"
            if v >= 0.40: return "background-color: #fff3cd"
            return "background-color: #f8d7da"
        except Exception:
            return ""

    # 表示カラムを整理
    _disp_cols = [
        "fov", "model",
        "custom_cell_count", "cosmx_cell_count", "custom_to_cosmx_ratio",
        "dapi_snr", "dapi_status",
        "matched_pair_iou_median", "one_to_one_match_rate",
        "match_rate_custom",
        "registration_status",
        "median_centroid_dist" if "median_centroid_dist" in _overview.columns else "mean_centroid_dist",
        "oversegmentation_flag", "undersegmentation_flag",
        "composite_score",
        "recommended_action",
    ]
    _disp_cols = [c for c in _disp_cols if c in _overview.columns]
    _ov_disp = _overview[_disp_cols].copy()

    # rename for display
    _rename = {
        "custom_cell_count":         "Custom cells",
        "cosmx_cell_count":          "CosMx cells",
        "custom_to_cosmx_ratio":     "Cell ratio",
        "dapi_snr":                  "DAPI SNR",
        "dapi_status":               "DAPI quality",
        "matched_pair_iou_median":   "Matched-pair IoU",
        "one_to_one_match_rate":     "1-to-1 rate",
        "match_rate_custom":         "Match rate",
        "registration_status":       "Registration",
        "median_centroid_dist":      "Median centroid dist",
        "mean_centroid_dist":        "Mean centroid dist",
        "oversegmentation_flag":     "Over-seg?",
        "undersegmentation_flag":    "Under-seg?",
        "composite_score":           "Score",
        "recommended_action":        "Action",
    }
    _ov_disp = _ov_disp.rename(columns={k: v for k, v in _rename.items() if k in _ov_disp.columns})

    styled = _ov_disp.style
    if "DAPI quality" in _ov_disp.columns:
        styled = styled.applymap(_color_dapi, subset=["DAPI quality"])
    if "Registration" in _ov_disp.columns:
        styled = styled.applymap(_color_reg, subset=["Registration"])
    if "Score" in _ov_disp.columns:
        styled = styled.applymap(_color_score, subset=["Score"])
    for flag_col in ["Over-seg?", "Under-seg?"]:
        if flag_col in _ov_disp.columns:
            styled = styled.applymap(_color_flag, subset=[flag_col])

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Composite score 棒グラフ ──────────────────────────────────────────────
    if "composite_score" in _overview.columns:
        fig_score = px.bar(
            _overview, x="fov", y="composite_score",
            color="dapi_status" if "dapi_status" in _overview.columns else None,
            color_discrete_map={"Good": "#2A9D8F", "Acceptable": "#E9C46A", "Warning": "#E63946"},
            labels={"composite_score": "Composite QC Score", "fov": "FOV"},
            title="Composite QC Score by FOV（高いほど良好）",
        )
        fig_score.add_hline(y=0.60, line_dash="dash", line_color="green",
                            annotation_text="Good threshold (0.60)")
        fig_score.add_hline(y=0.40, line_dash="dash", line_color="orange",
                            annotation_text="Review threshold (0.40)")
        fig_score.update_layout(
            height=300, margin=dict(t=40, b=20, l=20, r=20), yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_score, use_container_width=True, key="qdash_score_bar")

    # ── Download ──────────────────────────────────────────────────────────────
    _ov_csv = _overview.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download fov_qc_overview.csv",
        data=_ov_csv,
        file_name="fov_qc_overview.csv",
        mime="text/csv",
        key="qdash_dl_overview",
    )

    with st.expander("スコアの説明"):
        st.markdown("""
**Composite QC Score** の構成（0–1、高いほど良好）:

| 要素 | 重み | 説明 |
|------|------|------|
| DAPI SNR | 30% | mask が DAPI 核の上に乗っているか |
| Cell count consistency | 20% | custom と CosMx の細胞数が近いか |
| Area distribution | 15% | 核サイズが CosMx の 40–70% 程度か |
| One-to-one match rate | 20% | 1対1対応ができているか |
| Matched-pair IoU | 10% | 近いペアでの形状一致度 |
| Over/under-seg penalty | −15% each | 過分割・過結合のペナルティ |

**Population IoU はスコアに含みません。**
Population IoU が低くても、上記5軸が良好なら Cell Type Validation に進めます。
        """)

else:
    st.info(
        "fov_qc_overview.csv が見つかりません。以下のコマンドで生成してください:\n\n"
        "```bash\n"
        "python cell_segmentation_pipeline/scripts/run_fov_qc_overview.py\n"
        "```"
    )
