"""CosMx Segmentation QC Workbench — main entry point.

Launch:
    cd /path/to/CosMx_2026
    streamlit run cell_segmentation_pipeline/app/streamlit_app.py
"""
import sys
from pathlib import Path

# Make project root importable from all pages
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

st.set_page_config(
    page_title="CosMx Segmentation QC Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 CosMx Segmentation QC Workbench")
st.caption("Segmentation quality review for NanoString CosMx — GO Lab, Kyoto University")

st.markdown("""
## Navigation

Use the sidebar to open a review tool:

| Page | Purpose |
|---|---|
| **FOV Viewer** | Browse any FOV with zoom/pan overlay |
| **Model Comparison** | Side-by-side comparison of segmentation models |
| **Cell Inspector** | Inspect individual cells: crop, metadata, neighbourhood |
| **QC Dashboard** | Population-level statistics and histograms |
| **Random Cell Review** | Manual QC labelling — Good / Over-seg / Under-seg / Merged |
| **Cell Type Validation** | RNA/Protein cell type color map, marker expression, manual review |
| **RNA-Guided Segmentation** | Run Baysor on Cellpose/StarDist/InstanSeg priors; compare DAPI-only vs RNA-guided |

---

## Quick Status
""")

# Show available data summary
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]

col1, col2, col3 = st.columns(3)

pilot_masks = list((project_root / "outputs" / "pilot_4fov" / "masks").glob("*nuclear_mask.tif"))
new_masks   = list((project_root / "outputs" / "cell_segmentation_pipeline" / "masks").glob("*nuclear_mask.tif"))
raw_tifs    = list((project_root / "raw_data" / "pilot_4fov" / "slide1_RNA" / "morphology_images").glob("*.TIF"))

with col1:
    st.metric("Raw TIF files", len(raw_tifs))
with col2:
    st.metric("Pilot masks", len(pilot_masks), help="outputs/pilot_4fov/masks/")
with col3:
    st.metric("Pipeline masks", len(new_masks), help="outputs/cell_segmentation_pipeline/masks/")

if not pilot_masks and not new_masks:
    st.warning(
        "No segmentation masks found yet. Run the segmentation pipeline first:\n\n"
        "```bash\n"
        "python cell_segmentation_pipeline/scripts/run_segmentation.py \\\n"
        "  --input raw_data/pilot_4fov/slide1_RNA/morphology_images \\\n"
        "  --output outputs/cell_segmentation_pipeline \\\n"
        "  --model cellpose_cpsam\n"
        "```"
    )
else:
    st.success(f"Data ready — {len(pilot_masks) + len(new_masks)} mask file(s) found.")

# Molecular analysis status
mol_root = project_root / "outputs" / "cosmx_molecular_analysis"
ct_table_path = mol_root / "cell_type_table.csv"
anndata_path  = mol_root / "anndata" / "rna_annotated.h5ad"
with col1:
    pass
col4, col5 = st.columns(2)
with col4:
    st.metric(
        "Cell type table",
        "✓ Ready" if ct_table_path.exists() else "✗ Missing",
        help=str(ct_table_path),
    )
with col5:
    st.metric(
        "Annotated AnnData",
        "✓ Ready" if anndata_path.exists() else "✗ Missing",
        help=str(anndata_path),
    )

if ct_table_path.exists():
    import pandas as _pd
    _ct = _pd.read_csv(ct_table_path)
    st.info(
        f"**Molecular analysis ready** — {len(_ct)} annotated cells across "
        f"{_ct.get('fov_name', _ct.iloc[:, 0]).nunique()} FOVs. "
        f"Open **Cell Type Validation** in the sidebar."
    )
else:
    st.info(
        "Molecular analysis not yet run. To annotate cells with RNA/Protein data:\n\n"
        "```bash\n"
        "python cosmx_molecular_analysis/scripts/run_full_pipeline.py \\\n"
        "  --config cosmx_molecular_analysis/config/default_config.yaml\n"
        "```"
    )

st.markdown("---")
st.markdown(
    "_Tip: The **pilot_4fov** outputs (CosMx cpsam) are loaded by default. "
    "Switch the output root in the sidebar of each page to compare with new pipeline results._"
)
