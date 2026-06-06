# CosMx 2026 — Dementia Spatial Multi-omics Analysis
**GO Laboratory, Kyoto University**
**Spatial RNA + Protein: Human Alzheimer's Disease vs Control Brain**

---

## Table of Contents

1. [Study Overview](#1-study-overview)
2. [Data Description](#2-data-description)
3. [Directory Structure](#3-directory-structure)
4. [Environment Setup](#4-environment-setup)
5. [Quick Start — 4-FOV Pilot Analysis](#5-quick-start--4-fov-pilot-analysis)
6. [Segmentation Pipeline — Detailed Guide](#6-segmentation-pipeline--detailed-guide)
7. [Transcript Assignment](#7-transcript-assignment)
8. [Building AnnData & QC](#8-building-anndata--qc)
9. [Clustering & Cell Type Annotation](#9-clustering--cell-type-annotation)
10. [Spatial Proteomics Analysis](#10-spatial-proteomics-analysis)
11. [Spatial Transcriptomics Analysis](#11-spatial-transcriptomics-analysis)
12. [RNA–Protein Integration](#12-rnaprotein-integration)
13. [Scaling to All 144 FOVs](#13-scaling-to-all-144-fovs)
14. [Key Findings from Pilot](#14-key-findings-from-pilot)
15. [Publication Checklist](#15-publication-checklist)

---

## 1. Study Overview

This project applies **NanoString CosMx Spatial Molecular Imager (SMI)** to post-mortem human brain tissue from Alzheimer's disease (AD) patients and age-matched healthy controls. The aim is to generate spatially-resolved single-cell transcriptomic and proteomic profiles to identify disease-specific cell-type changes, spatial co-localization patterns, and molecular hallmarks of AD pathology.

### Biological Question
How do cell-type composition, gene expression, and protein abundance differ spatially between AD and control brains, across frontal cortex and hippocampus — two regions with distinct patterns of neurodegeneration?

### Experimental Design

| Parameter | Details |
|---|---|
| Technology | CosMx Spatial Molecular Imager (NanoString) |
| Tissue | Post-mortem human brain (FFPE sections) |
| Panel (RNA) | Human RNA 6K Discovery Panel (~6,000 genes) |
| Panel (Protein) | CosMx Immunofluorescence protein panel |
| Pixel size | 0.120274 µm/px |
| FOVs | 144 total (72 per slide × 2 slides) |
| Conditions | Alzheimer's Disease (AD) vs Control |
| Regions | Frontal cortex (F), Hippocampus (H) |

### Slides

| Slide ID | Study Directory | Modality | Run ID |
|---|---|---|---|
| 2511276KSlide116 | Study20251223_GOlab_cosmxrna1 | RNA 6K | 20251127_074826_S1 |
| 2511276KSlide2712 | Study20251223_GOlab_cosmxrna2 | RNA 6K | 20251127_074826_S2 |
| 251121Slide116 | Study20251223_GOlab_cosmxpro1 | Protein | 20251121_071111_S1 |
| 251121Slide116.2 | Study20251223_GOlab_cosmxpro2 | Protein | — |

---

## 2. Data Description

### 2.1 Pilot Dataset — 4 FOVs (Local, Self-Contained)

All files for the pilot analysis are available locally at `raw_data/pilot_4fov/`. **No external drive is required.**

| FOV | Condition | Brain Region | Sample ID | Gender | Age |
|---|---|---|---|---|---|
| FOV00001 | AD | Frontal cortex | 1 | M | 88 |
| FOV00007 | AD | Hippocampus | 1 | M | 88 |
| FOV00037 | Control | Frontal cortex | 4 | M | 68 |
| FOV00043 | Control | Hippocampus | 4 | M | 68 |

These 4 FOVs were selected to capture the two main conditions × two brain regions, enabling pilot validation of the full pipeline.

### 2.2 Morphology Image Channels (multi-channel TIFF)

Each FOV produces a multi-channel TIFF (C × H × W format, C-first axis):

| Channel Index | Marker | Role |
|---|---|---|
| 0 | DAPI | Nuclear stain — **primary segmentation channel** |
| 1 | PanCK | Epithelial marker (low in brain) |
| 2 | G (autofluorescence) | Background channel |
| 3 | Membrane | Cell boundary marker |
| 4 | CD45 | Immune cell marker |

> **Important for brain tissue:** Channel 0 (DAPI) is used for nuclear segmentation. Channel 2 was explored for cytoplasmic signal in early pilot runs but DAPI gives superior results for cpsam model.

### 2.3 Per-FOV Decoded Files

For each FOV, the CellStatsDir contains:

| File | Description |
|---|---|
| `CellLabels_F*.tif` | CosMx native cell label mask (integer, each cell = unique ID) |
| `CompartmentLabels_F*.tif` | Nucleus/cytoplasm/background compartment map |
| `CellBoundaries_F*.csv` | Polygon boundaries for native CosMx cells |
| `*_Cell_Stats_F*.csv` | Per-cell area, centroid, morphology metrics |
| `*_target_call_coord.csv.gz` | Transcript coordinates with quality scores |
| `*_complete_code_cell_target_call_coord.csv` | Full decoded transcript table per FOV |

### 2.4 FlatFiles (Slide-Level)

| File | Description |
|---|---|
| `*_exprMat_file.csv.gz` | Cell × gene expression matrix (CosMx native cells) |
| `*_metadata_file.csv.gz` | Per-cell metadata (FOV, centroid, area, InSituType labels) |
| `*_fov_positions_file.csv.gz` | FOV spatial coordinates on the slide |
| `*_polygons.csv.gz` | Cell polygon data |
| `*_tx_file.csv.gz` | **All transcripts with (x, y, z, gene, qv)** — 2.3 GB, required for custom cell assignment |

### 2.5 Full Dataset Location (External SSD)

The full 144-FOV dataset is on the external SSD mounted at `/mnt/external/`:

```
/mnt/external/data/
├── Study20251223_GOlab_cosmxrna1/   # Slide 1 RNA (72 FOVs)
├── Study20251223_GOlab_cosmxrna2/   # Slide 2 RNA (72 FOVs)
├── Study20251223_GOlab_cosmxpro1/   # Slide 1 Protein
└── Study20251223_GOlab_cosmxpro2/   # Slide 2 Protein
```

---

## 3. Directory Structure

```
CosMx_2026/
├── README.md                          # This file
├── config/
│   ├── study_config.yaml              # Central configuration (paths, QC, params)
│   ├── sample_manifest.csv            # FOV → biological metadata mapping
│   └── seg_best_params.json           # Best segmentation parameters from pilot tuning
│
├── raw_data/                          # Self-contained pilot data (no SSD needed)
│   ├── pilot_4fov/
│   │   ├── slide1_RNA/
│   │   │   ├── morphology_images/     # 4 × multi-channel DAPI TIFFs
│   │   │   ├── per_fov_decoded/       # FOV00001, 00007, 00037, 00043 decoded files
│   │   │   └── flatfiles/             # exprMat, metadata, tx_file (2.3 GB), polygons
│   │   └── slide1_protein/
│   │       ├── morphology_images/     # 4 × protein morphology TIFFs
│   │       ├── per_fov_decoded/       # Per-FOV protein decoded files
│   │       └── flatfiles/             # exprMat, metadata, fov_positions, polygons
│   └── additional_data/
│       └── Dementia_CosMx_Analysis - FOV_metadata.xlsx
│
├── scripts/                           # Main analysis pipeline (steps 00–12)
│   ├── 00_check_env.py
│   ├── 01_load_flatfiles.py
│   ├── 02_image_enhance.py
│   ├── 03_segment_cellpose.py
│   ├── 04_segment_stardist.py
│   ├── 05_segment_instanseg.py
│   ├── 06_assign_transcripts.py
│   ├── 07_build_anndata.py
│   ├── 08_qc_filter.py
│   ├── 09_normalize_cluster.py
│   ├── 10_cell_typing.py
│   ├── 11_spatial_analysis.py
│   ├── 12_compare_segmenters.py
│   ├── cosmx_utils.py                 # Shared helper functions
│   └── ssd_scripts/                   # Pilot exploration & tuning scripts
│       ├── step01_verify_setup.py     # Environment verification
│       ├── step02_enhance_images.py   # TopHat enhancement for 4 FOVs
│       ├── step03_segmentation.py     # Cellpose segmentation for 4 FOVs
│       ├── step04_qc_report.py        # QC summary report
│       ├── explore_0*.py              # Channel/z-plane/model exploration
│       ├── proseg_*.py                # ProSeg 3D-aware segmentation experiments
│       └── cosmx_seg_tuner*.py        # Automated parameter grid search
│
├── outputs/                           # Main pipeline outputs
│   ├── anndata/                       # .h5ad files (native + custom segmentation)
│   ├── enhanced_images/               # TopHat-enhanced TIFFs
│   ├── pilot_4fov/                    # Pilot segmentation results
│   │   ├── enhanced/                  # Enhanced TIFs
│   │   ├── masks/                     # Nuclear mask TIFs
│   │   ├── expanded_masks/            # Expanded (cytoplasmic) masks
│   │   ├── overlays/                  # Visual QC overlays
│   │   ├── cell_tables/               # Per-cell morphology CSVs
│   │   └── summary/                   # QC summary tables
│   ├── pilot_model_comparison*/       # Cellpose model comparison results
│   └── tuner_results*/                # Parameter sweep results
│
├── outputs_ssd/                       # Pilot exploration outputs
│   ├── proseg_pilot/                  # ProSeg 3D segmentation results
│   ├── cellprofiler_pilot/            # CellProfiler comparison
│   ├── pilot_segmentation*/           # Early segmentation runs
│   ├── explore_0*/                    # Channel/z-plane exploration outputs
│   └── stardist_tuning/              # StarDist parameter tuning
│
├── results/                           # Downstream biological analysis (R + Python)
│   ├── 00_inspect/
│   ├── 02A_build_protein_seurat/      # Seurat protein object
│   ├── 02B_build_rna/                 # RNA expression objects
│   ├── 03A_rna_brain_analysis/        # Cell type marker analysis
│   ├── 03B_protein_brain_aware/       # Protein spatial analysis
│   ├── 04_integrate_rna_protein_by_fov/  # Multi-modal integration
│   └── 08_each_FOV_highres_AIF1/     # High-resolution per-FOV visualizations
│
└── notebooks/                         # Jupyter notebooks for exploration
```

---

## 4. Environment Setup

### 4.1 Conda Environment

```bash
# Recommended: create a dedicated environment
conda create -n cosmx_env python=3.10
conda activate cosmx_env

# Core dependencies
pip install cellpose[gui]         # Cellpose with cpsam support
pip install stardist              # StarDist
pip install instanseg             # InstanSeg
pip install anndata scanpy squidpy
pip install tifffile scikit-image matplotlib seaborn
pip install pandas numpy scipy
pip install harmonypy             # Batch correction
pip install openpyxl              # Excel metadata reading
pip install pyyaml
```

For R-based downstream analysis:
```r
install.packages("Seurat")
BiocManager::install(c("scran", "scater"))
install.packages("ggplot2")
```

### 4.2 Verify Environment

```bash
conda activate cosmx_env
cd /home/shamim/CosMx_2026
python scripts/00_check_env.py
```

This checks GPU availability, all package versions, and confirms all 4-FOV data paths resolve correctly.

---

## 5. Quick Start — 4-FOV Pilot Analysis

Run the four step scripts in order. All output goes to `outputs/pilot_4fov/`.

```bash
conda activate cosmx_env
cd /home/shamim/CosMx_2026

# Step 1 — Verify data paths and packages
python scripts/ssd_scripts/step01_verify_setup.py

# Step 2 — TopHat image enhancement (DAPI channel)
python scripts/ssd_scripts/step02_enhance_images.py

# Step 3 — Cellpose cpsam segmentation + CosMx mask comparison
python scripts/ssd_scripts/step03_segmentation.py

# Step 4 — QC report (cell counts, size distributions, overlays)
python scripts/ssd_scripts/step04_qc_report.py
```

Expected runtime: ~5 minutes per FOV on GPU, ~20 minutes total.

---

## 6. Segmentation Pipeline — Detailed Guide

Segmentation is the most critical step for downstream quality. Poor segmentation leads to mixed transcripts across cells, inflated counts, and false biological signals.

### 6.1 Why Custom Segmentation?

CosMx's built-in segmentation (InSituType / native CellLabels) uses a fixed algorithm tuned for average tissue. For brain tissue, it:
- Underestimates cell counts (~341 cells/FOV in pilot vs ~947 in QuPath)
- Merges neighboring neurons
- Misses small glial cells

Custom segmentation with Cellpose cpsam recovers **~1,200–1,300 cells per FOV** in the pilot (3–4× more than CosMx native).

### 6.2 Image Enhancement — Step 2

**Best method (from pilot): White Top-Hat filter, radius 20 px**

```python
# Implemented in: scripts/02_image_enhance.py
# Parameters in:  config/study_config.yaml → enhancement section
method: "TopHat"
tophat_radius_px: 20
```

The Top-Hat filter suppresses uneven background illumination while preserving bright nuclei — essential for CosMx DAPI images which have spatially varying background. Do **not** use CLAHE alone; it oversaturates dim nuclei in sparse brain regions.

| Method | Notes |
|---|---|
| **TopHat r=20** | Best for both dense (hippocampus) and sparse (frontal) brain regions |
| TopHat r=10 | Too small — misses large neurons, inflates cell count artifacts |
| Rolling Ball | Useful for cytoplasmic signal but not for nuclear-only segmentation |
| CLAHE only | Oversaturates; only use as post-processing on very dim FOVs |

Run for the 4 pilot FOVs:
```bash
python scripts/02_image_enhance.py --slide 1 --fovs FOV00001 FOV00007 FOV00037 FOV00043
```

Output: `outputs/enhanced/<slide_id>/FOV*_enhanced.tif`

### 6.3 Cellpose Segmentation — Step 3

**Best model (from pilot): `cpsam` (Cellpose Segment Anything Model)**

```python
# Parameters in: config/study_config.yaml → segmentation.cellpose
model:              cpsam
diameter:           25          # px; ~3 µm nuclei @ 0.12 µm/px
cellprob_threshold: -1.0        # lower = more permissive (catches dim nuclei)
flow_threshold:     0.6
channel:            0           # DAPI
expand_distance_px: 5           # expand nuclear mask into cytoplasm for tx assignment
```

> **Critical:** `diameter=25` was tuned specifically for human brain on CosMx. At 0.12 µm/px, neurons are ~15–45 µm, giving 125–375 px diameter — but cpsam uses diameter as an initial scale hint, not a hard cutoff. Setting it too large (>35) misses small glia.

Run Cellpose on enhanced images:
```bash
python scripts/03_segment_cellpose.py \
    --model cpsam \
    --fovs FOV00001 FOV00007 FOV00037 FOV00043
```

Run all three models and compare:
```bash
python scripts/03_segment_cellpose.py --model cpsam nuclei cyto3
python scripts/12_compare_segmenters.py
```

### 6.4 Alternative Segmenters (for comparison)

| Segmenter | Script | Notes |
|---|---|---|
| StarDist | `04_segment_stardist.py` | Good for round nuclei; slightly faster than cpsam; scale=2.0 needed for CosMx resolution |
| InstanSeg | `05_segment_instanseg.py` | Multi-channel aware; useful if membrane channel is strong |
| Mesmer | `ssd_scripts/02_pilot_mesmer_comparison_v1.py` | Requires TensorFlow; tested in pilot |
| CellProfiler | `ssd_scripts/cellprofiler_pilot/` | Classical approach; best diameter: 15–55 px |

**Pilot comparison results (stored in `outputs/pilot_model_comparison_v2/`):**
- cpsam: highest cell recovery, best boundary accuracy for neurons
- StarDist: comparable on hippocampus, misses frontal cortex glia
- Mesmer: oversegments in high-density regions
- InstanSeg: experimental; promising for protein co-staining

### 6.5 Mask Expansion

After nuclear segmentation, masks are expanded 5 px (`expand_distance_px: 5`, ~0.6 µm) to capture peri-nuclear cytoplasmic transcripts. This is essential for:
- Assigning cytoplasmic mRNAs to correct cells
- Improving transcript capture per cell

Do **not** expand beyond 8 px in dense brain regions — neighboring cell territories will overlap.

### 6.6 Segmentation QC Metrics

After segmentation, review these metrics in `outputs/*/summary/`:

| Metric | Acceptable Range | Action if Outside |
|---|---|---|
| Cells per FOV | 400–2000 | Check diameter, enhancement |
| Median cell area (px²) | 300–3000 | Re-tune diameter |
| Median cell diameter (px) | 20–60 | Adjust diameter param |
| Cells < 50 px² | < 5% | Increase `min_cell_area_px` |
| Boundary Dice vs CosMx | > 0.3 | Expected — CosMx underestimates |

---

## 7. Transcript Assignment

After custom segmentation, CosMx transcripts (from `tx_file.csv.gz`) must be reassigned to the new cell masks. The native CosMx cell IDs in the expression matrix correspond to the **old** CellLabels, not your new masks.

```bash
python scripts/06_assign_transcripts.py \
    --seg_dir outputs/segmentation/cellpose/cpsam_d25/expanded/ \
    --fovs FOV00001 FOV00007 FOV00037 FOV00043
```

**Assignment method: `mask_lookup`** — for each transcript (x, y), look up the pixel value in the expanded mask. This is the most accurate method for dense spatial data.

**Quality filters applied automatically:**
- `min_qv: 20` — remove low-confidence transcript calls (Phred-scaled quality)
- Negative probe transcripts excluded (`NegPrb*`)
- FalseCode transcripts excluded (`FalseCode*`)

Output: `outputs/transcript_assignment/<seg_tag>/`

---

## 8. Building AnnData & QC

### 8.1 Build from Native CosMx (fast start)

```bash
python scripts/01_load_flatfiles.py
```

Produces `outputs/anndata/2511276KSlide116_native.h5ad` using CosMx native cell assignments. Useful for rapid exploration but cell counts are underestimated.

### 8.2 Build from Custom Segmentation (recommended for publication)

```bash
python scripts/07_build_anndata.py \
    --assign_dir outputs/transcript_assignment/cpsam_d25/ \
    --cell_tables_dir outputs/segmentation/cellpose/cpsam_d25/cell_tables/
```

### 8.3 QC Filtering

```bash
python scripts/08_qc_filter.py \
    --input outputs/anndata/2511276KSlide116_native.h5ad
```

**QC thresholds (from config):**

| Metric | Min | Max | Rationale |
|---|---|---|---|
| Transcripts per cell | 5 | 5000 | Remove empty/doublet cells |
| Genes per cell | 3 | — | Minimum for meaningful expression |
| Negative probe fraction | — | 10% | Flag high-background cells |
| Cell area | 10 µm² | 8000 µm² | Remove debris and tissue folds |

Brain cells tend to have **lower transcript counts** than epithelial tissue (200–800 transcripts/cell typical for neurons vs 1000+ in epithelium). Do not apply epithelial-tissue thresholds to brain data.

---

## 9. Clustering & Cell Type Annotation

```bash
# Normalize and cluster
python scripts/09_normalize_cluster.py \
    --input outputs/anndata/<stem>_qc.h5ad \
    --norm scran \
    --batch_key sample_id

# Annotate cell types
python scripts/10_cell_typing.py \
    --input outputs/analysis/<stem>/adata_clustered.h5ad
```

### 9.1 Normalization Recommendation

Use **scran** normalization for CosMx data — it handles the zero-inflated, sparse count distribution better than simple log1p, and is the gold standard for single-cell RNA-seq in publication.

### 9.2 Brain Cell Type Markers

The following marker genes are used in `10_cell_typing.py` for brain cell annotation:

| Cell Type | Key Markers |
|---|---|
| Neurons | *RBFOX3, MAP2, SNAP25, SYP, NEUN* |
| Astrocytes | *GFAP, AQP4, ALDH1L1, S100B* |
| Oligodendrocytes | *MBP, PLP1, MOG, OLIG2* |
| OPCs | *PDGFRA, CSPG4, NG2* |
| Microglia | *AIF1 (IBA1), CX3CR1, P2RY12, TMEM119* |
| Endothelial | *CLDN5, PECAM1, FLT1* |
| Pericytes / VSMC | *ACTA2, PDGFRB, RGS5* |
| AD-specific | *APOE, CLU, TREM2, C1Q* (complement activation in microglia) |

### 9.3 Batch Correction

With 2 slides (72 FOVs each), batch effects between slides are expected. Apply **Harmony** batch correction with `batch_key: "sample_id"`. Do not use slide ID as the sole batch key if biological groups are confounded with slides.

---

## 10. Spatial Proteomics Analysis

The protein slide (251121Slide116) provides **multiplexed immunofluorescence** for ~10–20 protein markers. This is analyzed separately from the RNA panel, then integrated at the FOV level.

### 10.1 Key Protein Markers Available

From pilot results in `results/03B_protein_brain_aware/` and `results/04_integrate_rna_protein_by_fov/`:

| Protein | Cell Type | Key Finding in Pilot |
|---|---|---|
| AIF1 (IBA1) | Microglia | Elevated in AD, correlates with RNA *AIF1* |
| CD68 | Activated microglia | Higher in AD frontal cortex |
| HLA-DR | Antigen-presenting microglia | Co-elevated with C1QA in AD |
| GFAP | Reactive astrocytes | Upregulated in AD hippocampus |
| Fibronectin | ECM / reactive glia | Elevated in AD |
| CLDN5 | BBB endothelial | Reduced in AD |
| SMA (ACTA2) | Pericytes / VSMC | Vessel remodeling marker |

### 10.2 Protein Pipeline

```bash
# Build Seurat protein object (R)
# results/02A_build_protein_seurat/

# Protein segmentation uses same morphology TIFs as RNA
# (protein and RNA slides share the same tissue sections)
python scripts/ssd_scripts/step02_enhance_images.py  # for protein slide
python scripts/03_segment_cellpose.py --model cpsam   # same parameters

# Quantify per-cell protein intensity
# → extract mean intensity per cell from each protein channel
```

### 10.3 Protein Analysis Steps

1. **Segment cells** on protein morphology images (same DAPI-based approach as RNA)
2. **Extract per-cell intensity** for each protein channel using the expanded mask
3. **Normalize**: CLR (centered log-ratio) normalization for protein data — do not use scran
4. **Cluster** protein data independently; compare clusters to RNA-based clusters
5. **Spatial visualization**: protein intensity maps overlaid on tissue coordinates

---

## 11. Spatial Transcriptomics Analysis

```bash
python scripts/11_spatial_analysis.py \
    --input outputs/analysis/<stem>/adata_annotated.h5ad
```

### 11.1 Analyses Performed

| Analysis | Tool | Purpose |
|---|---|---|
| Spatial neighbors graph | Squidpy | Basis for all spatial statistics |
| Neighborhood enrichment | Squidpy | Which cell types co-localize? |
| Ripley's L function | Squidpy | Spatial clustering vs random |
| Moran's I | Squidpy | Spatially variable genes |
| Co-expression correlation | Custom | Gene–gene spatial correlation |

### 11.2 Spatially Variable Genes

Use Moran's I to identify genes whose expression varies significantly across spatial positions:

```python
import squidpy as sq
sq.gr.spatial_neighbors(adata)
sq.gr.spatial_autocorr(adata, mode="moran")
```

For AD-specific analysis, focus on spatially variable genes in amyloid-rich regions (identified by co-registration with histology or by high local *APOE/CLU* expression).

### 11.3 FOV-Level Spatial Context

Each FOV covers approximately 0.8 × 0.8 mm of tissue. At 144 FOVs per experiment, the full slide covers a substantial portion of the brain section. When scaling to all 144 FOVs:
- FOV positions are in `fov_positions_file.csv.gz` — use these as global spatial coordinates
- Perform analyses at the **slide level** (not per-FOV) to capture long-range spatial patterns

---

## 12. RNA–Protein Integration

For publication, integrate the RNA and protein data from matched tissue sections to link transcriptomic cell identities with protein-level validation.

### 12.1 Matching Strategy

RNA and protein slides are **adjacent sections from the same tissue block**. Direct cell-level matching is not possible (different sections), but FOV-level integration is valid:

```python
# Merge at FOV level
# results/04_integrate_rna_protein_by_fov/
# Key outputs: joint_FOV_summary.tsv
```

### 12.2 Validated Correlations from Pilot

The pilot identified strong RNA–protein correlations (see `results/04_integrate_rna_protein_by_fov/scatter_*.png`):

| RNA Gene | Protein Marker | Pearson r | Cell Type |
|---|---|---|---|
| *AIF1* | AIF1 (IBA1) protein | ~0.85 | Microglia |
| *C1QA* | HLA-DR | ~0.78 | Activated microglia |
| *ACTA2* | SMA | ~0.91 | Pericytes/VSMC |
| *CLDN5* | Fibronectin | ~−0.6 | BBB integrity inverse |
| *APOE* | IL-1β | ~0.72 | AD neuroinflammation |

These correlations validate that the segmentation and assignment pipeline is producing biologically coherent results.

---

## 13. Scaling to All 144 FOVs

Once the 4-FOV pilot pipeline is validated, scale to the full dataset on the external SSD (`/mnt/external/data/`).

### 13.1 What Changes

| Step | 4-FOV Pilot | 144-FOV Full Run |
|---|---|---|
| Data source | `raw_data/pilot_4fov/` (local) | `/mnt/external/data/` (SSD) |
| Config `ssd_root` | `raw_data/pilot_4fov` | `/mnt/external/data` |
| Config `flat_dir` | `slide1_RNA/flatfiles` | `Study20251223_GOlab_cosmxrna1/flatFiles/2511276KSlide116` |
| `--fovs` arg | FOV00001 FOV00007 FOV00037 FOV00043 | omit (processes all FOVs) |
| Batch correction | Not needed (1 sample) | Required (Harmony, batch_key=sample_id) |
| Runtime | ~20 min | ~8–12 hours (GPU) |

### 13.2 Config Switch for Full Run

Edit `config/study_config.yaml`:

```yaml
paths:
  ssd_root: "/mnt/external/data"   # ← change from raw_data/pilot_4fov

  slide1:
    flat_dir:      "Study20251223_GOlab_cosmxrna1/flatFiles/2511276KSlide116"
    cell_stats_dir: "Study20251223_GOlab_cosmxrna1/DecodedFiles/2511276KSlide116/20251127_074826_S1/CellStatsDir"
    morphology2d:  "Study20251223_GOlab_cosmxrna1/DecodedFiles/2511276KSlide116/20251127_074826_S1/CellStatsDir/Morphology2D"
```

### 13.3 Batch Processing Script

```bash
# Full 144-FOV batch (requires external SSD mounted)
python scripts/ssd_scripts/03_batch_cellpose_all_144_ch2.py
```

### 13.4 Recommended Order for Full Run

```
00_check_env.py
↓
01_load_flatfiles.py          (both slides)
↓
02_image_enhance.py           (all 144 FOVs — ~4 h on GPU)
↓
03_segment_cellpose.py        (cpsam, all FOVs — ~6 h)
↓
06_assign_transcripts.py      (assign tx_file to new masks)
↓
07_build_anndata.py
↓
08_qc_filter.py
↓
09_normalize_cluster.py       (Harmony batch correction)
↓
10_cell_typing.py
↓
11_spatial_analysis.py
↓
12_compare_segmenters.py      (optional: compare cpsam vs StarDist on subset)
```

### 13.5 Parallelization

For 144 FOVs, parallelize the image enhancement and segmentation steps by FOV batch:

```bash
# Example: split into 4 batches of 36 FOVs
python scripts/02_image_enhance.py --fovs $(seq -f "FOV%05g" 1 36 | tr '\n' ' ')
python scripts/02_image_enhance.py --fovs $(seq -f "FOV%05g" 37 72 | tr '\n' ' ')
```

With a single GPU, expect ~2 minutes per FOV for enhancement + segmentation combined.

---

## 14. Key Findings from Pilot

These results from the 4-FOV pilot validate the pipeline and provide biological context for the full 144-FOV analysis.

### 14.1 Segmentation Performance

| Method | Cells/FOV (avg) | vs CosMx Native | Notes |
|---|---|---|---|
| CosMx Native | ~341 | 1× (reference) | Underestimates; fixed algorithm |
| QuPath (manual) | ~947 | 2.8× | Human expert count |
| **cpsam (TopHat r=20, d=25)** | **~1,200** | **3.5×** | **Recommended** |
| Nuclei model (d=20) | ~980 | 2.9× | More conservative |
| StarDist (scale=2.0) | ~1,050 | 3.1× | Comparable; slower |
| CellProfiler (d=15–55) | ~880 | 2.6× | Classical; reproducible |

### 14.2 Key Biological Signals Observed

- **Microglial activation in AD:** Significantly higher *AIF1*, *CD68*, *HLA-DR* (RNA and protein) in both frontal cortex and hippocampus. Strongest in hippocampus.
- **Complement cascade upregulation:** *C1QA*, *C1QB*, *C1QC* elevated in AD microglia clusters — spatially clustered near amyloid regions.
- **Astrocyte reactivity:** *GFAP*, *STAT3*, *LCN2* elevated in AD. Reactive astrocytes form spatial niches around microglial clusters.
- **BBB integrity loss:** *CLDN5* (tight junction) reduced in AD endothelial cells; inversely correlated with *Fibronectin* protein.
- **Neuronal loss in hippocampus:** Lower overall neuronal cell counts in FOV00007 (AD-H) vs FOV00043 (Control-H).

### 14.3 Pipeline Recommendations for Publication

Based on pilot analysis:

1. **Use cpsam + TopHat** — highest cell recovery, best correlation with expert QuPath count
2. **Do not use CosMx native cells** as primary analysis — publish custom segmentation
3. **Report both native and custom** results in supplementary (transparency)
4. **Transcript assignment efficiency:** target >70% transcripts assigned to cells (vs CosMx native ~55%)
5. **Use compartment information** (`CompartmentLabels`) to distinguish nuclear vs cytoplasmic transcripts for nuclear-localized RNAs

---

## 15. Publication Checklist

Before submitting, confirm the following:

### Data & Reproducibility
- [ ] All parameters documented in `config/study_config.yaml`
- [ ] Best segmentation parameters in `config/seg_best_params.json`
- [ ] `sample_manifest.csv` updated with all 144 FOV metadata
- [ ] Pipeline version-controlled (git tag at analysis freeze)
- [ ] Raw data deposited in appropriate repository (GEO / Zenodo)

### Segmentation
- [ ] cpsam with TopHat r=20 applied to all FOVs
- [ ] QC table generated for all FOVs (cell counts, sizes, area distributions)
- [ ] Visual overlays reviewed for ≥10% of FOVs
- [ ] Comparison to CosMx native segmentation provided (script `12_compare_segmenters.py`)

### Transcriptomics
- [ ] Custom transcript assignment from `tx_file.csv.gz` (not native CosMx assignment)
- [ ] Quality filter: QV ≥ 20, negative probes excluded
- [ ] scran normalization applied
- [ ] Harmony batch correction (by sample/slide)
- [ ] Cell types annotated with established brain markers
- [ ] Spatially variable genes identified (Moran's I)

### Proteomics
- [ ] Protein intensity extracted per cell per channel
- [ ] CLR normalization applied
- [ ] RNA–Protein correlation validated (see `results/04_integrate_rna_protein_by_fov/`)

### Figures
- [ ] UMAP colored by cell type, condition, FOV
- [ ] Spatial maps: cell type on tissue coordinates
- [ ] Differential expression: AD vs Control per cell type (volcano plots)
- [ ] Neighborhood enrichment heatmap (squidpy)
- [ ] RNA–Protein scatter plots for key markers

---

## Contact & Reference

**Laboratory:** GO Laboratory, Kyoto University
**Study:** Dementia CosMx 2026 — Spatial Multi-omics of Human AD Brain
**Data format:** NanoString CosMx SMI (FlatFile export)
**Pipeline:** Custom Python (Cellpose + Squidpy) + R (Seurat)

For pipeline questions, refer to script docstrings (`head -20 scripts/<script>.py`) and the configuration file `config/study_config.yaml`.
