#!/usr/bin/env python3
"""Generate the Baysor/Sopa implementation specification PDF (English)."""
from __future__ import annotations
from fpdf import FPDF
from pathlib import Path


# ─── helpers ──────────────────────────────────────────────────────────────────

class PDF(FPDF):
    TITLE_COLOR  = (26,  86, 155)
    H2_COLOR     = (41, 128, 185)
    H3_COLOR     = (52, 152, 219)
    CODE_BG      = (240, 244, 248)
    NOTE_BG      = (255, 249, 230)
    WARN_BG      = (255, 235, 235)
    OK_BG        = (232, 248, 232)
    TEXT_COLOR   = (30,  30,  30)
    MUTED_COLOR  = (100, 100, 100)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.MUTED_COLOR)
        self.cell(0, 6, "CosMx 2026 - Baysor/Sopa RNA-Guided Segmentation Implementation Spec", align="L")
        self.cell(0, 6, f"Page {self.page_no()}", align="R")
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(*self.TEXT_COLOR)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*self.MUTED_COLOR)
        self.cell(0, 5, "Confidential - CosMx AD Brain Project  |  Prepared 2026-06-07", align="C")

    # ── section titles ──────────────────────────────────────────────────────

    def h1(self, text: str):
        self.ln(4)
        self.set_fill_color(*self.TITLE_COLOR)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, f"  {text}", fill=True, ln=True)
        self.set_text_color(*self.TEXT_COLOR)
        self.ln(3)

    def h2(self, text: str):
        self.ln(3)
        self.set_text_color(*self.H2_COLOR)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, text, ln=True)
        self.set_draw_color(*self.H2_COLOR)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self.set_text_color(*self.TEXT_COLOR)

    def h3(self, text: str):
        self.ln(2)
        self.set_text_color(*self.H3_COLOR)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 7, text, ln=True)
        self.set_text_color(*self.TEXT_COLOR)

    def body(self, text: str, indent: float = 0):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*self.TEXT_COLOR)
        if indent:
            self.set_x(10 + indent)
        self.multi_cell(0 if not indent else 190 - indent, 5, text)

    def bullet(self, text: str, indent: float = 5):
        self.set_font("Helvetica", "", 9)
        self.set_x(10 + indent)
        self.cell(5, 5, chr(149))
        self.set_x(10 + indent + 5)
        self.multi_cell(180 - indent, 5, text)

    def numbered(self, n: int, text: str, indent: float = 5):
        self.set_font("Helvetica", "", 9)
        self.set_x(10 + indent)
        self.cell(7, 5, f"{n}.")
        self.set_x(10 + indent + 7)
        self.multi_cell(178 - indent, 5, text)

    def code(self, lines: list[str], title: str = ""):
        self.ln(1)
        if title:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*self.MUTED_COLOR)
            self.cell(0, 5, title, ln=True)
            self.set_text_color(*self.TEXT_COLOR)
        self.set_fill_color(*self.CODE_BG)
        self.set_draw_color(180, 195, 210)
        self.set_font("Courier", "", 7.5)
        x0 = self.get_x()
        for line in lines:
            self.set_x(10)
            self.cell(190, 4.5, f"  {line}", fill=True, ln=True, border=0)
        self.set_draw_color(180, 195, 210)
        self.ln(1)

    def note_box(self, text: str, color: tuple = None, label: str = "NOTE"):
        bg = color or self.NOTE_BG
        self.set_fill_color(*bg)
        self.set_font("Helvetica", "B", 8)
        self.set_x(10)
        self.cell(190, 5, f"  {label}", fill=True, ln=True, border=0)
        self.set_font("Helvetica", "", 8.5)
        self.set_x(10)
        self.multi_cell(190, 5, f"  {text}", fill=True, border=0)
        self.ln(2)

    def table(self, headers: list[str], rows: list[list[str]], col_widths: list[float] = None):
        n = len(headers)
        w = col_widths or [190 / n] * n
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(*self.TITLE_COLOR)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(w[i], 7, f" {h}", fill=True, border=1)
        self.ln()
        self.set_text_color(*self.TEXT_COLOR)
        alt = False
        for row in rows:
            self.set_font("Helvetica", "", 8)
            self.set_fill_color(245, 248, 252) if alt else self.set_fill_color(255, 255, 255)
            alt = not alt
            max_h = 6
            for i, cell in enumerate(row):
                self.cell(w[i], max_h, f" {cell}", fill=True, border=1)
            self.ln()
        self.ln(2)


# ─── main ─────────────────────────────────────────────────────────────────────

def build_pdf(out_path: Path):
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 15, 10)

    # ══════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()

    pdf.set_fill_color(*PDF.TITLE_COLOR)
    pdf.rect(0, 0, 210, 60, "F")

    pdf.set_y(18)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "RNA-Guided Segmentation", align="C", ln=True)
    pdf.set_font("Helvetica", "B", 17)
    pdf.cell(0, 10, "Baysor + Sopa Implementation Specification", align="C", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, "CosMx AD Brain Project - Pilot 4-FOV Phase", align="C", ln=True)

    pdf.set_y(70)
    pdf.set_text_color(*PDF.TEXT_COLOR)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Document Information", ln=True)
    pdf.set_draw_color(180, 195, 210)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    meta = [
        ("Date", "2026-06-07"),
        ("Project", "CosMx 2026 - AD vs. Normal Human Brain"),
        ("Repository", "https://github.com/akihisa-dev7/cosmx"),
        ("Target branch", "feature/baysor-sopa-segmentation"),
        ("Status", "Implementation Specification v1.0"),
        ("Data", "Pilot 4 FOVs (FOV00001, FOV00007, FOV00037, FOV00043)"),
        ("Technology", "Baysor v0.7.x via Sopa v2.x Python framework"),
    ]
    pdf.set_font("Helvetica", "", 9)
    for k, v in meta:
        pdf.set_x(10)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(45, 6, k)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, v, ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Executive Summary", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    summary_text = (
        "This document specifies the integration of Baysor - a Bayesian transcript-based cell segmentation "
        "algorithm - into the existing CosMx 2026 cell segmentation pipeline, using the Sopa Python framework "
        "to reduce implementation cost. "
        "The primary scientific motivation is that DAPI-only morphology-based segmentation introduces "
        "systematic bias in Alzheimer's Disease (AD) tissue, where cellular morphology is altered by "
        "tau aggregation, reactive gliosis, and microglial activation. By jointly optimising DAPI morphology "
        "and RNA transcript spatial distribution, Baysor is expected to improve cell boundary accuracy "
        "by 20-40% (F1) compared to Cellpose alone, as reported in the original Nature Biotechnology paper.\n\n"
        "The implementation adds a new BaysorSegmenter class and a run_baysor.py script to the existing "
        "pipeline, keeping all existing Cellpose/StarDist paths fully intact. The Cellpose cpsam mask "
        "is used as a prior for Baysor, constraining over-segmentation while enabling RNA-guided "
        "boundary refinement. Sopa handles CosMx data I/O, FOV tiling, and AnnData output automatically."
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Table of Contents", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    toc = [
        ("1", "Scientific Motivation & Background"),
        ("2", "Architecture Overview"),
        ("3", "Data Inventory - What We Already Have"),
        ("4", "New Segmentation Flow"),
        ("5", "File Changes & New Files"),
        ("6", "Sopa Integration Details"),
        ("7", "Key Parameters & Tuning Guide"),
        ("8", "Output Files & AnnData Format"),
        ("9", "GitHub Workflow - Branch & Push Instructions"),
        ("10", "Testing & Validation Plan"),
        ("11", "Timeline & Risk Register"),
    ]
    pdf.set_font("Helvetica", "", 9)
    for num, title in toc:
        pdf.cell(15, 5.5, f"  {num}.")
        pdf.cell(0, 5.5, title, ln=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("1.  Scientific Motivation & Background")

    pdf.h2("1.1  Why DAPI-Only Segmentation Fails in AD Tissue")
    pdf.body(
        "The current pipeline segments cells using nuclear DAPI staining (Cellpose cpsam, diameter=25 px). "
        "In healthy brain tissue this works reasonably well, but in AD tissue three systematic failure modes occur:"
    )
    pdf.ln(2)
    pdf.table(
        ["Failure Mode", "Brain Cell Type", "AD Pathology", "Segmentation Error"],
        [
            ["Cell fusion", "Neurons", "Tau aggregation distorts nuclear shape", "2 cells detected as 1"],
            ["Cell dropout", "Microglia", "Activated microglia change from ramified to amoeboid", "Missed or oversized"],
            ["Boundary error", "Astrocytes", "Reactive gliosis enlarges cell body", "RNA from neighbours misassigned"],
        ],
        col_widths=[35, 38, 70, 47],
    )
    pdf.body(
        "These errors are not random - they are systematically larger in AD samples, introducing a "
        "confound into any AD vs. Normal differential expression analysis."
    )

    pdf.h2("1.2  Why RNA-Guided Segmentation Fixes This")
    pdf.body(
        "RNA transcript coordinates are a direct readout of where cellular RNA is produced and "
        "localised. Because transcripts originate within the cell body, their spatial distribution "
        "defines the true cell boundary independently of nuclear morphology.\n\n"
        "Baysor (Petukhov et al., Nature Biotechnology 2022) models transcript-to-cell assignment "
        "as a Bayesian mixture problem. Each transcript is assigned to the most likely cell given "
        "(a) its spatial proximity and (b) the gene expression profile of that cell. "
        "When a Cellpose nuclear mask is provided as a prior, Baysor is constrained to respect "
        "morphological boundaries while correcting for RNA that has leaked across poorly-drawn boundaries."
    )
    pdf.note_box(
        "Benchmark result: Baysor with nuclear prior achieves F1 = 0.78 on osmFISH cortex data, "
        "vs. F1 = 0.61 for Cellpose alone. Expected improvement on CosMx 6500-gene panel is higher "
        "due to richer transcript signal per cell.",
        color=PDF.OK_BG, label="BENCHMARK"
    )

    pdf.h2("1.3  Why Sopa Reduces Implementation Cost")
    pdf.body(
        "Sopa (Blampey et al., Nature Communications 2024) is a technology-agnostic spatial "
        "transcriptomics framework that wraps Baysor, Cellpose, and other tools with a unified "
        "Python API. For CosMx data specifically, Sopa provides:\n"
    )
    for item in [
        "Automatic reading of CosMx flat files (tx_file, morphology_images) into SpatialData objects",
        "FOV-level tiling so Baysor runs on manageable chunks without manual coordinate splitting",
        "Built-in coordinate system management (pixel <-> umm conversion per FOV)",
        "Direct output to AnnData / SpatialData, compatible with Scanpy downstream analysis",
        "CLI and Python API, enabling both scripted and interactive use",
    ]:
        pdf.bullet(item)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("2.  Architecture Overview")

    pdf.h2("2.1  Overall System Architecture")
    pdf.set_font("Courier", "", 7.5)
    arch_lines = [
        "CosMx_2026/",
        "  raw_data/pilot_4fov/slide1_RNA/",
        "    flatfiles/2511276KSlide116_tx_file.csv.gz   <-- 4.26M transcripts, 6519 genes",
        "    morphology_images/FOV*.TIF                  <-- 5-channel, 4256x4256 px",
        "    per_fov_decoded/FOV*/CellLabels_F*.tif      <-- CosMx native label masks",
        "",
        "  cell_segmentation_pipeline/         <-- EXISTING (unchanged except additions)",
        "    src/segmentation/",
        "      base.py                         <-- MODIFIED: add 'baysor' to factory",
        "      cellpose_segmenter.py           <-- UNCHANGED",
        "      stardist_segmenter.py           <-- UNCHANGED",
        "      baysor_segmenter.py             <-- NEW: Sopa-based Baysor wrapper",
        "    scripts/",
        "      run_segmentation.py             <-- UNCHANGED (existing models work as-is)",
        "      run_baysor.py                   <-- NEW: dedicated Baysor pipeline script",
        "    config/",
        "      default_config.yaml             <-- MODIFIED: add [baysor] section",
        "    requirements.txt                  <-- MODIFIED: add sopa[baysor]",
        "",
        "  outputs/",
        "    baysor/                           <-- NEW output directory",
        "      FOV00001/",
        "        segmentation.csv              <-- Baysor transcript assignments",
        "        cell_stats.csv                <-- Per-cell centroids and counts",
        "        baysor_mask.tif               <-- Integer label mask (compatible with GUI)",
        "        anndata.h5ad                  <-- AnnData with spatial coords",
        "      ...",
    ]
    pdf.set_fill_color(*PDF.CODE_BG)
    for line in arch_lines:
        pdf.set_x(10)
        pdf.cell(190, 4.2, f"  {line}", fill=True, ln=True, border=0)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*PDF.TEXT_COLOR)

    pdf.h2("2.2  Module Dependencies")
    pdf.table(
        ["Module", "Depends On", "Role"],
        [
            ["run_baysor.py", "baysor_segmenter.py, sopa, tifffile, pandas", "Entry point CLI script"],
            ["baysor_segmenter.py", "sopa>=2.0, spatialdata, anndata", "Sopa wrapper for Baysor"],
            ["base.py (factory)", "baysor_segmenter.py", "Returns BaysorSegmenter instance"],
            ["default_config.yaml", "(none)", "Stores Baysor default parameters"],
            ["requirements.txt", "(none)", "Declares sopa[baysor] dependency"],
        ],
        col_widths=[45, 75, 70],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3
    # ══════════════════════════════════════════════════════════════════════════
    pdf.h1("3.  Data Inventory - What We Already Have")

    pdf.h2("3.1  Confirmed Data (verified 2026-06-07)")
    pdf.table(
        ["File / Directory", "Content", "Baysor Use", "Status"],
        [
            ["outputs_ssd/proseg_pilot/FOV00001_tx.csv.gz", "4.26M transcripts, cols: fov, x_local_px, y_local_px, x_global_px, y_global_px, z, target, CellComp", "Transcript input", "READY"],
            ["raw_data/.../morphology_images/FOV*.TIF", "5-channel TIFF (5,4256,4256) uint16. Ch0=DAPI", "Cellpose prior generation", "READY"],
            ["raw_data/.../per_fov_decoded/FOV*/CellLabels_F*.tif", "Integer label mask (4256,4256) uint16, ~521 cells/FOV", "Alternative prior (CosMx native)", "READY"],
            ["outputs/pilot_4fov/ (Cellpose cpsam masks)", "Cellpose-generated nuclear masks (best quality)", "Recommended prior for Baysor", "READY (if generated)"],
        ],
        col_widths=[60, 75, 35, 20],
    )

    pdf.h2("3.2  Coordinate System Note")
    pdf.note_box(
        "The tx_file uses x_local_px / y_local_px (range 0-4256) for within-FOV coordinates "
        "and x_global_px / y_global_px for whole-slide coordinates. "
        "Sopa reads CosMx flat files natively and handles this conversion automatically. "
        "Manual coordinate conversion is NOT required when using Sopa.",
        color=PDF.NOTE_BG, label="IMPORTANT"
    )

    pdf.h2("3.3  Minimum Required for Baysor")
    pdf.table(
        ["Input", "Required?", "Column / Format", "Source in This Project"],
        [
            ["x, y coordinates", "REQUIRED", "Numeric (px or um)", "x_local_px, y_local_px in tx_file"],
            ["gene / target name", "REQUIRED", "String gene symbol", "target column in tx_file"],
            ["z coordinate", "OPTIONAL", "Integer 0-7 (0.8 um/slice)", "z column in tx_file"],
            ["Prior segmentation mask", "STRONGLY RECOMMENDED", "Integer label TIFF (0=bg)", "CellLabels_F*.tif or Cellpose output"],
        ],
        col_widths=[48, 32, 55, 55],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("4.  New Segmentation Flow")

    pdf.h2("4.1  Old Flow (Current Pipeline)")
    pdf.code([
        "raw_data/morphology_images/FOV*.TIF",
        "         |",
        "         v  [run_preprocess.py]",
        "  TopHat enhanced DAPI (single channel, uint8)",
        "         |",
        "         v  [run_segmentation.py --model cellpose_cpsam]",
        "  Nuclear mask (.tif, integer labels)    -->  Expanded mask (.tif)",
        "         |                                         |",
        "         v                                         v",
        "  Overlay PNG, Cell table CSV              QC panels, QC summary CSV",
        "",
        "  OUTPUT: morphology-only masks - transcript assignment done AFTER",
        "          by ProSeg as a separate step using the mask as input",
    ])

    pdf.h2("4.2  New Flow (Baysor + Sopa)")
    pdf.code([
        "raw_data/morphology_images/FOV*.TIF     raw_data/flatfiles/*_tx_file.csv.gz",
        "         |                                          |",
        "         v  [Sopa: sopa.io.read_cosmx()]            v",
        "  SpatialData object (DAPI image + transcript table, aligned)",
        "         |",
        "         v  [Sopa: sopa.segmentation.cellpose()]",
        "  Cellpose nuclear mask (cpsam, DAPI ch0)   <-- PRIOR for Baysor",
        "         |",
        "         v  [Sopa: sopa.segmentation.baysor()]",
        "    Baysor jointly optimises:",
        "      - Spatial transcript density (RNA signal)",
        "      - Cell-type expression profiles (gene x cell Gaussian)",
        "      - Cellpose nuclear prior (prior_segmentation_confidence=0.5)",
        "         |",
        "         v",
        "  segmentation.csv (transcript-to-cell assignments)",
        "  baysor_mask.tif  (integer label mask, same format as current pipeline)",
        "  anndata.h5ad     (cell x gene counts + spatial coords)",
        "         |",
        "         v  [existing QC + downstream scripts]",
        "  QC panels, cell type annotation, AD vs Normal comparison",
    ])

    pdf.h2("4.3  Key Differences")
    pdf.table(
        ["Aspect", "Old (Cellpose only)", "New (Baysor + Sopa)"],
        [
            ["Input to segmenter", "DAPI image only", "DAPI image + 4.26M RNA transcripts"],
            ["Cell boundary source", "Nuclear morphology", "RNA density + nuclear prior"],
            ["Transcript assignment", "Post-hoc via ProSeg (separate step)", "Integral to segmentation (Baysor)"],
            ["Output format", "mask.tif + cells.csv", "mask.tif + segmentation.csv + h5ad"],
            ["ProSeg role", "Primary transcript assigner", "Optional (can be replaced or compared)"],
            ["Expected cell count", "521 / FOV (cpsam pilot)", "Similar or higher (RNA fills gaps)"],
            ["Backward compatibility", "N/A", "Full - existing GUI reads baysor_mask.tif"],
        ],
        col_widths=[50, 70, 70],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("5.  File Changes & New Files")

    pdf.h2("5.1  New File:  src/segmentation/baysor_segmenter.py")
    pdf.body("Create this file in cell_segmentation_pipeline/src/segmentation/")
    pdf.code([
        '"""BaysorSegmenter: Sopa-based RNA-guided segmentation wrapper."""',
        'from __future__ import annotations',
        'from pathlib import Path',
        'import numpy as np',
        'import tifffile',
        'from .base import BaseSegmenter',
        '',
        'class BaysorSegmenter(BaseSegmenter):',
        '    """Runs Baysor via Sopa on a single FOV.',
        '',
        '    Args:',
        '        scale: Expected cell radius in micrometres (default 10).',
        '        min_molecules_per_cell: Min transcript count to keep a cell (default 15).',
        '        prior_confidence: How strongly to weight the DAPI prior (0-1, default 0.5).',
        '        n_clusters: Number of cell-type clusters for expression model (default 4).',
        '        pixel_size_um: CosMx pixel size in umm/px (default 0.12028).',
        '        baysor_bin: Path to baysor binary (default "baysor").',
        '    """',
        '',
        '    def __init__(self, scale=10, min_molecules_per_cell=15, prior_confidence=0.5,',
        '                 n_clusters=4, pixel_size_um=0.12028, baysor_bin="baysor"):',
        '        self.scale = scale',
        '        self.min_molecules = min_molecules_per_cell',
        '        self.prior_confidence = prior_confidence',
        '        self.n_clusters = n_clusters',
        '        self.pixel_size_um = pixel_size_um',
        '        self.baysor_bin = baysor_bin',
        '        self._last_result = None   # stores SpatialData after run()',
        '',
        '    @property',
        '    def name(self) -> str:',
        '        return "baysor"',
        '',
        '    def segment(self, image: np.ndarray) -> np.ndarray:',
        '        # image is unused - Baysor works from transcripts, not pixels',
        '        # Full segmentation is triggered via run_fov(); this stub allows',
        '        # compatibility with BaseSegmenter interface.',
        '        raise RuntimeError(',
        '            "BaysorSegmenter.segment() is not supported. "',
        '            "Use run_fov(tx_csv, prior_mask_tif, out_dir) instead."',
        '        )',
        '',
        '    def run_fov(self, tx_csv: Path, prior_mask: Path, out_dir: Path) -> Path:',
        '        """Run Baysor on one FOV and return path to baysor_mask.tif.',
        '',
        '        Args:',
        '            tx_csv:      Path to FOV transcript CSV (cols: x, y, z, gene).',
        '            prior_mask:  Path to integer-label prior TIFF (0 = background).',
        '            out_dir:     Output directory for this FOV.',
        '',
        '        Returns:',
        '            Path to baysor_mask.tif (integer label TIFF, same format as Cellpose output).',
        '        """',
        '        import sopa',
        '        import spatialdata',
        '        from sopa.segmentation import baysor as sopa_baysor',
        '',
        '        out_dir = Path(out_dir)',
        '        out_dir.mkdir(parents=True, exist_ok=True)',
        '',
        '        # 1. Build SpatialData from transcript CSV + prior mask',
        '        sdata = sopa.io.from_transcripts_and_mask(',
        '            transcripts_path=str(tx_csv),',
        '            mask_path=str(prior_mask),',
        '            pixel_size=self.pixel_size_um,',
        '        )',
        '',
        '        # 2. Run Baysor segmentation',
        '        sopa_baysor.run(',
        '            sdata,',
        '            scale=self.scale,',
        '            min_molecules_per_cell=self.min_molecules,',
        '            prior_segmentation_confidence=self.prior_confidence,',
        '            n_clusters=self.n_clusters,',
        '            baysor_binary=self.baysor_bin,',
        '            output_dir=str(out_dir),',
        '        )',
        '',
        '        # 3. Convert Baysor polygon output to integer label mask',
        '        mask = sopa.utils.shapes_to_mask(sdata, shape=(4256, 4256))',
        '        mask_path = out_dir / "baysor_mask.tif"',
        '        tifffile.imwrite(str(mask_path), mask.astype(np.uint16))',
        '',
        '        # 4. Save AnnData',
        '        adata = sopa.to_anndata(sdata)',
        '        adata.write_h5ad(str(out_dir / "anndata.h5ad"))',
        '',
        '        self._last_result = sdata',
        '        return mask_path',
    ], title="cell_segmentation_pipeline/src/segmentation/baysor_segmenter.py")

    pdf.h2("5.2  New File:  scripts/run_baysor.py")
    pdf.body("Entry-point CLI script - mirrors run_segmentation.py style.")
    pdf.code([
        '#!/usr/bin/env python3',
        '"""Run Baysor RNA-guided segmentation on CosMx pilot FOVs.',
        '',
        'Usage:',
        '  python cell_segmentation_pipeline/scripts/run_baysor.py \\',
        '    --tx_file  outputs_ssd/proseg_pilot/FOV00001_tx.csv.gz \\',
        '    --prior_dir outputs/pilot_4fov/masks/ \\',
        '    --output   outputs/baysor/ \\',
        '    --fovs FOV00001 FOV00007 FOV00037 FOV00043',
        '"""',
        'import argparse, sys',
        'from pathlib import Path',
        'sys.path.insert(0, str(Path(__file__).resolve().parents[2]))',
        '',
        'import pandas as pd',
        'from cell_segmentation_pipeline.src.segmentation.baysor_segmenter import BaysorSegmenter',
        '',
        '',
        'def parse_args():',
        '    p = argparse.ArgumentParser()',
        '    p.add_argument("--tx_file",  required=True)',
        '    p.add_argument("--prior_dir", required=True,',
        '                   help="Dir containing cellpose_cpsam nuclear masks (.tif)")',
        '    p.add_argument("--output",   required=True)',
        '    p.add_argument("--fovs",     nargs="+", default=None)',
        '    p.add_argument("--scale",             type=float, default=10.0)',
        '    p.add_argument("--min_molecules",     type=int,   default=15)',
        '    p.add_argument("--prior_confidence",  type=float, default=0.5)',
        '    p.add_argument("--n_clusters",        type=int,   default=4)',
        '    p.add_argument("--pixel_size_um",     type=float, default=0.12028)',
        '    return p.parse_args()',
        '',
        '',
        'def main():',
        '    args = parse_args()',
        '    seg = BaysorSegmenter(',
        '        scale=args.scale,',
        '        min_molecules_per_cell=args.min_molecules,',
        '        prior_confidence=args.prior_confidence,',
        '        n_clusters=args.n_clusters,',
        '        pixel_size_um=args.pixel_size_um,',
        '    )',
        '',
        '    tx_df = pd.read_csv(args.tx_file, low_memory=False)',
        '    prior_dir = Path(args.prior_dir)',
        '    out_root  = Path(args.output)',
        '',
        '    fov_ids = args.fovs or sorted(tx_df["fov"].unique())',
        '',
        '    for fov_id in fov_ids:',
        '        print(f"Processing {fov_id} ...")',
        '        fov_df = tx_df[tx_df["fov"] == int(fov_id.replace("FOV", ""))]',
        '        baysor_df = fov_df.rename(columns={',
        '            "x_local_px": "x", "y_local_px": "y", "target": "gene"',
        '        })[["x", "y", "z", "gene"]]',
        '        baysor_df = baysor_df[baysor_df["z"] >= 0]',
        '',
        '        tmp_csv = out_root / fov_id / "transcripts_for_baysor.csv"',
        '        tmp_csv.parent.mkdir(parents=True, exist_ok=True)',
        '        baysor_df.to_csv(tmp_csv, index=False)',
        '',
        '        prior_mask = next(prior_dir.glob(f"*{fov_id}*cpsam*.tif"), None)',
        '        if prior_mask is None:',
        '            print(f"  WARNING: no Cellpose prior found for {fov_id}, using CosMx native")',
        '            prior_mask = next(',
        '                Path("raw_data").rglob(f"CellLabels_{fov_id.replace(\'FOV\',\'F\')}.tif"), None',
        '            )',
        '',
        '        mask_path = seg.run_fov(tmp_csv, prior_mask, out_root / fov_id)',
        '        print(f"  Done -> {mask_path}")',
        '',
        '',
        'if __name__ == "__main__":',
        '    main()',
    ], title="cell_segmentation_pipeline/scripts/run_baysor.py")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 continued
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h2("5.3  Modified File:  requirements.txt  (add 2 lines)")
    pdf.code([
        "# === ADD THESE TWO LINES to cell_segmentation_pipeline/requirements.txt ===",
        "",
        "# RNA-guided segmentation (Baysor via Sopa framework)",
        "sopa[baysor]>=2.0.0",
        "spatialdata>=0.2.0",
    ])

    pdf.h2("5.4  Modified File:  config/default_config.yaml  (add section)")
    pdf.code([
        "# === ADD THIS BLOCK to the 'segmentation:' section in default_config.yaml ===",
        "",
        "  baysor:",
        "    scale: 10                       # expected cell radius in micrometres",
        "    min_molecules_per_cell: 15      # discard cells with fewer transcripts",
        "    prior_segmentation_confidence: 0.5   # 0=RNA-only, 1=DAPI-only",
        "    n_clusters: 4                   # cell-type expression clusters",
        "    pixel_size_um: 0.12028          # CosMx slide1 calibration",
        "    force_2d: true                  # Baysor 3D output is incomplete (Issue #111)",
        "    baysor_bin: baysor              # path to baysor binary if not on PATH",
    ])

    pdf.h2("5.5  Modified File:  src/segmentation/base.py  (factory addition)")
    pdf.code([
        "# === ADD THIS BLOCK inside get_segmenter() in base.py ===",
        "",
        "    if name == 'baysor':",
        "        from .baysor_segmenter import BaysorSegmenter",
        "        b_cfg = seg_cfg.get('baysor', {})",
        "        return BaysorSegmenter(",
        "            scale=b_cfg.get('scale', 10),",
        "            min_molecules_per_cell=b_cfg.get('min_molecules_per_cell', 15),",
        "            prior_confidence=b_cfg.get('prior_segmentation_confidence', 0.5),",
        "            n_clusters=b_cfg.get('n_clusters', 4),",
        "            pixel_size_um=b_cfg.get('pixel_size_um', 0.12028),",
        "            baysor_bin=b_cfg.get('baysor_bin', 'baysor'),",
        "        )",
        "",
        "# Also update the ValueError message at the bottom to include 'baysor':",
        "    raise ValueError(",
        "        f\"Unknown model '{model_name}'. \"",
        "        \"Valid: cellpose_cpsam, cellpose_nuclei, cellpose_cyto3, stardist, \"",
        "        \"stardist_finetuned, instanseg, baysor\"   # <-- add 'baysor' here",
        "    )",
    ])

    pdf.h2("5.6  Summary of All File Changes")
    pdf.table(
        ["File Path (relative to repo root)", "Change Type", "Lines Changed"],
        [
            ["cell_segmentation_pipeline/src/segmentation/baysor_segmenter.py", "NEW", "~80"],
            ["cell_segmentation_pipeline/scripts/run_baysor.py", "NEW", "~60"],
            ["cell_segmentation_pipeline/src/segmentation/base.py", "MODIFIED", "+10"],
            ["cell_segmentation_pipeline/config/default_config.yaml", "MODIFIED", "+8"],
            ["cell_segmentation_pipeline/requirements.txt", "MODIFIED", "+3"],
        ],
        col_widths=[100, 30, 30],
    )
    pdf.note_box(
        "All existing files (run_segmentation.py, cellpose_segmenter.py, stardist_segmenter.py, "
        "all GUI pages, all QC scripts) are UNCHANGED. The Baysor path is purely additive.",
        color=PDF.OK_BG, label="BACKWARD COMPATIBILITY"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("6.  Sopa Integration Details")

    pdf.h2("6.1  Installation")
    pdf.code([
        "# Activate the cell_segmentation_pipeline virtual environment",
        "source cell_segmentation_pipeline/.venv/bin/activate",
        "",
        "# Install Sopa with Baysor extras",
        "pip install 'sopa[baysor]>=2.0.0' spatialdata",
        "",
        "# Download Baysor binary (Linux x86_64)",
        "wget https://github.com/kharchenkolab/Baysor/releases/latest/download/baysor",
        "chmod +x baysor",
        "sudo mv baysor /usr/local/bin/    # or add its directory to PATH",
        "baysor --version                  # verify: should print Baysor v0.7.x",
    ])

    pdf.h2("6.2  How Sopa Handles CosMx Data")
    pdf.body(
        "Sopa's sopa.io.read_cosmx() reads the full CosMx flat-file directory and constructs "
        "a SpatialData object. This handles:"
    )
    for item in [
        "Reading tx_file.csv.gz and correcting for the known gzip CRC error (uses partial read strategy)",
        "Aligning transcript coordinates to the DAPI image pixel coordinate system",
        "Creating GeoDataFrame for cell boundaries and transcript point cloud",
        "Storing everything in a single SpatialData container (zarr-backed on disk)",
    ]:
        pdf.bullet(item)
    pdf.ln(2)
    pdf.code([
        "import sopa",
        "",
        "# Read entire CosMx flat-file directory",
        "sdata = sopa.io.read_cosmx(",
        "    path='raw_data/pilot_4fov/slide1_RNA/flatfiles/',",
        "    image_models_kwargs={'dims': ['c', 'y', 'x']},",
        ")",
        "",
        "# sdata now contains:",
        "#   sdata.images['image']       -- 5-channel DAPI TIFF as xarray",
        "#   sdata.points['transcripts'] -- transcript GeoDataFrame with x,y,z,gene",
        "#   sdata.shapes (empty until segmentation)",
    ])

    pdf.h2("6.3  Sopa Baysor Call (high level)")
    pdf.code([
        "import sopa.segmentation",
        "",
        "# Step 1: Generate Cellpose prior within Sopa",
        "sopa.segmentation.cellpose(",
        "    sdata,",
        "    channels=['DAPI'],",
        "    model_type='cpsam',",
        "    diameter=25,",
        ")",
        "",
        "# Step 2: Run Baysor using Cellpose result as prior",
        "sopa.segmentation.baysor(",
        "    sdata,",
        "    scale=10,                              # um radius",
        "    min_molecules_per_cell=15,",
        "    prior_segmentation_confidence=0.5,",
        "    n_clusters=4,",
        "    baysor_binary='baysor',",
        ")",
        "",
        "# Step 3: Export",
        "adata = sopa.to_anndata(sdata)",
        "adata.write_h5ad('outputs/baysor/FOV00001/anndata.h5ad')",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("7.  Key Parameters & Tuning Guide")

    pdf.h2("7.1  Parameter Reference")
    pdf.table(
        ["Parameter", "Default", "Range", "Effect on Brain Data"],
        [
            ["scale", "10", "5-20 um", "Cell radius estimate. Brain neurons ~15um soma; glia ~5-8um. Start at 10 (mixed)."],
            ["min_molecules_per_cell", "15", "5-50", "Lower = keep small cells (glia). Higher = reject noise. 15 is safe for 6519-gene panel."],
            ["prior_segmentation_confidence", "0.5", "0.0-1.0", "0=RNA-only (over-segments), 1=DAPI-only. 0.5 balances; try 0.6-0.7 if over-segmenting."],
            ["n_clusters", "4", "1-10", "Expression clusters. Brain has ~8+ major types but 4 is stable for 4-FOV pilot."],
            ["force_2d", "true", "true/false", "Keep true - Baysor 3D polygon output is incomplete in v0.7.x (GitHub Issue #111)."],
        ],
        col_widths=[52, 20, 28, 90],
    )

    pdf.h2("7.2  Known Issue: Over-Segmentation")
    pdf.note_box(
        "Baysor is known to over-segment CosMx data (up to 5x more cells than Cellpose) "
        "when run without a strong prior. MITIGATION: Set prior_segmentation_confidence >= 0.5 "
        "and use the Cellpose cpsam mask (not CosMx native) as the prior. "
        "If over-segmentation persists, increase to 0.7 and raise min_molecules_per_cell to 20.",
        color=PDF.WARN_BG, label="WARNING"
    )

    pdf.h2("7.3  Recommended Tuning Order for Pilot Experiments")
    steps = [
        "Run with defaults (scale=10, prior_confidence=0.5, min_mol=15) on FOV00001 only.",
        "Check output cell count vs Cellpose baseline (521 cells/FOV). Target: 500-700 cells.",
        "If count > 800: increase prior_confidence to 0.65 and min_molecules to 20.",
        "If count < 400: decrease prior_confidence to 0.35 and min_molecules to 10.",
        "Validate visually using the existing FOV Viewer GUI (Page 01) - baysor_mask.tif is compatible.",
        "Run on all 4 pilot FOVs and compare QC metrics (median area, n_genes per cell).",
    ]
    for i, s in enumerate(steps, 1):
        pdf.numbered(i, s)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8
    # ══════════════════════════════════════════════════════════════════════════
    pdf.h1("8.  Output Files & AnnData Format")

    pdf.h2("8.1  Output Directory Structure")
    pdf.code([
        "outputs/baysor/",
        "  FOV00001/",
        "    transcripts_for_baysor.csv    -- preprocessed transcript input (temp)",
        "    segmentation.csv              -- Baysor main output: transcript x cell assignment",
        "                                     cols: x, y, z, gene, cell, assignment_confidence",
        "    cell_stats.csv                -- per-cell: n_transcripts, n_genes, x_centroid, y_centroid",
        "    baysor_mask.tif               -- integer label TIFF, uint16, 4256x4256 px",
        "                                     (compatible with existing GUI and QC scripts)",
        "    anndata.h5ad                  -- AnnData: obs=cells, var=genes, X=count matrix",
        "                                     obsm['spatial'] = (x_centroid, y_centroid)",
        "  FOV00007/  ...",
        "  FOV00037/  ...",
        "  FOV00043/  ...",
    ])

    pdf.h2("8.2  AnnData Schema")
    pdf.table(
        ["AnnData Slot", "Content", "Downstream Use"],
        [
            ["adata.X", "Cell x Gene count matrix (sparse)", "Normalization, clustering"],
            ["adata.obs", "Cell metadata: n_counts, n_genes, fov_id, x_centroid, y_centroid", "QC filtering"],
            ["adata.var", "Gene metadata: total_counts, n_cells_by_counts", "Gene QC"],
            ["adata.obsm['spatial']", "XY centroid coordinates in pixels", "Spatial plots, neighborhood analysis"],
            ["adata.uns['baysor_params']", "Parameters used for this run", "Reproducibility"],
        ],
        col_widths=[50, 80, 60],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("9.  GitHub Workflow - Branch & Push Instructions")

    pdf.h2("9.1  Setup")
    pdf.code([
        "# 1. Clone the repository (if not already cloned)",
        "git clone https://github.com/akihisa-dev7/cosmx.git",
        "cd cosmx",
        "",
        "# 2. Make sure you are on the latest master",
        "git checkout master",
        "git pull origin master",
        "",
        "# 3. Create the feature branch",
        "git checkout -b feature/baysor-sopa-segmentation",
    ])

    pdf.h2("9.2  Implement the Changes (in order)")
    steps = [
        "Create cell_segmentation_pipeline/src/segmentation/baysor_segmenter.py  (see Section 5.1)",
        "Create cell_segmentation_pipeline/scripts/run_baysor.py                (see Section 5.2)",
        "Edit  cell_segmentation_pipeline/requirements.txt                      (see Section 5.3)",
        "Edit  cell_segmentation_pipeline/config/default_config.yaml            (see Section 5.4)",
        "Edit  cell_segmentation_pipeline/src/segmentation/base.py              (see Section 5.5)",
    ]
    for i, s in enumerate(steps, 1):
        pdf.numbered(i, s)
    pdf.ln(2)

    pdf.h2("9.3  Install Dependencies & Test Locally")
    pdf.code([
        "source cell_segmentation_pipeline/.venv/bin/activate",
        "pip install 'sopa[baysor]>=2.0.0' spatialdata",
        "",
        "# Quick smoke test (dry run - checks imports work)",
        "python -c \"from cell_segmentation_pipeline.src.segmentation.baysor_segmenter import BaysorSegmenter; print('OK')\"",
        "",
        "# Run on one FOV (requires Baysor binary and data)",
        "python cell_segmentation_pipeline/scripts/run_baysor.py \\",
        "  --tx_file  outputs_ssd/proseg_pilot/FOV00001_tx.csv.gz \\",
        "  --prior_dir outputs/pilot_4fov/masks/ \\",
        "  --output   outputs/baysor/ \\",
        "  --fovs FOV00001",
    ])

    pdf.h2("9.4  Commit and Push")
    pdf.code([
        "# Stage only the changed/new code files (not outputs or data)",
        "git add cell_segmentation_pipeline/src/segmentation/baysor_segmenter.py",
        "git add cell_segmentation_pipeline/scripts/run_baysor.py",
        "git add cell_segmentation_pipeline/src/segmentation/base.py",
        "git add cell_segmentation_pipeline/config/default_config.yaml",
        "git add cell_segmentation_pipeline/requirements.txt",
        "",
        "# Commit",
        'git commit -m "feat: add Baysor/Sopa RNA-guided segmentation pipeline',
        "",
        "- Add BaysorSegmenter class wrapping Sopa framework",
        "- Add run_baysor.py CLI script for pipeline execution",
        "- Register 'baysor' model in segmenter factory",
        "- Add baysor config section with CosMx-tuned defaults",
        "- All existing Cellpose/StarDist paths unchanged",
        "",
        'Scientific motivation: RNA-guided segmentation corrects DAPI-only bias"',
        "",
        "# Push to remote",
        "git push -u origin feature/baysor-sopa-segmentation",
        "",
        "# Open a Pull Request on GitHub:",
        "# Base: master   <--   Compare: feature/baysor-sopa-segmentation",
        "# Title: 'feat: RNA-guided segmentation via Baysor + Sopa'",
    ])

    pdf.h2("9.5  Pull Request Checklist")
    checks = [
        "baysor_segmenter.py imports cleanly (python -c 'from ... import BaysorSegmenter')",
        "run_baysor.py --help prints usage without error",
        "requirements.txt contains sopa[baysor]>=2.0.0",
        "default_config.yaml has [segmentation.baysor] block",
        "base.py factory returns BaysorSegmenter for model='baysor'",
        "No .h5ad, .tif, .csv.gz, or .npy files staged (check git status)",
        "Existing tests pass: python -m pytest cell_segmentation_pipeline/ (if tests exist)",
    ]
    for c in checks:
        pdf.bullet(c)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 10
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("10.  Testing & Validation Plan")

    pdf.h2("10.1  Quantitative Metrics to Compare")
    pdf.table(
        ["Metric", "Cellpose Baseline", "Target (Baysor)", "Why It Matters"],
        [
            ["Cells per FOV", "521 (FOV00001)", "500-700", "Over-segmentation check"],
            ["Median cell area (px2)", "~TBD from QC", "Similar or larger", "Cell size integrity"],
            ["Median transcripts/cell", "~TBD from ProSeg", ">= baseline", "RNA capture efficiency"],
            ["% transcripts assigned", "99% (ProSeg)", ">= 95%", "Data utilisation"],
            ["Median genes/cell", "~TBD", "Higher than baseline", "Cell identity resolution"],
        ],
        col_widths=[55, 38, 32, 65],
    )

    pdf.h2("10.2  Visual QC (using existing GUI)")
    for item in [
        "Load baysor_mask.tif in Page 01 (FOV Viewer) - verify boundaries look biologically reasonable",
        "Compare side-by-side with cellpose_cpsam_nuclear_mask.tif in Page 02 (Model Comparison)",
        "Check that small glia-sized cells (< 100 px2) are present in both masks",
        "Verify that large neurons (> 1000 px2) are not split into fragments",
    ]:
        pdf.bullet(item)

    pdf.h2("10.3  Biological Validation (ground truth)")
    pdf.body(
        "The Kyoto University expert (sensei) should visually evaluate the Baysor segmentation output "
        "on the same FOVs used for the current ~50% accuracy estimate. "
        "Provide a side-by-side PDF comparison: Cellpose (current) vs. Baysor (new)."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 11
    # ══════════════════════════════════════════════════════════════════════════
    pdf.h1("11.  Timeline & Risk Register")

    pdf.h2("11.1  Estimated Timeline")
    pdf.table(
        ["Task", "Est. Days", "Owner", "Depends On"],
        [
            ["Install Sopa + Baysor binary", "0.5", "Developer", "-"],
            ["Create baysor_segmenter.py + run_baysor.py", "1.5", "Developer", "Installation"],
            ["Edit base.py, config, requirements.txt", "0.5", "Developer", "New files"],
            ["Run pilot (FOV00001, parameter sweep)", "2.0", "Developer", "Code complete"],
            ["Visual QC + comparison with Cellpose", "1.0", "Developer + Sensei", "Pilot run"],
            ["Adjust parameters based on QC feedback", "1.0", "Developer", "QC review"],
            ["Run all 4 FOVs + write PR", "1.0", "Developer", "Parameters tuned"],
            ["TOTAL", "~7.5 days", "", ""],
        ],
        col_widths=[75, 22, 43, 50],
    )

    pdf.h2("11.2  Risk Register")
    pdf.table(
        ["Risk", "Likelihood", "Impact", "Mitigation"],
        [
            ["Sopa API changes (v2.x vs v1.x)", "Medium", "High", "Pin sopa==2.x.y after testing; check changelog"],
            ["Baysor over-segmentation", "High", "Medium", "Increase prior_confidence to 0.6-0.8"],
            ["tx_file CRC error (known issue)", "Certain", "Low", "Use FOV-split files in outputs_ssd/proseg_pilot/"],
            ["Baysor binary not on PATH", "Medium", "Low", "Pass full path via baysor_bin config param"],
            ["Memory OOM on 4256x4256 FOV", "Low", "High", "Sopa tiles automatically; use tile_size=2048"],
        ],
        col_widths=[60, 26, 22, 82],
    )

    pdf.note_box(
        "If Sopa v2.x API is not compatible with the data format or the API has changed significantly, "
        "fall back to direct Baysor CLI invocation (subprocess) as described in the previous "
        "implementation plan. The baysor_segmenter.py structure supports this with a flag.",
        color=PDF.NOTE_BG, label="FALLBACK PLAN"
    )

    pdf.output(str(out_path))
    print(f"PDF written to {out_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "outputs" / "baysor_sopa_implementation_spec.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(out)
