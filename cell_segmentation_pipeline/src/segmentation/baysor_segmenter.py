"""BaysorSegmenter: RNA-guided segmentation via direct Baysor CLI.

Baysor jointly optimises transcript spatial density and nuclear DAPI prior to
produce more accurate cell boundaries than DAPI-only methods, especially in AD
tissue where cell morphology is altered by disease pathology.

Requires:
    - baysor binary on PATH (download from github.com/kharchenkolab/Baysor/releases)
    - anndata, scipy, tifffile, pandas, numpy (all in requirements.txt)

Usage:
    seg = BaysorSegmenter()
    mask_path = seg.run_fov(
        tx_csv      = Path("outputs_ssd/proseg_pilot/FOV00001_tx.csv.gz"),
        prior_mask  = Path("outputs/pilot_4fov/masks/FOV00001_AD_F_cpsam_nuclear_mask.tif"),
        out_dir     = Path("outputs/baysor/FOV00001"),
        fov_col     = "fov",
        fov_id      = 1,
    )
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

import anndata
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from .base import BaseSegmenter


# ── TOML template (no external toml library needed) ───────────────────────────

_TOML_TEMPLATE = """\
[data]
x = "x"
y = "y"
{z_line}
gene = "gene"
min_molecules_per_gene = 0

[segmentation]
scale = {scale:.1f}
scale_std = "50%"
min_molecules_per_cell = {min_molecules}
n_clusters = {n_clusters}
force_2d = {force_2d}
prior_segmentation_confidence = {prior_confidence:.2f}
"""


class BaysorSegmenter(BaseSegmenter):
    """RNA-guided segmenter wrapping the Baysor CLI.

    Args:
        scale_um:          Expected cell radius in micrometres (default 10).
        min_molecules:     Min transcript count to keep a cell (default 15).
        prior_confidence:  Weight of DAPI prior vs RNA signal, 0–1 (default 0.5).
        n_clusters:        Number of expression-type clusters (default 4).
        pixel_size_um:     CosMx calibration in um/px (default 0.12028).
        baysor_bin:        Baysor executable name or full path (default "baysor").
        prior_dilation_px: How many px to dilate prior mask for Voronoi boundary (default 15).
        use_z:             Whether to pass z-coordinates to Baysor (default False).
    """

    # Docker image to use when baysor binary is not available locally
    DOCKER_IMAGE = "vpetukhov/baysor:latest"

    def __init__(
        self,
        scale_um: float = 10.0,
        min_molecules: int = 15,
        prior_confidence: float = 0.5,
        n_clusters: int = 4,
        pixel_size_um: float = 0.12028,
        baysor_bin: str = "baysor",
        prior_dilation_px: int = 15,
        use_z: bool = False,
        use_docker: bool = False,
    ) -> None:
        self.scale_um = scale_um
        self.min_molecules = min_molecules
        self.prior_confidence = prior_confidence
        self.n_clusters = n_clusters
        self.pixel_size_um = pixel_size_um
        self.baysor_bin = baysor_bin
        self.prior_dilation_px = prior_dilation_px
        self.use_z = use_z
        self.use_docker = use_docker

    @property
    def name(self) -> str:
        return "baysor"

    def segment(self, image: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "BaysorSegmenter.segment() is not supported — Baysor requires transcript "
            "coordinates, not a pixel image. Use run_fov() instead."
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def run_fov(
        self,
        tx_csv: Path,
        prior_mask: Path,
        out_dir: Path,
        fov_col: str = "fov",
        fov_id: Optional[int] = None,
        image_shape: tuple[int, int] = (4256, 4256),
    ) -> Path:
        """Run Baysor on a single FOV and return the path to baysor_mask.tif.

        Args:
            tx_csv:       Transcript CSV (or .csv.gz). Must contain x_local_px,
                          y_local_px, target columns (standard CosMx tx_file format).
                          If fov_id is given, only rows matching that FOV are used.
            prior_mask:   Integer-label TIFF from Cellpose (0 = background).
            out_dir:      Directory to write all Baysor outputs.
            fov_col:      Column name for FOV identifier in tx_csv.
            fov_id:       Numeric FOV ID to filter (None = use entire file).
            image_shape:  (H, W) of the DAPI image, default CosMx (4256, 4256).

        Returns:
            Path to baysor_mask.tif (integer-label TIFF, uint16, same size as image).
        """
        self._check_baysor_binary()

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Prepare transcript CSV
        print(f"  [baysor] Loading transcripts from {tx_csv.name} ...")
        tx_df = self._load_and_prep_transcripts(tx_csv, fov_col, fov_id)
        if len(tx_df) == 0:
            raise ValueError(f"No valid transcripts found for fov_id={fov_id}")
        print(f"  [baysor] {len(tx_df):,} transcripts, {tx_df['gene'].nunique()} genes")

        baysor_csv = out_dir / "transcripts_for_baysor.csv"
        tx_df.to_csv(baysor_csv, index=False)

        # 2. Write TOML config
        toml_path = self._write_config(out_dir)

        # 3. Run Baysor
        self._run_baysor(baysor_csv, prior_mask, out_dir, toml_path)

        # 4. Parse outputs
        seg_df, stats_df = self._read_baysor_output(out_dir)
        n_cells = len(stats_df) if stats_df is not None else 0
        print(f"  [baysor] {n_cells} cells segmented")

        # 5. Build label mask
        mask = self._build_label_mask(stats_df, prior_mask, image_shape)
        mask_path = out_dir / "baysor_mask.tif"
        tifffile.imwrite(str(mask_path), mask)
        print(f"  [baysor] Mask saved -> {mask_path}")

        # 6. Build AnnData
        if seg_df is not None and stats_df is not None:
            adata = self._build_anndata(seg_df, stats_df, fov_id)
            adata.write_h5ad(str(out_dir / "anndata.h5ad"))
            print(f"  [baysor] AnnData saved ({adata.n_obs} cells, {adata.n_vars} genes)")

        return mask_path

    # ── Docker Compose 3-stage helpers ────────────────────────────────────────

    def prepare_fov(
        self,
        tx_csv: Path,
        prior_mask: Optional[Path],
        out_dir: Path,
        fov_col: str = "fov",
        fov_id: Optional[int] = None,
    ) -> None:
        """Stage 1: write transcripts CSV + TOML config; print baysor docker command.

        Run this inside the *segmentation* container.  After it completes, run
        the printed command inside the *baysor* container, then call postprocess_fov().
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [prepare] Loading transcripts from {tx_csv.name} ...")
        tx_df = self._load_and_prep_transcripts(tx_csv, fov_col, fov_id)
        if len(tx_df) == 0:
            raise ValueError(f"No valid transcripts for fov_id={fov_id}")
        print(f"  [prepare] {len(tx_df):,} transcripts, {tx_df['gene'].nunique()} genes")

        baysor_csv = out_dir / "transcripts_for_baysor.csv"
        tx_df.to_csv(baysor_csv, index=False)

        toml_path = self._write_config(out_dir)

        if prior_mask and prior_mask.exists():
            import shutil
            prior_copy = out_dir / "prior_mask.tif"
            if not prior_copy.exists():
                shutil.copy2(prior_mask, prior_copy)
            prior_arg = "/workspace/" + str(prior_copy.relative_to(Path("/workspace")))
        else:
            prior_arg = None

        ws_out    = "/workspace/" + str(out_dir.resolve().relative_to(Path("/workspace")))
        ws_tx     = ws_out + "/transcripts_for_baysor.csv"
        ws_toml   = ws_out + "/baysor_config.toml"
        ws_prior  = ws_out + "/prior_mask.tif" if prior_arg else ""

        cmd_parts = [
            "docker compose run --rm baysor",
            f"/usr/local/bin/baysor run",
            f"-c {ws_toml}",
            f"-o {ws_out}/",
            ws_tx,
        ]
        if ws_prior:
            cmd_parts.append(ws_prior)

        print(f"\n  [prepare] Files written to: {out_dir}")
        print(f"  [prepare] Next — run in baysor container:")
        print(f"\n    {' '.join(cmd_parts)}\n")

    def postprocess_fov(
        self,
        out_dir: Path,
        prior_mask: Optional[Path] = None,
        image_shape: tuple[int, int] = (4256, 4256),
        fov_id: Optional[int] = None,
    ) -> Path:
        """Stage 3: read Baysor output, build label mask + AnnData.

        Run this inside the *segmentation* container after Baysor has finished.
        Returns path to baysor_mask.tif.
        """
        out_dir = Path(out_dir)

        seg_df, stats_df = self._read_baysor_output(out_dir)
        n_cells = len(stats_df) if stats_df is not None else 0
        print(f"  [postprocess] {n_cells} cells from Baysor output")

        prior_path = prior_mask or (out_dir / "prior_mask.tif")
        mask = self._build_label_mask(stats_df, prior_path, image_shape)
        mask_path = out_dir / "baysor_mask.tif"
        tifffile.imwrite(str(mask_path), mask)
        print(f"  [postprocess] Mask saved -> {mask_path}")

        if seg_df is not None and stats_df is not None:
            adata = self._build_anndata(seg_df, stats_df, fov_id)
            adata.write_h5ad(str(out_dir / "anndata.h5ad"))
            print(f"  [postprocess] AnnData saved ({adata.n_obs} cells, {adata.n_vars} genes)")

        return mask_path

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_baysor_binary(self) -> None:
        if self.use_docker:
            result = subprocess.run(
                ["docker", "run", "--rm", self.DOCKER_IMAGE, "/bin/baysor", "--version"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise FileNotFoundError(
                    f"Docker image '{self.DOCKER_IMAGE}' not available or Docker daemon not running.\n"
                    "Start Docker and pull the image:\n"
                    "  sudo systemctl start docker\n"
                    f"  docker pull {self.DOCKER_IMAGE}"
                )
            print(f"  [baysor] Docker: {result.stdout.strip()}")
            return

        result = subprocess.run(
            [self.baysor_bin, "--version"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise FileNotFoundError(
                f"Baysor binary '{self.baysor_bin}' not found.\n"
                "Option A — Download binary (v0.7.1, Linux x86_64):\n"
                "  wget https://github.com/kharchenkolab/Baysor/releases/download/"
                "v0.7.1/baysor-x86_x64-linux-v0.7.1_build.zip\n"
                "  unzip baysor-x86_x64-linux-v0.7.1_build.zip\n"
                "  chmod +x bin/baysor && sudo mv bin/baysor /usr/local/bin/\n\n"
                "Option B — Docker (recommended if Docker daemon is running):\n"
                f"  docker pull {self.DOCKER_IMAGE}\n"
                "  Then pass use_docker=True to BaysorSegmenter or --use_docker to run_baysor.py"
            )
        print(f"  [baysor] Binary: {result.stdout.strip()}")

    def _load_and_prep_transcripts(
        self, tx_csv: Path, fov_col: str, fov_id: Optional[int]
    ) -> pd.DataFrame:
        df = pd.read_csv(tx_csv, low_memory=False)

        # Filter by FOV
        if fov_id is not None and fov_col in df.columns:
            df = df[df[fov_col] == fov_id].copy()

        # Rename columns to Baysor standard
        col_map: dict[str, str] = {}
        for src, dst in [
            ("x_local_px", "x"), ("x_global_px", "x"),
            ("y_local_px", "y"), ("y_global_px", "y"),
            ("target", "gene"),
        ]:
            if src in df.columns and dst not in col_map.values():
                col_map[src] = dst
        df = df.rename(columns=col_map)

        # Keep z if available
        keep = ["x", "y", "gene"]
        if self.use_z and "z" in df.columns:
            keep.append("z")

        df = df[[c for c in keep if c in df.columns]].copy()

        # Filter noise transcripts (z = -1)
        if "z" in df.columns:
            df = df[df["z"] >= 0]

        return df.dropna(subset=["x", "y", "gene"])

    def _write_config(self, out_dir: Path) -> Path:
        scale_px = self.scale_um / self.pixel_size_um
        z_line = 'z = "z"' if self.use_z else "# z column disabled (force_2d = true)"
        force_2d = "false" if self.use_z else "true"

        toml = _TOML_TEMPLATE.format(
            z_line=z_line,
            scale=scale_px,
            min_molecules=self.min_molecules,
            n_clusters=self.n_clusters,
            force_2d=force_2d,
            prior_confidence=self.prior_confidence,
        )
        toml_path = out_dir / "baysor_config.toml"
        toml_path.write_text(toml)
        return toml_path

    def _run_baysor(
        self,
        tx_csv: Path,
        prior_mask: Path,
        out_dir: Path,
        toml_path: Path,
    ) -> None:
        has_prior = prior_mask is not None and Path(prior_mask).exists()
        if not has_prior:
            warnings.warn("Prior mask not found — running Baysor without prior (may over-segment)")

        if self.use_docker:
            self._run_baysor_docker(tx_csv, prior_mask if has_prior else None, out_dir, toml_path)
        else:
            self._run_baysor_native(tx_csv, prior_mask if has_prior else None, out_dir, toml_path)

    def _run_baysor_native(self, tx_csv, prior_mask, out_dir, toml_path) -> None:
        cmd = [
            self.baysor_bin, "run",
            "--config", str(toml_path),
            "-o", str(out_dir) + "/",
            str(tx_csv),
        ]
        if prior_mask:
            cmd.append(str(prior_mask))
        print(f"  [baysor] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Baysor exited with code {result.returncode}.")

    def _run_baysor_docker(self, tx_csv, prior_mask, out_dir, toml_path) -> None:
        # Mount out_dir (which contains tx_csv, toml, and will receive output)
        mount_dir = out_dir.resolve()
        tx_rel    = Path("/data") / tx_csv.name
        toml_rel  = Path("/data") / toml_path.name

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{mount_dir}:/data",
            self.DOCKER_IMAGE,
            "/bin/baysor", "run",
            "--config", str(toml_rel),
            "-o", "/data/",
            str(tx_rel),
        ]
        if prior_mask:
            import shutil
            prior_copy = mount_dir / "prior_mask.tif"
            shutil.copy2(prior_mask, prior_copy)
            cmd.append("/data/prior_mask.tif")

        print(f"  [baysor] Docker cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Baysor Docker exited with code {result.returncode}.")

    def _read_baysor_output(
        self, out_dir: Path
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Read segmentation.csv and segmentation_cell_stats.csv from out_dir."""
        # Baysor may write files with slightly different names depending on version
        seg_candidates = list(out_dir.glob("segmentation*.csv"))
        stats_candidates = [f for f in seg_candidates if "cell_stats" in f.name or "cells" in f.name]
        tx_candidates   = [f for f in seg_candidates if "cell_stats" not in f.name and "cells" not in f.name]

        seg_df = pd.read_csv(tx_candidates[0]) if tx_candidates else None
        stats_df = pd.read_csv(stats_candidates[0]) if stats_candidates else None

        return seg_df, stats_df

    def _build_label_mask(
        self,
        stats_df: Optional[pd.DataFrame],
        prior_path: Path,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Create an integer-label mask via Voronoi tessellation of cell centroids.

        Cells are only assigned to pixels that lie within the dilated prior mask
        (background stays 0). This constrains Baysor cells to plausible regions.
        """
        H, W = image_shape

        # Load prior mask for background constraint
        prior = tifffile.imread(str(prior_path)).astype(np.uint32)
        if prior.shape != (H, W):
            # Resize if needed (shouldn't happen with aligned data)
            from skimage.transform import resize
            prior = (resize(prior, (H, W), order=0, preserve_range=True)).astype(np.uint32)

        if stats_df is None or len(stats_df) == 0:
            warnings.warn("No cells in Baysor output — returning empty mask")
            return np.zeros((H, W), dtype=np.uint16)

        # Detect coordinate and ID columns (flexible for different Baysor versions)
        x_col = next((c for c in ["x", "x_centroid", "cx"] if c in stats_df.columns), None)
        y_col = next((c for c in ["y", "y_centroid", "cy"] if c in stats_df.columns), None)
        id_col = next((c for c in ["cell", "cell_id", "name"] if c in stats_df.columns), None)

        if x_col is None or y_col is None:
            warnings.warn(f"Cannot find x/y centroid columns in {stats_df.columns.tolist()}")
            return np.zeros((H, W), dtype=np.uint16)

        cx = stats_df[x_col].values.astype(float)
        cy = stats_df[y_col].values.astype(float)

        # Clip centroids to image bounds
        cx = np.clip(cx, 0, W - 1)
        cy = np.clip(cy, 0, H - 1)

        # Build Voronoi: assign every pixel to nearest centroid
        centroids = np.stack([cx, cy], axis=1)   # (N, 2) — x,y
        tree = cKDTree(centroids)

        yy, xx = np.mgrid[0:H, 0:W]
        pixels = np.stack([xx.ravel(), yy.ravel()], axis=1)   # (H*W, 2)

        _, indices = tree.query(pixels, k=1, workers=-1)
        pixel_labels = (indices + 1).reshape(H, W).astype(np.uint16)   # 1-indexed

        # Constrain to dilated prior mask (cells only where tissue exists)
        tissue_mask = binary_dilation(prior > 0, iterations=self.prior_dilation_px)
        pixel_labels[~tissue_mask] = 0

        return pixel_labels

    def _build_anndata(
        self,
        seg_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        fov_id: Optional[int],
    ) -> anndata.AnnData:
        """Build cell x gene AnnData from Baysor transcript assignments."""
        from scipy.sparse import csr_matrix

        cell_col = next((c for c in ["cell", "cell_id"] if c in seg_df.columns), None)
        gene_col = next((c for c in ["gene", "target"] if c in seg_df.columns), None)

        if cell_col is None or gene_col is None:
            raise ValueError(
                f"Cannot find cell/gene columns in segmentation.csv: {seg_df.columns.tolist()}"
            )

        # Drop unassigned transcripts (cell == 0 or NaN)
        assigned = seg_df[seg_df[cell_col].notna() & (seg_df[cell_col] != 0)].copy()

        # Build count matrix
        counts = (
            assigned.groupby([cell_col, gene_col])
            .size()
            .unstack(fill_value=0)
        )

        adata = anndata.AnnData(
            X=csr_matrix(counts.values),
            obs=pd.DataFrame(index=counts.index.astype(str)),
            var=pd.DataFrame(index=counts.columns),
        )
        adata.obs["n_counts"] = np.array(counts.sum(axis=1))
        adata.obs["n_genes"] = np.array((counts > 0).sum(axis=1))

        # Add spatial coordinates from stats
        x_col = next((c for c in ["x", "x_centroid"] if c in stats_df.columns), None)
        y_col = next((c for c in ["y", "y_centroid"] if c in stats_df.columns), None)
        id_col = next((c for c in ["cell", "cell_id"] if c in stats_df.columns), None)
        if x_col and y_col and id_col:
            stats_idx = stats_df.set_index(id_col.astype(str) if hasattr(id_col, 'astype') else id_col)
            stats_idx.index = stats_idx.index.astype(str)
            shared = adata.obs_names[adata.obs_names.isin(stats_idx.index)]
            coords = stats_idx.loc[shared, [x_col, y_col]].values
            adata.obsm["spatial"] = np.zeros((adata.n_obs, 2))
            idx_pos = [list(adata.obs_names).index(s) for s in shared]
            adata.obsm["spatial"][idx_pos] = coords

        adata.uns["baysor_params"] = {
            "scale_um": self.scale_um,
            "min_molecules": self.min_molecules,
            "prior_confidence": self.prior_confidence,
            "n_clusters": self.n_clusters,
            "pixel_size_um": self.pixel_size_um,
            "fov_id": fov_id,
        }

        return adata
