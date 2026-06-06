# CosMx Molecular Analysis Pipeline

## 目的

CosMx SMI の RNA / Protein データを細胞ごとに解析し、各細胞が「何者か」「何をしているか」を RNA / Protein 情報から推定するアプリケーション。

既存の `cell_segmentation_pipeline/` で作成した **custom segmentation mask** に RNA transcript を割り当て、QC・正規化・クラスタリング・細胞種アノテーション・空間解析・RNA-Protein 統合まで一括実行する。

---

## 入力データ一覧

| データ | パス | 内容 |
|--------|------|------|
| Segmentation mask | `outputs/pilot_4fov/expanded_masks/FOV*_expanded_mask.tif` | custom expanded mask (uint32) |
| Cell table | `outputs/pilot_4fov/cell_tables/FOV*_cells.csv` | 細胞形態 (面積・セントロイドなど) |
| RNA tx_file | `raw_data/pilot_4fov/slide1_RNA/flatfiles/*_tx_file.csv.gz` | RNA 1分子ごとの x/y/target |
| RNA exprMat | `raw_data/pilot_4fov/slide1_RNA/flatfiles/*_exprMat_file.csv.gz` | 細胞×遺伝子カウント行列 (native segmentation) |
| RNA metadata | `raw_data/pilot_4fov/slide1_RNA/flatfiles/*_metadata_file.csv.gz` | native 細胞メタデータ |
| Protein exprMat | `raw_data/pilot_4fov/slide1_protein/flatfiles/*_exprMat_file.csv.gz` | 細胞×タンパク質強度行列 |

---

## 出力データ一覧

| ファイル | 内容 |
|---------|------|
| `transcript_assignment/assigned_transcripts.csv.gz` | RNA 割り当て結果 (custom cell_id 付き) |
| `transcript_assignment/rna_counts_matrix.csv.gz` | cell × gene カウント行列 |
| `transcript_assignment/assignment_summary.csv` | FOV 別割り当て統計 |
| `anndata/rna_custom_segmentation.h5ad` | 生カウント AnnData |
| `anndata/rna_qc_filtered.h5ad` | QC 後 AnnData |
| `anndata/rna_clustered.h5ad` | 正規化・クラスタリング済み AnnData |
| `anndata/rna_annotated.h5ad` | 細胞種アノテーション済み AnnData |
| `cell_type_table.csv` | 細胞ごとの細胞種テーブル |
| `spatial/spatial_cell_type_map_FOV*.png` | 空間細胞種マップ |
| `spatial/cell_type_composition_by_fov.csv` | FOV 別細胞種割合 |
| `spatial/cell_density_by_fov.csv` | FOV 別細胞密度 |
| `integration/joint_fov_summary.csv` | FOV 単位 RNA-Protein 統合サマリ |
| `integration/rna_protein_correlation.csv` | RNA-Protein 相関 |

---

## セットアップ方法

```bash
# 1. プロジェクトルートに移動
cd /path/to/CosMx_2026

# 2. 仮想環境を作成・有効化
python3 -m venv cosmx_molecular_analysis/.venv
source cosmx_molecular_analysis/.venv/bin/activate

# 3. 依存パッケージをインストール
pip install scanpy anndata leidenalg umap-learn tifffile \
            pyyaml scipy scikit-learn matplotlib seaborn pandas numpy
```

---

## 4 FOV Pilot の実行手順

すべて `CosMx_2026/` ディレクトリから実行してください。

```bash
source cosmx_molecular_analysis/.venv/bin/activate
```

### Step 1: RNA Transcript Assignment

```bash
python cosmx_molecular_analysis/scripts/01_assign_transcripts.py \
  --tx-file "raw_data/pilot_4fov/slide1_RNA/flatfiles/2511276KSlide116_tx_file.csv.gz" \
  --exprmat "raw_data/pilot_4fov/slide1_RNA/flatfiles/2511276KSlide116_exprMat_file.csv.gz" \
  --metadata "raw_data/pilot_4fov/slide1_RNA/flatfiles/2511276KSlide116_metadata_file.csv.gz" \
  --mask-dir "outputs/pilot_4fov/expanded_masks" \
  --cell-table-dir "outputs/pilot_4fov/cell_tables" \
  --output "outputs/cosmx_molecular_analysis/transcript_assignment" \
  --fovs FOV00001 FOV00007 FOV00037 FOV00043
```

### Step 2: Build AnnData

```bash
python cosmx_molecular_analysis/scripts/02_build_anndata.py \
  --counts "outputs/cosmx_molecular_analysis/transcript_assignment/rna_counts_matrix.csv.gz" \
  --cell-table-dir "outputs/pilot_4fov/cell_tables" \
  --output "outputs/cosmx_molecular_analysis/anndata/rna_custom_segmentation.h5ad" \
  --fovs FOV00001 FOV00007 FOV00037 FOV00043
```

### Step 3: QC Filtering

```bash
python cosmx_molecular_analysis/scripts/03_qc_filter.py \
  --input "outputs/cosmx_molecular_analysis/anndata/rna_custom_segmentation.h5ad" \
  --output "outputs/cosmx_molecular_analysis/anndata/rna_qc_filtered.h5ad"
```

### Step 4: Normalize & Cluster

```bash
python cosmx_molecular_analysis/scripts/04_normalize_cluster.py \
  --input "outputs/cosmx_molecular_analysis/anndata/rna_qc_filtered.h5ad" \
  --output "outputs/cosmx_molecular_analysis/anndata/rna_clustered.h5ad"
```

### Step 5: Cell Type Annotation

```bash
python cosmx_molecular_analysis/scripts/05_cell_typing.py \
  --input "outputs/cosmx_molecular_analysis/anndata/rna_clustered.h5ad" \
  --markers "cosmx_molecular_analysis/config/marker_genes.yaml" \
  --output "outputs/cosmx_molecular_analysis/anndata/rna_annotated.h5ad"
```

### Step 6: Spatial Analysis

```bash
python cosmx_molecular_analysis/scripts/06_spatial_analysis.py \
  --input "outputs/cosmx_molecular_analysis/anndata/rna_annotated.h5ad" \
  --output "outputs/cosmx_molecular_analysis/spatial"
```

### Step 7: RNA–Protein Integration

```bash
python cosmx_molecular_analysis/scripts/07_integrate_rna_protein.py \
  --rna "outputs/cosmx_molecular_analysis/anndata/rna_annotated.h5ad" \
  --protein "raw_data/pilot_4fov/slide1_protein/flatfiles/251121Slide116_exprMat_file.csv.gz" \
  --output "outputs/cosmx_molecular_analysis/integration" \
  --fovs FOV00001 FOV00007 FOV00037 FOV00043
```

### Full Pipeline (一括実行)

```bash
python cosmx_molecular_analysis/scripts/run_full_pipeline.py \
  --config cosmx_molecular_analysis/config/default_config.yaml
```

特定のステップをスキップする場合:
```bash
python cosmx_molecular_analysis/scripts/run_full_pipeline.py \
  --config cosmx_molecular_analysis/config/default_config.yaml \
  --skip-steps 7
```

---

## 各ステップの説明

### Step 1: Transcript Assignment (ハイブリッド方式)

**tx_file が存在する FOV** (例: FOV00001, FOV00007):
- RNA 1分子の `x_local_px`, `y_local_px` を custom mask にルックアップ
- `mask[y, x] > 0` なら対応する custom cell_id に割り当て
- NegPrb*, FalseCode* を除外

**tx_file が存在しない FOV** (例: FOV00037, FOV00043):
- native metadata の `CenterX_local_px`, `CenterY_local_px` で custom mask をルックアップ
- native cell_ID → custom cell_id のマッピングを作成
- native exprMat のカウントを custom cell 単位で集計

> **注意**: 本データセットでは tx_file はスライドの一部 (FOV 1–13) しかカバーしない。
> FOV37/43 は native exprMat からのフォールバックを使用。

### Step 2: AnnData Construction

- counts matrix + cell morphology table → AnnData
- `obsm["spatial"]` に centroid 座標 (µm) を格納

### Step 3: QC Filtering

| パラメータ | デフォルト値 |
|-----------|-------------|
| min_transcripts_per_cell | 5 |
| max_transcripts_per_cell | 5000 |
| min_genes_per_cell | 3 |
| max_negative_probe_fraction | 0.10 |
| min_cell_area_um² | 10 |
| max_cell_area_um² | 8000 |

### Step 4: Normalization & Clustering

1. Library size normalization (target_sum = 10,000)
2. log1p
3. Highly variable genes selection (top 2000)
4. PCA (30 components)
5. k-NN graph (k=15)
6. UMAP
7. Leiden clustering (resolution=0.5)

### Step 5: Cell Type Annotation

マーカー遺伝子スコア (各細胞種の marker genes の平均正規化発現量) を計算し、最高スコアの細胞種を予測細胞種とする。  
スコア差が `min_score_margin` (デフォルト 0.15) 未満の場合は `LowConfidence` ラベルを付与。

### Step 6: Spatial Analysis

- FOV ごとの細胞種空間マップ (scatter plot)
- FOV 別細胞種割合
- FOV 別細胞密度 (cells/mm²)
- 近傍細胞種解析 (KDTree、半径 100px)
- AD vs Control 細胞種割合比較

### Step 7: RNA–Protein FOV-level Integration

> RNA スライド ≠ Protein スライドのため、**細胞単位での直接統合は行わない**。  
> FOV 単位の平均スコア/強度で統合する。

RNA: 細胞種スコア平均 + 細胞種割合 (per FOV)  
Protein: タンパク質強度平均 (per FOV)  
↓  
FOV 単位で merge → Pearson 相関計算

---

## よくあるエラー

### `BadGzipFile: CRC check failed`
tx_file の末尾が破損している場合。パイプライン内で `zcat` 経由読み込みに自動フォールバックするため通常は問題なし。

### `ValueError: invalid literal for int() with base 10`
tx_file に非数値座標が含まれる場合。`pd.to_numeric(..., errors='coerce')` で自動的にスキップ。

### `ModuleNotFoundError: No module named 'scanpy'`
仮想環境が有効化されていない:
```bash
source cosmx_molecular_analysis/.venv/bin/activate
```

### Segmentation fault (メモリ不足)
exprMat (全 72 FOV × 6521 genes) を一括ロードするとクラッシュする場合がある。  
スクリプトはチャンク読み込みで対処済み。それでも失敗する場合は RAM 使用量を確認。

### `No mask found for FOVXXXXX`
`--mask-dir` に対象 FOV のマスクファイルが存在しない。`outputs/pilot_4fov/expanded_masks/` に `FOV*_expanded_mask.tif` があるか確認。

---

## RNA/Protein 統合時の注意

1. **RNA スライドと Protein スライドは別物**: 同一 `cell_ID` でも同一細胞ではない。細胞単位では直接結合しない。

2. **FOV 単位統合が基本**: FOV ごとの平均値レベルで RNA マーカースコアと Protein 強度を比較する。

3. **Protein パネルは免疫マーカー中心**: 本データの Protein パネル (CD68, CD31, SMA など) は免疫細胞マーカーが主体。脳固有マーカー (GFAP, MBP など) は含まれない。

4. **custom cell_id の命名**: `<FOV名>_<mask_label>` 形式 (例: `FOV00001_42`)。CosMx native の `cell_ID` とは別物。

---

## ディレクトリ構成

```
cosmx_molecular_analysis/
├── config/
│   ├── default_config.yaml       # デフォルト設定
│   └── marker_genes.yaml         # 細胞種マーカー遺伝子
├── src/
│   ├── io.py                     # データ読み込みユーティリティ
│   ├── transcript_assignment.py  # RNA transcript 割り当て
│   ├── protein_processing.py     # Protein データ前処理
│   ├── anndata_builder.py        # AnnData 構築
│   ├── qc.py                     # QC フィルタリング
│   ├── normalization.py          # 正規化
│   ├── clustering.py             # クラスタリング
│   ├── cell_typing.py            # 細胞種アノテーション
│   ├── spatial_analysis.py       # 空間解析
│   ├── integration.py            # RNA-Protein 統合
│   └── visualization.py          # 可視化ユーティリティ
├── scripts/
│   ├── 01_assign_transcripts.py
│   ├── 02_build_anndata.py
│   ├── 03_qc_filter.py
│   ├── 04_normalize_cluster.py
│   ├── 05_cell_typing.py
│   ├── 06_spatial_analysis.py
│   ├── 07_integrate_rna_protein.py
│   └── run_full_pipeline.py
├── .venv/                        # 仮想環境
└── README.md
```
