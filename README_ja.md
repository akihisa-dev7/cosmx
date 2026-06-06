# CosMx 2026 — 認知症空間マルチオミクス解析
**京都大学 GO研究室**
**空間RNA + タンパク質：ヒトアルツハイマー病 vs 対照脳**

---

## 目次

1. [研究概要](#1-研究概要)
2. [データの説明](#2-データの説明)
3. [ディレクトリ構成](#3-ディレクトリ構成)
4. [環境構築](#4-環境構築)
5. [クイックスタート — 4 FOVパイロット解析](#5-クイックスタート--4-fovパイロット解析)
6. [セグメンテーションパイプライン — 詳細ガイド](#6-セグメンテーションパイプライン--詳細ガイド)
7. [トランスクリプト割り当て](#7-トランスクリプト割り当て)
8. [AnnDataの構築とQC](#8-anndataの構築とqc)
9. [クラスタリングと細胞種アノテーション](#9-クラスタリングと細胞種アノテーション)
10. [空間プロテオミクス解析](#10-空間プロテオミクス解析)
11. [空間トランスクリプトミクス解析](#11-空間トランスクリプトミクス解析)
12. [RNA–タンパク質統合解析](#12-rnaタンパク質統合解析)
13. [全144 FOVへのスケールアップ](#13-全144-fovへのスケールアップ)
14. [パイロット解析の主要知見](#14-パイロット解析の主要知見)
15. [論文投稿チェックリスト](#15-論文投稿チェックリスト)

---

## 1. 研究概要

本プロジェクトでは、**NanoString CosMx空間分子イメージャー（SMI）**を用いて、アルツハイマー病（AD）患者および年齢適合健常対照者の死後ヒト脳組織を解析します。目的は、疾患特異的な細胞種変化、空間的共局在パターン、およびAD病理の分子的特徴を明らかにするため、空間的に解像された単一細胞レベルのトランスクリプトームおよびプロテオームプロファイルを生成することです。

### 生物学的問い
前頭皮質と海馬（神経変性の程度が異なる2つの領域）において、AD脳と対照脳の間で、細胞種組成・遺伝子発現・タンパク質存在量はどのように空間的に異なるか？

### 実験デザイン

| パラメータ | 詳細 |
|---|---|
| 使用技術 | CosMx空間分子イメージャー（NanoString） |
| 組織 | 死後ヒト脳（FFPE切片） |
| パネル（RNA） | Human RNA 6K Discoveryパネル（~6,000遺伝子） |
| パネル（タンパク質） | CosMx免疫蛍光タンパク質パネル |
| 画素サイズ | 0.120274 µm/px |
| FOV数 | 合計144（スライド1枚あたり72 FOV × 2枚） |
| 条件 | アルツハイマー病（AD）vs 対照 |
| 脳領域 | 前頭皮質（F）、海馬（H） |

### スライド情報

| スライドID | スタディディレクトリ | モダリティ | ランID |
|---|---|---|---|
| 2511276KSlide116 | Study20251223_GOlab_cosmxrna1 | RNA 6K | 20251127_074826_S1 |
| 2511276KSlide2712 | Study20251223_GOlab_cosmxrna2 | RNA 6K | 20251127_074826_S2 |
| 251121Slide116 | Study20251223_GOlab_cosmxpro1 | タンパク質 | 20251121_071111_S1 |
| 251121Slide116.2 | Study20251223_GOlab_cosmxpro2 | タンパク質 | — |

---

## 2. データの説明

### 2.1 パイロットデータセット — 4 FOV（ローカル、自己完結型）

パイロット解析に必要な全ファイルは `raw_data/pilot_4fov/` にローカル保存されています。**外付けドライブは不要です。**

| FOV | 条件 | 脳領域 | サンプルID | 性別 | 年齢 |
|---|---|---|---|---|---|
| FOV00001 | AD | 前頭皮質 | 1 | 男性 | 88歳 |
| FOV00007 | AD | 海馬 | 1 | 男性 | 88歳 |
| FOV00037 | 対照 | 前頭皮質 | 4 | 男性 | 68歳 |
| FOV00043 | 対照 | 海馬 | 4 | 男性 | 68歳 |

この4 FOVは、2条件（AD/対照）× 2脳領域（前頭皮質/海馬）を網羅するように選択されており、パイプライン全体のパイロット検証に使用します。

### 2.2 形態画像チャンネル（マルチチャンネルTIFF）

各FOVはマルチチャンネルTIFF形式（C × H × W、チャンネル優先）で出力されます：

| チャンネルインデックス | マーカー | 役割 |
|---|---|---|
| 0 | DAPI | 核染色 — **主要セグメンテーションチャンネル** |
| 1 | PanCK | 上皮細胞マーカー（脳では低発現） |
| 2 | G（自家蛍光） | バックグラウンドチャンネル |
| 3 | Membrane | 細胞膜マーカー |
| 4 | CD45 | 免疫細胞マーカー |

> **脳組織における重要事項：** チャンネル0（DAPI）を核セグメンテーションに使用します。チャンネル2は初期パイロットで細胞質シグナルの探索に使用されましたが、cpsamモデルではDAPIの方が優れた結果を示しました。

### 2.3 FOVごとのデコードファイル

各FOVのCellStatsDirには以下のファイルが含まれます：

| ファイル | 説明 |
|---|---|
| `CellLabels_F*.tif` | CosMx標準細胞ラベルマスク（整数値、各細胞に固有ID） |
| `CompartmentLabels_F*.tif` | 核/細胞質/背景のコンパートメントマップ |
| `CellBoundaries_F*.csv` | CosMx標準細胞のポリゴン境界 |
| `*_Cell_Stats_F*.csv` | 細胞ごとの面積、重心、形態学的指標 |
| `*_target_call_coord.csv.gz` | 品質スコア付きトランスクリプト座標 |
| `*_complete_code_cell_target_call_coord.csv` | FOVごとの全デコードトランスクリプトテーブル |

### 2.4 フラットファイル（スライドレベル）

| ファイル | 説明 |
|---|---|
| `*_exprMat_file.csv.gz` | 細胞×遺伝子発現行列（CosMx標準細胞） |
| `*_metadata_file.csv.gz` | 細胞ごとのメタデータ（FOV、重心、面積、InSituTypeラベル） |
| `*_fov_positions_file.csv.gz` | スライド上のFOV空間座標 |
| `*_polygons.csv.gz` | 細胞ポリゴンデータ |
| `*_tx_file.csv.gz` | **全トランスクリプト（x, y, z, 遺伝子, qv）** — 2.3 GB、カスタム細胞割り当てに必須 |

### 2.5 全データセットの場所（外付けSSD）

全144 FOVのデータセットは外付けSSD（`/mnt/external/` にマウント）に保存されています：

```
/mnt/external/data/
├── Study20251223_GOlab_cosmxrna1/   # スライド1 RNA（72 FOV）
├── Study20251223_GOlab_cosmxrna2/   # スライド2 RNA（72 FOV）
├── Study20251223_GOlab_cosmxpro1/   # スライド1 タンパク質
└── Study20251223_GOlab_cosmxpro2/   # スライド2 タンパク質
```

---

## 3. ディレクトリ構成

```
CosMx_2026/
├── README.md                          # 英語版README
├── README_ja.md                       # 日本語版README（本ファイル）
├── config/
│   ├── study_config.yaml              # 中央設定ファイル（パス、QC、パラメータ）
│   ├── sample_manifest.csv            # FOV → 生物学的メタデータのマッピング
│   └── seg_best_params.json           # パイロットチューニングによる最適セグメンテーションパラメータ
│
├── raw_data/                          # パイロットデータ（SSD不要、自己完結型）
│   ├── pilot_4fov/
│   │   ├── slide1_RNA/
│   │   │   ├── morphology_images/     # 4枚のマルチチャンネルDAAPI TIFF
│   │   │   ├── per_fov_decoded/       # FOV00001, 00007, 00037, 00043のデコードファイル
│   │   │   └── flatfiles/             # exprMat, metadata, tx_file（2.3 GB）, polygons
│   │   └── slide1_protein/
│   │       ├── morphology_images/     # 4枚のタンパク質形態TIFF
│   │       ├── per_fov_decoded/       # FOVごとのタンパク質デコードファイル
│   │       └── flatfiles/             # exprMat, metadata, fov_positions, polygons
│   └── additional_data/
│       └── Dementia_CosMx_Analysis - FOV_metadata.xlsx
│
├── scripts/                           # メイン解析パイプライン（ステップ00〜12）
│   ├── 00_check_env.py                # 環境確認
│   ├── 01_load_flatfiles.py           # フラットファイル読み込み
│   ├── 02_image_enhance.py            # 画像強調処理
│   ├── 03_segment_cellpose.py         # Cellposeセグメンテーション
│   ├── 04_segment_stardist.py         # StarDistセグメンテーション
│   ├── 05_segment_instanseg.py        # InstanSegセグメンテーション
│   ├── 06_assign_transcripts.py       # トランスクリプト割り当て
│   ├── 07_build_anndata.py            # AnnData構築
│   ├── 08_qc_filter.py                # QCフィルタリング
│   ├── 09_normalize_cluster.py        # 正規化・クラスタリング
│   ├── 10_cell_typing.py              # 細胞種アノテーション
│   ├── 11_spatial_analysis.py         # 空間解析
│   ├── 12_compare_segmenters.py       # セグメンターの比較
│   ├── cosmx_utils.py                 # 共有ヘルパー関数
│   └── ssd_scripts/                   # パイロット探索・チューニングスクリプト
│       ├── step01_verify_setup.py     # 環境検証
│       ├── step02_enhance_images.py   # 4 FOVのTopHat強調処理
│       ├── step03_segmentation.py     # 4 FOVのCellposeセグメンテーション
│       ├── step04_qc_report.py        # QCサマリーレポート
│       ├── explore_0*.py              # チャンネル/z平面/モデルの探索
│       ├── proseg_*.py                # ProSeg 3D対応セグメンテーション実験
│       └── cosmx_seg_tuner*.py        # パラメータグリッドサーチの自動化
│
├── outputs/                           # メインパイプラインの出力
│   ├── anndata/                       # .h5adファイル（標準 + カスタムセグメンテーション）
│   ├── enhanced_images/               # TopHat強調済みTIFF
│   ├── pilot_4fov/                    # パイロットセグメンテーション結果
│   │   ├── enhanced/                  # 強調済みTIF
│   │   ├── masks/                     # 核マスクTIF
│   │   ├── expanded_masks/            # 拡張（細胞質）マスク
│   │   ├── overlays/                  # 視覚的QCオーバーレイ
│   │   ├── cell_tables/               # 細胞ごとの形態学的CSV
│   │   └── summary/                   # QCサマリーテーブル
│   └── pilot_model_comparison*/       # Cellposeモデル比較結果
│
├── outputs_ssd/                       # パイロット探索の出力
│   ├── proseg_pilot/                  # ProSeg 3Dセグメンテーション結果
│   ├── cellprofiler_pilot/            # CellProfilerとの比較
│   ├── pilot_segmentation*/           # 初期セグメンテーション実行結果
│   └── explore_0*/                    # チャンネル/z平面の探索出力
│
├── results/                           # 下流の生物学的解析（R + Python）
│   ├── 02A_build_protein_seurat/      # Seuratタンパク質オブジェクト
│   ├── 02B_build_rna/                 # RNA発現オブジェクト
│   ├── 03A_rna_brain_analysis/        # 細胞種マーカー解析
│   ├── 03B_protein_brain_aware/       # タンパク質空間解析
│   ├── 04_integrate_rna_protein_by_fov/  # マルチモーダル統合
│   └── 08_each_FOV_highres_AIF1/     # FOVごとの高解像度可視化
│
└── notebooks/                         # 探索用Jupyterノートブック
```

---

## 4. 環境構築

### 4.1 Conda環境

```bash
# 専用環境の作成（推奨）
conda create -n cosmx_env python=3.10
conda activate cosmx_env

# 主要依存パッケージ
pip install cellpose[gui]         # cpsamサポート付きCellpose
pip install stardist              # StarDist
pip install instanseg             # InstanSeg
pip install anndata scanpy squidpy
pip install tifffile scikit-image matplotlib seaborn
pip install pandas numpy scipy
pip install harmonypy             # バッチ補正
pip install openpyxl              # Excelメタデータ読み込み
pip install pyyaml
```

R言語による下流解析の場合：
```r
install.packages("Seurat")
BiocManager::install(c("scran", "scater"))
install.packages("ggplot2")
```

### 4.2 環境の確認

```bash
conda activate cosmx_env
cd /home/shamim/CosMx_2026
python scripts/00_check_env.py
```

GPUの利用可能状況、全パッケージバージョン、4 FOVデータパスの正常解決を確認します。

---

## 5. クイックスタート — 4 FOVパイロット解析

以下の4つのステップスクリプトを順番に実行します。出力はすべて `outputs/pilot_4fov/` に保存されます。

```bash
conda activate cosmx_env
cd /home/shamim/CosMx_2026

# ステップ1 — データパスとパッケージの検証
python scripts/ssd_scripts/step01_verify_setup.py

# ステップ2 — TopHat画像強調処理（DAPIチャンネル）
python scripts/ssd_scripts/step02_enhance_images.py

# ステップ3 — Cellpose cpsamセグメンテーション + CosMxマスクとの比較
python scripts/ssd_scripts/step03_segmentation.py

# ステップ4 — QCレポート（細胞数、サイズ分布、オーバーレイ）
python scripts/ssd_scripts/step04_qc_report.py
```

予想実行時間：GPU使用でFOVあたり約5分、合計約20分。

---

## 6. セグメンテーションパイプライン — 詳細ガイド

セグメンテーションは下流解析の品質を左右する最重要ステップです。セグメンテーションの精度が低いと、トランスクリプトが細胞間で混入し、カウント値が過大評価され、生物学的シグナルが偽陽性となります。

### 6.1 カスタムセグメンテーションが必要な理由

CosMx標準セグメンテーション（InSituType / 標準CellLabels）は、平均的な組織向けに調整された固定アルゴリズムを使用します。脳組織では以下の問題が生じます：
- 細胞数の過少評価（パイロットで約341細胞/FOV vs QuPath約947）
- 隣接するニューロンの統合
- 小さなグリア細胞の見逃し

Cellpose cpsamを用いたカスタムセグメンテーションでは、パイロットで**FOVあたり約1,200〜1,300細胞**を検出（CosMx標準の3〜4倍）。

### 6.2 画像強調処理 — ステップ2

**最適手法（パイロットより）：ホワイトTop-Hatフィルター、半径20 px**

```python
# 実装: scripts/02_image_enhance.py
# パラメータ: config/study_config.yaml → enhancement セクション
method: "TopHat"
tophat_radius_px: 20
```

Top-Hatフィルターは、明るい核を保持しながら不均一なバックグラウンド照明を抑制します。CosMxのDAPI画像は空間的に変化するバックグラウンドを持つため、このフィルターは特に有効です。CLAHEのみは使用しないでください — 前頭皮質などの細胞密度が低い領域で暗い核が過飽和になります。

| 手法 | 備考 |
|---|---|
| **TopHat r=20** | 密度の高い海馬・密度の低い前頭皮質の両方で最良 |
| TopHat r=10 | 小さすぎる — 大型ニューロンを見逃し、人工的なカウント増加を招く |
| Rolling Ball | 細胞質シグナルには有用だが、核のみのセグメンテーションには不向き |
| CLAHEのみ | 過飽和が生じる。非常に暗いFOVの後処理にのみ使用 |

4パイロットFOVの実行：
```bash
python scripts/02_image_enhance.py --slide 1 --fovs FOV00001 FOV00007 FOV00037 FOV00043
```

出力：`outputs/enhanced/<slide_id>/FOV*_enhanced.tif`

### 6.3 Cellposeセグメンテーション — ステップ3

**最適モデル（パイロットより）：`cpsam`（Cellpose Segment Anythingモデル）**

```python
# パラメータ: config/study_config.yaml → segmentation.cellpose
model:              cpsam
diameter:           25          # px；0.12 µm/pxで約3 µmの核
cellprob_threshold: -1.0        # 低いほど許容度が高い（暗い核も検出）
flow_threshold:     0.6
channel:            0           # DAPI
expand_distance_px: 5           # トランスクリプト割り当てのため核マスクを細胞質に拡張
```

> **重要：** `diameter=25` はCosMxでのヒト脳に対して専用チューニングされた値です。0.12 µm/pxでは、ニューロンは直径約15〜45 µm（125〜375 px）ですが、cpsamはdiameterを初期スケールのヒントとして使用します（ハードカットオフではありません）。35を超えると小さなグリア細胞を見逃します。

強調済み画像でCellposeを実行：
```bash
python scripts/03_segment_cellpose.py \
    --model cpsam \
    --fovs FOV00001 FOV00007 FOV00037 FOV00043
```

全3モデルを実行して比較：
```bash
python scripts/03_segment_cellpose.py --model cpsam nuclei cyto3
python scripts/12_compare_segmenters.py
```

### 6.4 代替セグメンター（比較用）

| セグメンター | スクリプト | 備考 |
|---|---|---|
| StarDist | `04_segment_stardist.py` | 円形核に強い；cpsamよりやや高速；CosMx解像度にはscale=2.0が必要 |
| InstanSeg | `05_segment_instanseg.py` | マルチチャンネル対応；膜チャンネルが強い場合に有用 |
| Mesmer | `ssd_scripts/02_pilot_mesmer_comparison_v1.py` | TensorFlow必要；パイロットでテスト済み |
| CellProfiler | `ssd_scripts/cellprofiler_pilot/` | 古典的手法；再現性が高い；最適直径：15〜55 px |

**パイロット比較結果（`outputs/pilot_model_comparison_v2/` に保存）：**
- cpsam：最高の細胞回収率、ニューロン境界の精度が最良
- StarDist：海馬では同等だが、前頭皮質のグリア細胞を見逃す
- Mesmer：高密度領域で過剰分割
- InstanSeg：実験的；タンパク質共染色で有望

### 6.5 マスクの拡張

核セグメンテーション後、マスクを5 px（`expand_distance_px: 5`、約0.6 µm）拡張し、核周囲の細胞質トランスクリプトを捕捉します。これは以下のために必須です：
- 細胞質mRNAを正しい細胞に割り当てる
- 細胞あたりのトランスクリプト捕捉を改善する

密度の高い脳領域では8 px以上の拡張は行わないでください — 隣接細胞の領域が重複します。

### 6.6 セグメンテーションQC指標

セグメンテーション後、`outputs/*/summary/` の以下の指標を確認してください：

| 指標 | 許容範囲 | 範囲外の場合の対処 |
|---|---|---|
| FOVあたり細胞数 | 400〜2000 | diameter・強調処理を確認 |
| 細胞面積中央値（px²） | 300〜3000 | diameterを再チューニング |
| 細胞直径中央値（px） | 20〜60 | diameterパラメータを調整 |
| 50 px²未満の細胞 | 5%未満 | `min_cell_area_px` を増加 |
| CosMxとの境界Dice係数 | > 0.3 | 想定内 — CosMxは過少評価 |

---

## 7. トランスクリプト割り当て

カスタムセグメンテーション後、CosMxトランスクリプト（`tx_file.csv.gz` から）を新しい細胞マスクに再割り当てする必要があります。発現行列中のCosMx標準細胞IDは**旧CellLabels**に対応しており、新しいマスクには対応していません。

```bash
python scripts/06_assign_transcripts.py \
    --seg_dir outputs/segmentation/cellpose/cpsam_d25/expanded/ \
    --fovs FOV00001 FOV00007 FOV00037 FOV00043
```

**割り当て手法：`mask_lookup`** — 各トランスクリプト（x, y）に対して拡張マスクのピクセル値を参照します。高密度空間データに対して最も正確な手法です。

**自動適用される品質フィルター：**
- `min_qv: 20` — 低信頼度のトランスクリプトコールを除去（Phredスケール品質値）
- ネガティブプローブトランスクリプトを除外（`NegPrb*`）
- FalseCodeトランスクリプトを除外（`FalseCode*`）

出力：`outputs/transcript_assignment/<seg_tag>/`

---

## 8. AnnDataの構築とQC

### 8.1 CosMx標準からの構築（高速スタート）

```bash
python scripts/01_load_flatfiles.py
```

CosMx標準細胞割り当てを使用して `outputs/anndata/2511276KSlide116_native.h5ad` を生成します。迅速な探索に有用ですが、細胞数は過少評価されます。

### 8.2 カスタムセグメンテーションからの構築（論文掲載推奨）

```bash
python scripts/07_build_anndata.py \
    --assign_dir outputs/transcript_assignment/cpsam_d25/ \
    --cell_tables_dir outputs/segmentation/cellpose/cpsam_d25/cell_tables/
```

### 8.3 QCフィルタリング

```bash
python scripts/08_qc_filter.py \
    --input outputs/anndata/2511276KSlide116_native.h5ad
```

**QC閾値（設定ファイルより）：**

| 指標 | 最小値 | 最大値 | 根拠 |
|---|---|---|---|
| 細胞あたりトランスクリプト数 | 5 | 5000 | 空のセル/ダブレットを除去 |
| 細胞あたり遺伝子数 | 3 | — | 意味のある発現の最小値 |
| ネガティブプローブ比率 | — | 10% | 高バックグラウンド細胞をフラグ |
| 細胞面積 | 10 µm² | 8000 µm² | デブリと組織の折り畳みを除去 |

脳細胞は上皮組織に比べて**トランスクリプト数が少ない**傾向があります（ニューロンで典型的に200〜800 transcripts/細胞、上皮では1000以上）。上皮組織向けの閾値を脳データに適用しないでください。

---

## 9. クラスタリングと細胞種アノテーション

```bash
# 正規化とクラスタリング
python scripts/09_normalize_cluster.py \
    --input outputs/anndata/<stem>_qc.h5ad \
    --norm scran \
    --batch_key sample_id

# 細胞種アノテーション
python scripts/10_cell_typing.py \
    --input outputs/analysis/<stem>/adata_clustered.h5ad
```

### 9.1 正規化の推奨

CosMxデータには**scran正規化**を使用してください — ゼロインフレートで疎なカウント分布を単純なlog1pより適切に扱え、単一細胞RNA-seqの論文における標準的手法です。

### 9.2 脳細胞種マーカー遺伝子

`10_cell_typing.py` では以下のマーカー遺伝子を脳細胞種アノテーションに使用します：

| 細胞種 | 主要マーカー |
|---|---|
| ニューロン | *RBFOX3, MAP2, SNAP25, SYP, NEUN* |
| アストロサイト | *GFAP, AQP4, ALDH1L1, S100B* |
| オリゴデンドロサイト | *MBP, PLP1, MOG, OLIG2* |
| OPC（オリゴデンドロサイト前駆細胞） | *PDGFRA, CSPG4, NG2* |
| ミクログリア | *AIF1 (IBA1), CX3CR1, P2RY12, TMEM119* |
| 内皮細胞 | *CLDN5, PECAM1, FLT1* |
| ペリサイト / VSMC | *ACTA2, PDGFRB, RGS5* |
| AD特異的 | *APOE, CLU, TREM2, C1Q*（ミクログリアにおける補体活性化） |

### 9.3 バッチ補正

2枚のスライド（各72 FOV）では、スライド間のバッチエフェクトが予想されます。`batch_key: "sample_id"` を指定した**Harmony**バッチ補正を適用してください。生物学的グループとスライドが交絡している場合、スライドIDのみをバッチキーとして使用しないでください。

---

## 10. 空間プロテオミクス解析

タンパク質スライド（251121Slide116）は**多重免疫蛍光染色**により約10〜20種のタンパク質マーカーを提供します。これはRNAパネルとは別に解析した後、FOVレベルで統合します。

### 10.1 主要タンパク質マーカー

パイロット結果（`results/03B_protein_brain_aware/` および `results/04_integrate_rna_protein_by_fov/`）より：

| タンパク質 | 細胞種 | パイロットでの主要知見 |
|---|---|---|
| AIF1（IBA1） | ミクログリア | ADで上昇、RNA *AIF1* と相関 |
| CD68 | 活性化ミクログリア | AD前頭皮質で高値 |
| HLA-DR | 抗原提示ミクログリア | ADでC1QAと共上昇 |
| GFAP | 反応性アストロサイト | AD海馬で上方制御 |
| Fibronectin | ECM/反応性グリア | ADで上昇 |
| CLDN5 | BBB内皮細胞 | ADで低下 |
| SMA（ACTA2） | ペリサイト/VSMC | 血管リモデリングマーカー |

### 10.2 タンパク質パイプライン

```bash
# Seuratタンパク質オブジェクトの構築（R）
# results/02A_build_protein_seurat/

# タンパク質セグメンテーションはRNAと同じ形態TIFを使用
# （タンパク質・RNAスライドは同一組織切片から作製）
python scripts/ssd_scripts/step02_enhance_images.py  # タンパク質スライド用
python scripts/03_segment_cellpose.py --model cpsam   # 同一パラメータ

# 細胞ごとのタンパク質輝度定量
# → 拡張マスクを使用して各タンパク質チャンネルの平均輝度を抽出
```

### 10.3 タンパク質解析ステップ

1. **細胞のセグメンテーション** — タンパク質形態画像上（RNAと同じDAPIベースの手法）
2. **細胞ごとの輝度抽出** — 各タンパク質チャンネルで拡張マスクを使用
3. **正規化** — タンパク質データにはCLR（Centered Log-Ratio）正規化を使用（scranは使用しない）
4. **クラスタリング** — タンパク質データを独立してクラスタリング；RNAベースのクラスターと比較
5. **空間可視化** — 組織座標上にタンパク質輝度マップを重ねて表示

---

## 11. 空間トランスクリプトミクス解析

```bash
python scripts/11_spatial_analysis.py \
    --input outputs/analysis/<stem>/adata_annotated.h5ad
```

### 11.1 実施する解析

| 解析 | ツール | 目的 |
|---|---|---|
| 空間近傍グラフ | Squidpy | すべての空間統計の基盤 |
| 近傍エンリッチメント | Squidpy | どの細胞種が共局在するか？ |
| RipleyのL関数 | Squidpy | ランダムとの比較による空間クラスタリング |
| MoranのI | Squidpy | 空間的に変動する遺伝子 |
| 共発現相関 | カスタム実装 | 遺伝子間の空間的相関 |

### 11.2 空間的変動遺伝子（SVG）

MoranのIを用いて、空間位置により発現が有意に変動する遺伝子を同定します：

```python
import squidpy as sq
sq.gr.spatial_neighbors(adata)
sq.gr.spatial_autocorr(adata, mode="moran")
```

AD特異的解析では、アミロイド豊富な領域（組織学画像との位置合わせ、または *APOE/CLU* 高発現領域で同定）の空間的変動遺伝子に焦点を当ててください。

### 11.3 FOVレベルの空間コンテキスト

各FOVは約0.8 × 0.8 mmの組織をカバーします。144 FOVの全実験では、脳切片の大部分をカバーできます。全144 FOVへのスケールアップ時：
- FOV位置は `fov_positions_file.csv.gz` に記録 — グローバル空間座標として使用
- 長距離空間パターンを捉えるため、**スライドレベル**（FOV単位ではなく）での解析を実施

---

## 12. RNA–タンパク質統合解析

論文発表に向けて、RNAと隣接切片のタンパク質データを統合し、トランスクリプトームレベルの細胞同定とタンパク質レベルの検証を結び付けます。

### 12.1 対応付けの戦略

RNAスライドとタンパク質スライドは**同一組織ブロックの連続切片**から作製されています。細胞レベルの直接対応付けは不可能（異なる切片のため）ですが、FOVレベルの統合は有効です：

```python
# FOVレベルでのマージ
# results/04_integrate_rna_protein_by_fov/
# 主要出力: joint_FOV_summary.tsv
```

### 12.2 パイロットで検証されたRNA–タンパク質相関

パイロット解析により強いRNA–タンパク質相関が確認されました（`results/04_integrate_rna_protein_by_fov/scatter_*.png` 参照）：

| RNA遺伝子 | タンパク質マーカー | Pearson r | 細胞種 |
|---|---|---|---|
| *AIF1* | AIF1（IBA1）タンパク質 | ~0.85 | ミクログリア |
| *C1QA* | HLA-DR | ~0.78 | 活性化ミクログリア |
| *ACTA2* | SMA | ~0.91 | ペリサイト/VSMC |
| *CLDN5* | Fibronectin | ~−0.6 | BBB完全性（逆相関） |
| *APOE* | IL-1β | ~0.72 | AD神経炎症 |

これらの相関は、セグメンテーションおよびトランスクリプト割り当てパイプラインが生物学的に整合した結果を生成していることを検証します。

---

## 13. 全144 FOVへのスケールアップ

4 FOVパイロットパイプラインが検証されたら、外付けSSD（`/mnt/external/data/`）上の全データセットへスケールアップします。

### 13.1 変更点

| ステップ | 4 FOVパイロット | 144 FOV全実行 |
|---|---|---|
| データソース | `raw_data/pilot_4fov/`（ローカル） | `/mnt/external/data/`（SSD） |
| 設定 `ssd_root` | `raw_data/pilot_4fov` | `/mnt/external/data` |
| 設定 `flat_dir` | `slide1_RNA/flatfiles` | `Study20251223_GOlab_cosmxrna1/flatFiles/2511276KSlide116` |
| `--fovs` 引数 | FOV00001 FOV00007 FOV00037 FOV00043 | 省略（全FOVを処理） |
| バッチ補正 | 不要（1サンプル） | 必須（Harmony、batch_key=sample_id） |
| 実行時間 | 約20分 | 約8〜12時間（GPU使用） |

### 13.2 全実行用の設定切り替え

`config/study_config.yaml` を編集：

```yaml
paths:
  ssd_root: "/mnt/external/data"   # ← raw_data/pilot_4fov から変更

  slide1:
    flat_dir:      "Study20251223_GOlab_cosmxrna1/flatFiles/2511276KSlide116"
    cell_stats_dir: "Study20251223_GOlab_cosmxrna1/DecodedFiles/2511276KSlide116/20251127_074826_S1/CellStatsDir"
    morphology2d:  "Study20251223_GOlab_cosmxrna1/DecodedFiles/2511276KSlide116/20251127_074826_S1/CellStatsDir/Morphology2D"
```

### 13.3 バッチ処理スクリプト

```bash
# 全144 FOVのバッチ処理（外付けSSDのマウントが必要）
python scripts/ssd_scripts/03_batch_cellpose_all_144_ch2.py
```

### 13.4 全実行の推奨実行順序

```
00_check_env.py
↓
01_load_flatfiles.py          （両スライド）
↓
02_image_enhance.py           （全144 FOV — GPU使用で約4時間）
↓
03_segment_cellpose.py        （cpsam、全FOV — 約6時間）
↓
06_assign_transcripts.py      （tx_fileを新マスクに割り当て）
↓
07_build_anndata.py
↓
08_qc_filter.py
↓
09_normalize_cluster.py       （Harmonyバッチ補正）
↓
10_cell_typing.py
↓
11_spatial_analysis.py
↓
12_compare_segmenters.py      （オプション：サブセットでcpsam vs StarDistを比較）
```

### 13.5 並列化

144 FOVの処理では、画像強調とセグメンテーションをFOVのバッチに分けて並列化します：

```bash
# 例：36 FOVずつ4バッチに分割
python scripts/02_image_enhance.py --fovs $(seq -f "FOV%05g" 1 36 | tr '\n' ' ')
python scripts/02_image_enhance.py --fovs $(seq -f "FOV%05g" 37 72 | tr '\n' ' ')
```

GPU1基で、強調処理 + セグメンテーション合計でFOVあたり約2分を想定してください。

---

## 14. パイロット解析の主要知見

4 FOVパイロットのこれらの結果は、パイプラインの妥当性を検証し、全144 FOV解析の生物学的コンテキストを提供します。

### 14.1 セグメンテーション性能

| 手法 | 細胞数/FOV（平均） | CosMx標準比 | 備考 |
|---|---|---|---|
| CosMx標準 | ~341 | 1×（参照） | 過少評価；固定アルゴリズム |
| QuPath（手動） | ~947 | 2.8× | 専門家によるカウント |
| **cpsam（TopHat r=20, d=25）** | **~1,200** | **3.5×** | **推奨** |
| Nucleiモデル（d=20） | ~980 | 2.9× | より保守的 |
| StarDist（scale=2.0） | ~1,050 | 3.1× | 同等；やや低速 |
| CellProfiler（d=15〜55） | ~880 | 2.6× | 古典的；再現性が高い |

### 14.2 観察された主要な生物学的シグナル

- **ADにおけるミクログリア活性化：** 前頭皮質・海馬の両方で *AIF1*、*CD68*、*HLA-DR*（RNAおよびタンパク質）が有意に上昇。海馬でより顕著。
- **補体カスケードの上方制御：** *C1QA*、*C1QB*、*C1QC* がADミクログリアクラスターで上昇 — アミロイド領域周囲に空間的クラスターを形成。
- **アストロサイトの反応性：** *GFAP*、*STAT3*、*LCN2* がADで上昇。反応性アストロサイトはミクログリアクラスター周囲に空間的ニッチを形成。
- **BBB完全性の喪失：** *CLDN5*（タイトジャンクション）がAD内皮細胞で低下；*Fibronectin* タンパク質と逆相関。
- **海馬のニューロン喪失：** FOV00007（AD-H）ではFOV00043（対照-H）と比較してニューロン細胞数が全体的に低値。

### 14.3 論文発表に向けたパイプラインの推奨事項

パイロット解析に基づく推奨：

1. **cpsam + TopHat を使用** — 最高の細胞回収率、QuPath専門家カウントとの最良の相関
2. **CosMx標準細胞を主解析に使用しない** — カスタムセグメンテーションで論文発表
3. **補足資料に標準・カスタムの両方を報告**（透明性のため）
4. **トランスクリプト割り当て効率の目標：** 細胞に割り当てられたトランスクリプト > 70%（CosMx標準は約55%）
5. **コンパートメント情報の活用：** `CompartmentLabels` を使用して、核局在型RNAの核・細胞質トランスクリプトを区別

---

## 15. 論文投稿チェックリスト

投稿前に以下を確認してください：

### データと再現性
- [ ] 全パラメータを `config/study_config.yaml` に文書化
- [ ] 最適セグメンテーションパラメータを `config/seg_best_params.json` に記録
- [ ] `sample_manifest.csv` を全144 FOVのメタデータで更新
- [ ] パイプラインをバージョン管理（解析凍結時にgitタグ付け）
- [ ] 生データを適切なリポジトリに寄託（GEO / Zenodo）

### セグメンテーション
- [ ] TopHat r=20を用いたcpsamを全FOVに適用
- [ ] 全FOVのQCテーブル生成（細胞数、サイズ、面積分布）
- [ ] 少なくとも10%のFOVで視覚的オーバーレイを確認
- [ ] CosMx標準セグメンテーションとの比較を提示（`12_compare_segmenters.py`）

### トランスクリプトミクス
- [ ] `tx_file.csv.gz` からカスタムトランスクリプト割り当て実施（CosMx標準割り当てではない）
- [ ] 品質フィルター適用：QV ≥ 20、ネガティブプローブ除外
- [ ] scran正規化適用
- [ ] Harmonyバッチ補正（サンプル/スライドで補正）
- [ ] 確立された脳マーカーで細胞種アノテーション完了
- [ ] 空間的変動遺伝子を同定（MoranのI）

### プロテオミクス
- [ ] 細胞ごと・チャンネルごとにタンパク質輝度を抽出
- [ ] CLR正規化適用
- [ ] RNA–タンパク質相関を検証（`results/04_integrate_rna_protein_by_fov/` 参照）

### 図表
- [ ] 細胞種・条件・FOVで色分けしたUMAP
- [ ] 組織座標上の細胞種空間マップ
- [ ] 細胞種ごとのAD vs 対照の差次発現（ボルカノプロット）
- [ ] 近傍エンリッチメントヒートマップ（Squidpy）
- [ ] 主要マーカーのRNA–タンパク質散布図

---

## 連絡先・参照情報

**研究室：** 京都大学 GO研究室  
**研究：** 認知症CosMx 2026 — ヒトAD脳の空間マルチオミクス  
**データ形式：** NanoString CosMx SMI（FlatFileエクスポート）  
**パイプライン：** カスタムPython（Cellpose + Squidpy）+ R（Seurat）

パイプラインに関するご質問は、スクリプトのdocstring（`head -20 scripts/<スクリプト名>.py`）および設定ファイル `config/study_config.yaml` を参照してください。
