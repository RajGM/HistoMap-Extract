# HistoMap-Extract

**A computer vision pipeline for extracting structured geospatial data from historical planning maps — designed to support large-scale digitisation in economic history and digital humanities research.**

---

## Motivation

Historical maps are among the richest untapped sources in economic and social history. Yet extracting structured, machine-readable information from scanned cartographic documents — building footprints, road networks, administrative boundaries, textual annotations — remains a labour-intensive bottleneck for researchers working at scale.

This project addresses that bottleneck directly. Drawing on the **ICDAR 2021 Historical Map Segmentation benchmark** — a dataset of Parisian municipal atlases (1894–1937) produced for urban planning — `HistoMap-Extract` implements an end-to-end computer vision pipeline that ingests scanned map images and outputs structured **GeoJSON**, ready for integration into GIS environments and quantitative historical analysis.

The pipeline is designed with the needs of digital humanities research in mind: interpretable outputs, documented failure modes, and evaluation against established benchmarks so that historians can assess reliability before trusting extracted data for analysis.

---

## What It Does

Given a scanned historical map, the pipeline produces:

| Output | Format | Method |
|---|---|---|
| Urban / building boundaries | GeoJSON polygons | U-Net semantic segmentation |
| Road network | GeoJSON linestrings | Morphological skeletonisation |
| Place name annotations | Structured CSV | TrOCR transformer OCR |

---

## Evaluation Results

Evaluated against the ICDAR 2021 benchmark protocol on the full validation set (6 images):

| Task | Metric | This Pipeline | ICDAR 2021 Winner |
|---|---|---|---|
| Map area segmentation (Task 2) | mIoU | **0.59** | 0.91 |
| Building block detection (Task 1) | F1 | **0.74** | 0.87 |
| Graticule localisation (Task 3) | RMSE (px) | — | 3.2 |

*Trained for 15 epochs on Task 2 only (26 training images, RTX 3050 Laptop GPU). Scores are expected to improve with full multi-task training and data augmentation.*

Training loss curve (15 epochs):

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.8693 | 0.9179 |
| 5 | 0.2565 | 0.2547 |
| 10 | 0.1669 | 0.1650 |
| 15 | 0.1475 | 0.1475 |

No overfitting observed — train and val loss converge cleanly.

---

## Dataset

This project uses the **ICDAR 2021 MapSeg Competition Dataset** (v1.0.0):

> Chazalon et al., *ICDAR 2021 Competition on Historical Map Segmentation*, ICDAR 2021.
> DOI: [10.5281/zenodo.4817662](https://doi.org/10.5281/zenodo.4817662)

The dataset covers historical atlases of the City of Paris (1894–1937), drawn at 1:5000 scale by the *Service du plan* for urban management — directly analogous to the British Ordnance Survey planning documents used in large-scale historical demographic and economic studies.

| Task | Description | Train / Val / Test |
|---|---|---|
| Task 1 | Building block detection | 1 / 1 / 3 |
| Task 2 | Map area segmentation | 26 / 6 / 95 |
| Task 3 | Graticule line intersection localisation | 26 / 6 / 95 |

**Expected folder structure:**
```
icdar21-mapsef-v.1.0.0/
├── 1-detbblocks/
│   ├── train/        *-INPUT.jpg, *-INPUT-MASK.png, *-OUTPUT-GT.png
│   ├── validation/
│   └── test/
├── 2-segmaparea/
│   ├── train/        *-INPUT.jpg, *-OUTPUT-GT.png
│   ├── validation/
│   └── test/
└── 3-locglinesinter/
    ├── train/        *-INPUT.jpg, *-OUTPUT-GT.csv
    ├── validation/
    └── test/
```

---

## Pipeline Architecture

```
Input: Scanned map image (.jpg / .tiff)
        │
        ▼
  Preprocessing
  (contrast normalisation, sliding window tiling at 512x512)
        │
        ▼
  Semantic Segmentation (U-Net, ResNet-34 encoder, ImageNet pretrained)
  → binary map area mask (background / map content)
        │
        ▼
  Vectorisation
  → polygonise()        — map area polygons (GeoJSON)
  → skeletonise_roads() — road centrelines (GeoJSON)
        │
        ▼
  OCR (TrOCR — microsoft/trocr-base-printed)
  → place names, labels, annotations → CSV
        │
        ▼
Output: boundaries.geojson + roads.geojson + place_names.csv + preview.png
```

---

## Relevance to Digital Humanities Research

This pipeline is directly motivated by the digitisation challenges faced by large-scale historical research projects including:

- **CAMPOP** (Cambridge Group for the History of Population and Social Structure) — extracting occupational, boundary, and infrastructure data from historical OS maps at scale
- **PopulationsPast / EconomiesPast** — georeferenced parish-level datasets requiring scalable map digitisation
- Planning document digitisation for public sector partners (e.g. MHCLG)

The GeoJSON output is compatible with CAMPOP's existing GIS framework and can be validated directly against parish-level boundary shapefiles (UK Data Archive SN-852232).

---

## Repository Structure

```
HistoMap-Extract/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── pipeline.py              ← end-to-end: image in -> GeoJSON + CSV out
├── train.py                 ← fine-tune segmentation model on ICDAR data
├── evaluate.py              ← compute mIoU, F1 against ground truth
│
├── models/
│   ├── __init__.py
│   ├── segmentation.py      ← U-Net model, training loop, inference
│   └── ocr.py               ← TrOCR wrapper, text region detection
│
├── utils/
│   ├── __init__.py
│   ├── georeference.py      ← mask -> GeoJSON, road skeletonisation
│   └── visualise.py         ← preview overlay generation
│
├── notebooks/
│   ├── 01_segmentation.ipynb
│   ├── 02_ocr.ipynb
│   └── 03_evaluation.ipynb
│
└── outputs/                 ← generated by pipeline.py (gitignored)
    ├── boundaries.geojson
    ├── roads.geojson
    ├── place_names.csv
    └── preview.png
```

---

## Installation

```bash
git clone https://github.com/RajGM/HistoMap-Extract
cd HistoMap-Extract

# Create conda environment (recommended — avoids Windows build issues)
conda create -n histomap python=3.11 -y
conda activate histomap

# Install PyTorch with CUDA (RTX 30xx series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
conda install -c conda-forge scikit-image opencv geopandas -y
pip install segmentation-models-pytorch transformers sentencepiece shapely pyproj matplotlib tqdm pandas scipy python-dotenv python-Levenshtein icdar21-mapseg-eval Pillow
```

**Configure paths** — copy `.env.example` to `.env`:
```
ICDAR_ROOT=F:/CAMPOP/icdar21-mapsef-v.1.0.0
MODEL_WEIGHTS=models/segmentation_weights.pth
OUTPUT_DIR=outputs/
```

---

## Usage

### Train
```bash
python train.py --data "path/to/icdar21-mapsef-v.1.0.0" --epochs 15 --device cuda
```

### Run Pipeline (single image)
```bash
python pipeline.py \
  --input "path/to/icdar21-mapsef-v.1.0.0/2-segmaparea/validation/201-INPUT.jpg" \
  --output outputs/ \
  --weights models/segmentation_weights.pth
```

### Run Pipeline (all validation images, PowerShell)
```powershell
$valDir = "F:/CAMPOP/icdar21-mapsef-v.1.0.0/2-segmaparea/validation"
Get-ChildItem "$valDir\*-INPUT.jpg" | ForEach-Object {
    $name = $_.BaseName -replace '-INPUT', ''
    python pipeline.py --input $_.FullName --output "outputs/$name/" --weights models/segmentation_weights.pth --no-ocr
}
```

### Evaluate
```bash
python evaluate.py --pred outputs/ --gt "path/to/icdar21-mapsef-v.1.0.0"
```

---

## Environment

- Python 3.11 (conda)
- PyTorch 2.3+ with CUDA 12.1
- Tested on Windows 11, NVIDIA RTX 3050 Laptop GPU (4GB VRAM)

---

## Citation

If you use this pipeline, please cite the ICDAR 2021 dataset:

```bibtex
@InProceedings{chazalon.21.icdar.mapseg,
  author    = {Joseph Chazalon and Edwin Carlinet and Yizi Chen and Julien Perret
               and Bertrand Dum{\'e}nieu and Cl{\'e}ment Mallet and Thierry G{\'e}raud
               and Vincent Nguyen and Nam Nguyen and Josef Baloun
               and Ladislav Lenc and Pavel Kr{\'a}l},
  title     = {ICDAR 2021 Competition on Historical Map Segmentation},
  booktitle = {Proceedings of the 16th International Conference on
               Document Analysis and Recognition (ICDAR'21)},
  year      = {2021},
  address   = {Lausanne, Switzerland},
}
```

---

## Related Work

- Chazalon et al. (2021). *ICDAR 2021 Competition on Historical Map Segmentation.* ICDAR.
- Petitpierre (2025). *Generalizable Multiscale Segmentation of Heterogeneous Map Collections.* EPFL DHLAB.
- Chen et al. (2023). *Cross-attention Spatio-temporal Context Transformer for Semantic Segmentation of Historical Maps.* ACM SIGSPATIAL.
- Wen et al. (2023). *From Historical Maps to Geospatial Data: A Review of Automated Extraction Methods.* IJGIS.

---

## Author

**Raj Gaurav Maurya**

Research: Computer vision for document digitisation · AI for social impact · 
Computational auditing of generative vision-language models

[rajgm.com](https://rajgm.com) · [github.com/RajGM](https://github.com/RajGM) · [linkedin.com/in/rajgm29](https://linkedin.com/in/rajgm29)

---

## Licence

MIT — open for academic use and extension.