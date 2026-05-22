# SegFormer fine-tuning on RID2 — Roof Superstructure Segmentation

Pipeline component: semantic segmentation of rooftop objects (PV modules,
dormers, skylights, balconies, chimneys) on the DOP 10 cm aerial imagery.

## What is in this folder

Only the training **script** is checked in (`train_segformer.py`).
**Excluded** from the repository (see `.gitignore`):

- `colab_data.zip` — 2.1 GB packed RID2 dataset, build from
  [Roof Information Dataset v2](https://github.com/TUMFTM/RID).
- `images/`, `masks/` — extracted dataset (3 813 train + 953 test image/mask pairs).
- `train.csv`, `test.csv` — index files for the splits.
- Resulting `.pt` checkpoints — large binaries.

## Setup

- Architecture: `nvidia/mit-b0` SegFormer backbone, 6-class head (PV / Dormer /
  Window / Balcony / Other / Background).
- Trained on Google Colab Free Tier (NVIDIA T4 GPU).
- AdamW, cosine LR schedule, `NUM_EPOCHS = 20`. The checkpoint used in this
  project was trained for 14 epochs; the script keeps the best checkpoint by
  validation mIoU (`best_model.pt`).
- Best checkpoint (~110 MB) is loaded by `src/superstructure_detector.py` at
  runtime to classify roof objects on user-queried buildings.

## Reproducing the training

1. Build `colab_data.zip` from RID2 (image / mask pairs in PNG, 512×512).
2. Upload it to Google Drive at `My Drive/colab_data.zip`.
3. Open a new Colab notebook, GPU runtime (T4), paste the cells from
   `train_segformer.py` (top of file documents the cell layout).
4. After training, copy `segformer_rid2_superstructures.pt` to `../models/rid2_checkpoints/`.
