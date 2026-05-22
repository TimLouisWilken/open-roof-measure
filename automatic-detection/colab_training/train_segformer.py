"""
SegFormer Fine-Tuning auf RID2 — Roof Superstructure Segmentation
==================================================================
Für Google Colab Free (NVIDIA T4).

ANLEITUNG
---------
1. colab_data.zip auf Google Drive hochladen (eine Datei, ~2.1 GB):
     My Drive/colab_data.zip

2. In Colab: neues Notebook, Runtime → GPU (T4)

3. Zelle für Zelle einfügen und ausführen.
   Zelle 1 mountet Drive und entpackt den Zip auf die lokale SSD.

4. Bei Session-Timeout: RESUME_FROM anpassen und neu starten.
   (Daten werden jedes Mal neu entpackt — dauert ~30s)

5. Nach Training: best_model.pt vom Drive auf den Mac kopieren
   → in den Ordner "models/" des Projekts.

Klassen:
  0 = PV Module, 1 = Dormer, 2 = Window,
  3 = Balcony, 4 = Other (inkl. Chimney), 5 = Background
"""

# ============================================================
# ZELLE 1: Drive mounten + Zip entpacken + Dependencies
# ============================================================
# Diese Zeilen in die ERSTE Colab-Zelle:
"""
from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers

# Zip von Drive auf lokale SSD entpacken (viel schneller als direkt von Drive lesen)
import zipfile, os
zip_path = "/content/drive/MyDrive/Built Environment /colab_data.zip"
extract_to = "/content/rid2_data"
if not os.path.exists(extract_to + "/images"):
    print("Entpacke colab_data.zip ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"Fertig! {len(os.listdir(extract_to + '/images'))} Bilder entpackt.")
else:
    print("Daten bereits entpackt.")
"""

# ============================================================
# ZELLE 2: Konfiguration
# ============================================================
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Daten auf lokaler SSD (entpackt aus Zip) ──
DATA_ROOT = Path("/content/rid2_data")
# ── Checkpoints auf Drive (überleben Session-Timeout) ──
CKPT_DIR  = Path("/content/drive/MyDrive/rid2_checkpoints")

IMG_DIR   = DATA_ROOT / "images"
MASK_DIR  = DATA_ROOT / "masks"
TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV  = DATA_ROOT / "test.csv"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameter ──
NUM_CLASSES   = 6
BATCH_SIZE    = 4
NUM_EPOCHS    = 20
LEARNING_RATE = 6e-5
IMG_SIZE      = 512

# ── Fortsetzen nach Session-Timeout ──
# Beim ersten Start: None
# Nach Timeout z.B.: "checkpoint_epoch_08.pt"
RESUME_FROM = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Daten:  {DATA_ROOT}")
print(f"Images: {len(list(IMG_DIR.glob('*.png')))} Dateien")


# ============================================================
# ZELLE 3: Dataset
# ============================================================
class RID2Dataset(Dataset):
    def __init__(self, image_names, img_dir, mask_dir, augment=False):
        self.image_names = image_names
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img = Image.open(self.img_dir / name).convert("RGB")
        mask = np.array(Image.open(self.mask_dir / name), dtype=np.int64)

        if self.augment:
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = np.flipud(mask).copy()
            k = np.random.randint(4)
            if k > 0:
                img = img.rotate(90 * k)
                mask = np.rot90(mask, k).copy()

        return self.img_transform(img), torch.from_numpy(mask).long()


def load_split(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    names = df["image_names"].tolist()
    return [n for n in names if (IMG_DIR / n).exists()]

train_names = load_split(TRAIN_CSV)
test_names  = load_split(TEST_CSV)
print(f"Train: {len(train_names)} | Test: {len(test_names)}")

train_loader = DataLoader(
    RID2Dataset(train_names, IMG_DIR, MASK_DIR, augment=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    pin_memory=True, drop_last=True)

test_loader = DataLoader(
    RID2Dataset(test_names, IMG_DIR, MASK_DIR, augment=False),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    pin_memory=True)


# ============================================================
# ZELLE 4: Modell laden
# ============================================================
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
model = model.to(DEVICE)

# Background dominiert (~95% der Pixel) → runtergewichten
class_weights = torch.tensor(
    [3.0, 5.0, 5.0, 3.0, 4.0, 0.3],
    dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                              weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# Checkpoint laden falls fortgesetzt
start_epoch = 0
best_miou = 0.0

if RESUME_FROM:
    ckpt_path = CKPT_DIR / RESUME_FROM
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"★ Fortgesetzt ab Epoch {start_epoch} (best mIoU: {best_miou:.4f})")


# ============================================================
# ZELLE 5: Training starten
# ============================================================
CLASS_NAMES = ["PV Module", "Dormer", "Window", "Balcony",
               "Other/Chimney", "Background"]


def compute_miou(preds, labels, num_classes=NUM_CLASSES):
    ious = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (labels == cls)).sum().item()
        union = ((preds == cls) | (labels == cls)).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for i, (pixels, labels) in enumerate(loader):
        pixels, labels = pixels.to(device), labels.to(device)
        logits = model(pixel_values=pixels).logits
        logits_up = nn.functional.interpolate(
            logits, size=labels.shape[-2:],
            mode="bilinear", align_corners=False)
        loss = criterion(logits_up, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 200 == 0:
            print(f"    Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_ious = 0, []
    for pixels, labels in loader:
        pixels, labels = pixels.to(device), labels.to(device)
        logits = model(pixel_values=pixels).logits
        logits_up = nn.functional.interpolate(
            logits, size=labels.shape[-2:],
            mode="bilinear", align_corners=False)
        total_loss += criterion(logits_up, labels).item()
        preds = logits_up.argmax(dim=1)
        all_ious.append(compute_miou(preds.cpu(), labels.cpu()))
    return total_loss / len(loader), np.mean(all_ious)


print(f"\n{'='*60}")
print(f"  SegFormer MIT-B2 auf RID2  |  {len(train_names)} Train")
print(f"  Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  Epochs: {NUM_EPOCHS}")
print(f"{'='*60}\n")

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_one_epoch(model, train_loader, criterion,
                                 optimizer, DEVICE)
    val_loss, val_miou = evaluate(model, test_loader, criterion, DEVICE)
    scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
          f"| Val mIoU: {val_miou:.4f} | LR: {lr:.2e}")

    # Checkpoint auf Drive speichern
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_miou": val_miou,
        "best_miou": max(best_miou, val_miou),
    }
    torch.save(ckpt, CKPT_DIR / f"checkpoint_epoch_{epoch+1:02d}.pt")
    print(f"  Checkpoint: checkpoint_epoch_{epoch+1:02d}.pt")

    if val_miou > best_miou:
        best_miou = val_miou
        torch.save(ckpt, CKPT_DIR / "best_model.pt")
        print(f"  ★ Neues bestes Modell! mIoU = {best_miou:.4f}")
    print()

print(f"Fertig! Bestes Modell: mIoU = {best_miou:.4f}")


# ============================================================
# ZELLE 6: Evaluation (nach Training oder nach Checkpoint laden)
# ============================================================
@torch.no_grad()
def evaluate_per_class(model, loader, device):
    model.eval()
    inter = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)
    for pixels, labels in loader:
        pixels = pixels.to(device)
        logits = model(pixel_values=pixels).logits
        logits_up = nn.functional.interpolate(
            logits, size=labels.shape[-2:],
            mode="bilinear", align_corners=False)
        preds = logits_up.argmax(dim=1).cpu().numpy()
        labels = labels.numpy()
        for c in range(NUM_CLASSES):
            inter[c] += ((preds == c) & (labels == c)).sum()
            union[c] += ((preds == c) | (labels == c)).sum()

    print(f"\n{'='*50}")
    print(f"  Per-Class IoU  ({len(loader.dataset)} Test-Bilder)")
    print(f"{'='*50}")
    for c in range(NUM_CLASSES):
        iou = inter[c] / max(union[c], 1)
        print(f"  {CLASS_NAMES[c]:20s}  IoU = {iou:.4f}")
    fg_ious = [inter[c] / max(union[c], 1) for c in range(5)]
    print(f"  {'mIoU (alle)':20s}      = "
          f"{np.mean([inter[c]/max(union[c],1) for c in range(6)]):.4f}")
    print(f"  {'mIoU (ohne BG)':20s}      = {np.mean(fg_ious):.4f}")

best = torch.load(CKPT_DIR / "best_model.pt", map_location=DEVICE)
model.load_state_dict(best["model_state_dict"])
evaluate_per_class(model, test_loader, DEVICE)


# ============================================================
# ZELLE 7: Export für Inference auf dem Mac
# ============================================================
export_path = CKPT_DIR / "segformer_rid2_superstructures.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "img_size": IMG_SIZE,
    "best_miou": best_miou,
}, export_path)
print(f"\nExport: {export_path}")
print("→ Diese Datei auf den Mac kopieren in: models/segformer_rid2_superstructures.pt")
