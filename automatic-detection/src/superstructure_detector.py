"""
Superstructure detection on NRW DOP using SegFormer trained on RID2.

Usage:
    from superstructure_detector import detect_superstructures
    result = detect_superstructures(dop_arr, mask, gsd=0.10)

The trained model (segformer_rid2_superstructures.pt) must be copied
from Google Drive after training.
"""

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

MODEL_PATH = Path(__file__).parent.parent / "models" / "rid2_checkpoints" / "segformer_rid2_superstructures.pt"

CLASS_NAMES = ["PV Module", "Dormer", "Window", "Balcony", "Other/Chimney", "Background"]
RELEVANT_CLASSES = {0: "PV Module", 1: "Dormer", 2: "Dachfenster", 4: "Chimney/Other"}

_model = None
_device = None


def _load_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    from transformers import SegformerForSemanticSegmentation
    ckpt = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
    num_classes = ckpt.get("num_classes", 6)

    _model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    _model.load_state_dict(ckpt["model_state_dict"])
    _model = _model.to(_device)
    _model.eval()
    return _model, _device


def detect_superstructures(dop_arr, mask, gsd=0.10):
    """Detect roof superstructures in a DOP chip.

    Parameters
    ----------
    dop_arr : (C, H, W) or (H, W, 3) uint8 — aerial photo
    mask : (H, W) bool — roof mask
    gsd : float — DOP ground sampling distance in m/px

    Returns
    -------
    dict with:
        class_masks: dict {class_name: (H, W) bool}
        counts: dict {class_name: int} — number of objects per class
        areas_m2: dict {class_name: float} — total area per class
    """
    model, device = _load_model()

    if dop_arr.ndim == 3 and dop_arr.shape[0] in (3, 4):
        rgb = dop_arr[:3].transpose(1, 2, 0)
    else:
        rgb = dop_arr[:, :, :3] if dop_arr.shape[2] >= 3 else dop_arr

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h_orig, w_orig = rgb.shape[:2]

    # RID2 is 8cm/px, NRW DOP is 10cm/px — resample to 512×512
    img_pil = Image.fromarray(rgb).resize((512, 512), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    pixel_values = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        logits_up = nn.functional.interpolate(
            logits, size=(512, 512),
            mode="bilinear", align_corners=False)
        pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()

    # Resize prediction back to original DOP size
    pred_pil = Image.fromarray(pred.astype(np.uint8)).resize(
        (w_orig, h_orig), Image.NEAREST)
    pred_full = np.array(pred_pil)

    # Apply roof mask
    pred_full[~mask] = 5  # background

    from scipy.ndimage import label as ndi_label

    class_masks = {}
    counts = {}
    areas_m2 = {}
    pixel_area = gsd * gsd

    MIN_AREA_M2 = {"Chimney/Other": 0.25, "Dachfenster": 0.30}

    for cls_id, cls_name in RELEVANT_CLASSES.items():
        cls_mask = (pred_full == cls_id)
        labeled, n_objects = ndi_label(cls_mask)
        min_a = MIN_AREA_M2.get(cls_name, 0)
        if min_a > 0 and n_objects > 0:
            min_px = int(min_a / pixel_area)
            for obj_id in range(1, n_objects + 1):
                if (labeled == obj_id).sum() < min_px:
                    cls_mask[labeled == obj_id] = False
            labeled, n_objects = ndi_label(cls_mask)
        class_masks[cls_name] = cls_mask
        counts[cls_name] = n_objects
        areas_m2[cls_name] = float(cls_mask.sum()) * pixel_area

    return {
        "class_masks": class_masks,
        "counts": counts,
        "areas_m2": areas_m2,
        "pred_full": pred_full,
    }
