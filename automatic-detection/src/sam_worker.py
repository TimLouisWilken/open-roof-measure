"""
SAM (Segment Anything) worker — runs in the dedicated `sam_env` conda env
(numpy<2 + torch 2.2.2).  Lives as a persistent subprocess of the main
`roof_tool.py`; communicates line-by-line over stdin/stdout with JSON messages.

Protocol
--------
Request (one JSON object per line, line-delimited):
  {
    "npy_path":  "/tmp/.../dop.npy",   # HxWx3 uint8 numpy array on disk
    "points":    [[x1, y1], [x2, y2]], # foreground (1) / background (0)
    "labels":    [1, 1, 0],
    "box":       [x0, y0, x1, y1],     # optional XYXY bounding box
    "out_path":  "/tmp/.../mask.npy",  # writes best mask here as bool
    "multimask": true,                  # return 3 candidate masks
    "select":    "largest" | "highest_score"  # default: highest_score
  }

Response (one JSON per line):
  { "ok": true,  "out_path": "...", "scores": [0.95, 0.88, 0.77] }
  { "ok": false, "error": "..." }

Sentinel line `{"cmd": "shutdown"}` terminates the worker cleanly.
"""
import sys, os, json, traceback
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Import numpy/torch/SAM — this is the expensive part (~3 s on CPU)
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

SAM_CKPT_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "sam_vit_b_01ec64.pth",
)


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _handle(req, predictor, cached_img_path):
    """Process one request. Returns (response_dict, updated_cached_img_path)."""
    npy_path = req.get("npy_path")
    if not npy_path or not os.path.exists(npy_path):
        return {"ok": False, "error": f"npy_path not found: {npy_path}"}, cached_img_path

    # Re-use previously-set image if same path (avoids re-running the encoder)
    if npy_path != cached_img_path:
        img = np.load(npy_path)
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            return {"ok": False,
                    "error": f"expected HxWx3 uint8, got {img.shape} {img.dtype}"}, cached_img_path
        predictor.set_image(img)
        cached_img_path = npy_path

    # Prepare prompts
    points = req.get("points") or []
    labels = req.get("labels") or [1] * len(points)
    box    = req.get("box")

    point_coords = np.array(points, dtype=np.float32).reshape(-1, 2) if points else None
    point_labels = np.array(labels, dtype=np.int32)          if points else None
    box_arr      = np.array(box,    dtype=np.float32) if box else None

    multimask = bool(req.get("multimask", True))

    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_arr,
            multimask_output=multimask,
        )

    # Select a mask:
    select = req.get("select", "highest_score")
    if select == "largest":
        idx = int(np.argmax(masks.sum(axis=(1, 2))))
    else:
        idx = int(np.argmax(scores))

    best = masks[idx].astype(bool)

    out_path = req.get("out_path")
    if out_path:
        np.save(out_path, best)

    return {
        "ok":       True,
        "out_path": out_path,
        "scores":   [float(s) for s in scores],
        "picked":   idx,
    }, cached_img_path


def main():
    ckpt = os.environ.get("SAM_CKPT", SAM_CKPT_DEFAULT)
    if not os.path.exists(ckpt):
        _emit({"ok": False, "error": f"SAM checkpoint missing: {ckpt}"})
        return 1

    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.eval()
    predictor = SamPredictor(sam)
    _emit({"ok": True, "ready": True, "model": "vit_b", "ckpt": ckpt})

    cached_img_path = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"ok": False, "error": f"bad json: {e}"})
            continue

        if req.get("cmd") == "shutdown":
            _emit({"ok": True, "shutdown": True})
            return 0

        try:
            resp, cached_img_path = _handle(req, predictor, cached_img_path)
            _emit(resp)
        except Exception:
            _emit({"ok": False, "error": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
