"""Visualize cached RMN lines (raw after filter) + classified result side by side."""
import os, sys, json
import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'src'))

from wireframe_fusion import FEAT_SIZE

BUILDINGS = {
    'S1': dict(id='S1', address='[Site S1]', gsd=0.5),
    'S2': dict(id='S2', address='[Site S2]', gsd=0.5),
    'S3': dict(id='S3', address='[Site S3]', gsd=0.5),
}

for vid in (sys.argv[1:] if len(sys.argv) > 1 else ['S1', 'S2', 'S3']):
    out_dir = os.path.join(BASE, 'results', 'validation', vid)
    cache_path = os.path.join(out_dir, 'rmn_cache.npz')
    dop_path = os.path.join(out_dir, 'dop.jpg')

    if not os.path.exists(cache_path):
        print(f"{vid}: no cache found, skipping")
        continue

    dop = cv2.imread(dop_path)
    h_img, w_img = dop.shape[:2]

    cached = np.load(cache_path)
    lines_raw = cached['lines']
    scores_raw = cached['scores']

    # Same filter as test_validation_survey.py
    keep = scores_raw >= 0.3
    lines_filt = lines_raw[keep]
    scores_filt = scores_raw[keep]

    # Scale from FEAT_SIZE to ndom (100x100 for NRW 50m tile at 0.5m GSD)
    ndom_size = 100
    sr = ndom_size / FEAT_SIZE
    sc = ndom_size / FEAT_SIZE

    # ndom to image scale
    img_scale = h_img / ndom_size

    # --- Panel 1: All raw RMN lines (after score filter, before mask filter) ---
    panel_raw = dop.copy()
    for i, ln in enumerate(lines_filt):
        r1, c1 = ln[0, 0] * sr * img_scale, ln[0, 1] * sc * img_scale
        r2, c2 = ln[1, 0] * sr * img_scale, ln[1, 1] * sc * img_scale
        pt1 = (int(c1), int(r1))
        pt2 = (int(c2), int(r2))
        score = scores_filt[i]
        # Color by score: green=high, yellow=medium
        g = int(min(255, score * 400))
        b = int(max(0, (1 - score) * 200))
        color = (b, g, 200)
        cv2.line(panel_raw, pt1, pt2, color, 1)
        # Index label
        mx, my = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
        cv2.putText(panel_raw, str(i), (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv2.putText(panel_raw, f"{vid} RMN raw ({len(lines_filt)} lines, score>=0.3)",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # --- Panel 2: classified overlay (just load existing) ---
    overlay_path = os.path.join(out_dir, 'fusion_overlay.jpg')
    panel_cls = cv2.imread(overlay_path) if os.path.exists(overlay_path) else dop.copy()

    cv2.putText(panel_cls, f"{vid} V7 classified",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Combine side by side
    combined = np.hstack([panel_raw, panel_cls])
    out_path = os.path.join(out_dir, 'rmn_vs_classified.jpg')
    cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"{vid}: saved {out_path}")
    print(f"  Raw lines (score>=0.3): {len(lines_filt)}/{len(lines_raw)}")
