"""Zoom into S2 south gable end — show all perpendicular RMN fragments
with index, endpoints, and which slope they're on."""
import os, sys
import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'src'))

from wireframe_fusion import FEAT_SIZE

out_dir = os.path.join(BASE, 'results', 'validation', 'S2')
dop = cv2.imread(os.path.join(out_dir, 'dop.jpg'))
h_img, w_img = dop.shape[:2]

cached = np.load(os.path.join(out_dir, 'rmn_cache.npz'))
lines_raw = cached['lines']
scores_raw = cached['scores']

keep = scores_raw >= 0.3
lines = lines_raw[keep]
scores = scores_raw[keep]

ndom_size = 100
sr = ndom_size / FEAT_SIZE
sc = ndom_size / FEAT_SIZE
img_scale = h_img / ndom_size

# Ridge angle for S2 is ~88.6°
ridge_angle = 88.6
ridge_rad = np.radians(ridge_angle)
ridge_dir = np.array([np.sin(ridge_rad), np.cos(ridge_rad)])
perp_dir = np.array([-ridge_dir[1], ridge_dir[0]])

# Draw full image with ALL perpendicular lines highlighted
vis = dop.copy()

for i, ln in enumerate(lines):
    # Convert to ndom coords
    r1_nd, c1_nd = ln[0, 0] * sr, ln[0, 1] * sc
    r2_nd, c2_nd = ln[1, 0] * sr, ln[1, 1] * sc

    # Line angle
    dr = r2_nd - r1_nd
    dc = c2_nd - c1_nd
    angle = np.degrees(np.arctan2(dr, dc)) % 180
    angle_diff = min(abs(angle - ridge_angle % 180), 180 - abs(angle - ridge_angle % 180))

    is_perp = angle_diff > 70

    # Convert to image coords
    r1_img = r1_nd * img_scale
    c1_img = c1_nd * img_scale
    r2_img = r2_nd * img_scale
    c2_img = c2_nd * img_scale

    pt1 = (int(c1_img), int(r1_img))
    pt2 = (int(c2_img), int(r2_img))

    length_m = np.hypot(r2_nd - r1_nd, c2_nd - c1_nd) * 0.5

    if is_perp and length_m >= 1.0:
        # Color perpendicular lines by position (south=red, north=blue)
        center_along = ((r1_nd + r2_nd)/2 - 50) * ridge_dir[0] + ((c1_nd + c2_nd)/2 - 50) * ridge_dir[1]
        if center_along > 0:  # south end
            color = (0, 0, 255)  # red
        else:
            color = (255, 100, 0)  # blue
        cv2.line(vis, pt1, pt2, color, 2)
        # Label with index
        mx = (pt1[0] + pt2[0]) // 2
        my = (pt1[1] + pt2[1]) // 2
        cv2.putText(vis, f"{i}", (mx+3, my-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        # Endpoint markers
        cv2.circle(vis, pt1, 3, (0, 255, 0), -1)  # green = start
        cv2.circle(vis, pt2, 3, (0, 255, 255), -1)  # yellow = end
    else:
        cv2.line(vis, pt1, pt2, (80, 80, 80), 1)

# Add legend
cv2.putText(vis, "RED=south perp, BLUE=north perp", (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
cv2.putText(vis, "Green dot=pt1, Yellow dot=pt2", (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

out_path = os.path.join(out_dir, 'rmn_perp_debug.jpg')
cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"Saved: {out_path}")

# Now print detailed info for south-end perpendicular lines
print(f"\nSouth-end perpendicular fragments (pos_along > 0):")
print(f"{'idx':>4} {'len_m':>6} {'score':>5} {'ang_diff':>8} "
      f"{'r1_nd':>6} {'c1_nd':>6} {'r2_nd':>6} {'c2_nd':>6} "
      f"{'along':>7} {'perp':>7}")

origin = np.array([50.0, 50.0])  # approx mask center

for i, ln in enumerate(lines):
    r1_nd, c1_nd = ln[0, 0] * sr, ln[0, 1] * sc
    r2_nd, c2_nd = ln[1, 0] * sr, ln[1, 1] * sc

    dr = r2_nd - r1_nd
    dc = c2_nd - c1_nd
    angle = np.degrees(np.arctan2(dr, dc)) % 180
    angle_diff = min(abs(angle - ridge_angle % 180), 180 - abs(angle - ridge_angle % 180))

    if angle_diff <= 70:
        continue

    length_m = np.hypot(r2_nd - r1_nd, c2_nd - c1_nd) * 0.5
    if length_m < 1.0:
        continue

    center = np.array([(r1_nd + r2_nd)/2, (c1_nd + c2_nd)/2])
    pos_along = float(np.dot(center - origin, ridge_dir))
    pos_perp = float(np.dot(center - origin, perp_dir))

    if pos_along > 0:  # south end
        print(f"{i:4d} {length_m:6.1f} {scores[i]:5.2f} {angle_diff:8.1f} "
              f"{r1_nd:6.1f} {c1_nd:6.1f} {r2_nd:6.1f} {c2_nd:6.1f} "
              f"{pos_along:7.1f} {pos_perp:7.1f}")
