"""Validate automated roof measurement pipeline against Airteam drone survey reports.

Ground truth: DIN SPEC 5452-5 certified Airteam surveys (+-10cm accuracy).
Pipeline: nDOM + DOP -> detect_roof -> analyze_roof -> RoofMapNet -> wireframe fusion.

Usage:
    python test_validation_survey.py              # V1 only (default)
    python test_validation_survey.py V1 V2 V3 V7  # specific buildings
    python test_validation_survey.py ALL           # all available
"""

import os
import sys
import json
import subprocess
import traceback
import numpy as np
import cv2
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'src'))

from roof_analyzer import analyze_roof
from wireframe_fusion import (fuse_lines, generate_satteldach_wireframe,
                              generate_polygon_wireframe, auto_align_drone,
                              classify_rmn_lines, classify_rmn_lines_v10,
                              FEAT_SIZE)
from single_building_estimate import latlon_to_utm32

# Per-state GSD for nDOM
GSD_NDS = 1.0    # Niedersachsen DOM1/DGM1 at 1m
GSD_NRW = 0.5    # NRW nDOM at 0.5m

TILE_M = 50      # crop window size in metres


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# Precise coordinates of private validation sites (V1, S1-S3) are not stored in
# the repository. They are read from sites.local.json (gitignored), see
# sites.example.json for the format.
SITES_FILE = os.path.join(BASE, 'sites.local.json')


def resolve_coords(vid, bldg):
    """Fill in lat/lon from sites.local.json if the building has none."""
    if bldg.get('lat') is not None and bldg.get('lon') is not None:
        return bldg
    if not os.path.exists(SITES_FILE):
        sys.exit(f"ERROR: coordinates for {vid} are not in the repository.\n"
                 f"Create {SITES_FILE} (format: sites.example.json) with "
                 f"{{\"{vid}\": {{\"lat\": ..., \"lon\": ...}}}}.")
    with open(SITES_FILE) as f:
        sites = json.load(f)
    if vid not in sites:
        sys.exit(f"ERROR: {vid} missing in {SITES_FILE}.")
    return {**bldg, 'lat': float(sites[vid]['lat']), 'lon': float(sites[vid]['lon'])}


BUILDINGS = {
    'V1': dict(
        id='V1', address='[Site V1]', city='Niedersachsen', state='Niedersachsen',
        lat=None, lon=None, gsd=GSD_NDS,  # from sites.local.json
    ),
    'V2': dict(
        id='V2', address='[Site V2]', city='Niedersachsen', state='Niedersachsen',
        lat=52.396, lon=9.597, gsd=GSD_NDS,
    ),
    'V3': dict(
        id='V3', address='[Site V3]', city='Niedersachsen',
        state='Niedersachsen', lat=53.67, lon=8.55, gsd=GSD_NDS,
    ),
    'V7': dict(
        id='V7', address='[Site V7]', city='NRW',
        state='Nordrhein-Westfalen', lat=51.987, lon=8.5, gsd=GSD_NRW,
    ),
    'S1': dict(
        id='S1', address='[Site S1]', city='NRW',
        state='Nordrhein-Westfalen', lat=None, lon=None, gsd=GSD_NRW,
        merge_alkis=3.0,
    ),
    'S2': dict(
        id='S2', address='[Site S2]', city='NRW',
        state='Nordrhein-Westfalen', lat=None, lon=None, gsd=GSD_NRW,
        merge_alkis=3.0, max_units=2,
    ),
    'S3': dict(
        id='S3', address='[Site S3]', city='NRW',
        state='Nordrhein-Westfalen', lat=None, lon=None, gsd=GSD_NRW,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — NIEDERSACHSEN (COG tiles with TIFF georeference tags)
# ══════════════════════════════════════════════════════════════════════════════

def _read_tif_georeference(tif_path):
    """Read georeference from TIFF tags 33922 (tiepoint) and 33550 (pixel scale).

    Returns (tiepoint_e, tiepoint_n, scale_x, scale_y).
    tiepoint_e/n = UTM easting/northing of top-left pixel.
    scale_x/y = pixel size in metres.
    """
    from PIL import Image
    img = Image.open(tif_path)
    scale = img.tag_v2.get(33550)     # (scaleX, scaleY, scaleZ)
    tiepoint = img.tag_v2.get(33922)  # (i, j, k, x, y, z)
    img.close()
    if scale is None or tiepoint is None:
        raise ValueError(f"Missing georeference tags in {tif_path}")
    scale_x = float(scale[0])
    scale_y = float(scale[1])
    tie_e = float(tiepoint[3])
    tie_n = float(tiepoint[4])
    return tie_e, tie_n, scale_x, scale_y


def load_niedersachsen_ndom(vid, easting, northing, half_m=25):
    """Load nDOM = DOM_COG - DGM_COG, crop 50x50m window at 1m GSD."""
    import rasterio

    data_dir = os.path.join(BASE, 'data', 'niedersachsen')
    dom_path = os.path.join(data_dir, f'{vid.lower()}_dom_cog.tif')
    dgm_path = os.path.join(data_dir, f'{vid.lower()}_dgm_cog.tif')

    if not os.path.exists(dom_path) or not os.path.exists(dgm_path):
        raise FileNotFoundError(f"Missing DOM/DGM COG for {vid} in {data_dir}")

    with rasterio.open(dom_path) as ds:
        dom_full = ds.read(1).astype(np.float32)
        bounds = ds.bounds  # (left, bottom, right, top) in UTM
        res_x = ds.res[0]   # pixel size X
        res_y = ds.res[1]   # pixel size Y
        tie_e = bounds.left
        tie_n = bounds.top

    with rasterio.open(dgm_path) as ds:
        dgm_full = ds.read(1).astype(np.float32)

    # Compute pixel coordinates of the building center in the COG
    col_center = (easting - tie_e) / res_x
    row_center = (tie_n - northing) / res_y  # northing decreases downward

    # Crop window (pixel coords at 1m GSD)
    half_px = int(half_m)
    row_start = int(round(row_center - half_px))
    row_end = int(round(row_center + half_px))
    col_start = int(round(col_center - half_px))
    col_end = int(round(col_center + half_px))

    # Clamp to array bounds
    h, w = dom_full.shape
    row_start = max(0, row_start)
    row_end = min(h, row_end)
    col_start = max(0, col_start)
    col_end = min(w, col_end)

    ndom_crop = dom_full[row_start:row_end, col_start:col_end] - \
                dgm_full[row_start:row_end, col_start:col_end]

    # UTM coordinates of the crop origin
    e_min = tie_e + col_start * res_x
    n_max = tie_n - row_start * res_y
    n_min = n_max - ndom_crop.shape[0] * res_y

    return ndom_crop, e_min, n_min


def load_niedersachsen_dop(vid, easting, northing, half_m=25):
    """Load DOP from 600x600 WMS TIF, crop to 50m x 50m (~250x250 at 20cm GSD).

    Returns (dop_arr, rgb_crop) where dop_arr is (3, H, W) and rgb_crop is (H, W, 3).
    """
    data_dir = os.path.join(BASE, 'data', 'niedersachsen')
    dop_path = os.path.join(data_dir, f'{vid.lower()}_dop.tif')

    if not os.path.exists(dop_path):
        raise FileNotFoundError(f"Missing DOP for {vid} at {dop_path}")

    from PIL import Image
    img = Image.open(dop_path)
    rgb = np.array(img)  # (H, W, 3) uint8

    # Read georeference from TIFF tags
    tie_e, tie_n, scale_x, scale_y = _read_tif_georeference(dop_path)
    img.close()

    # Pixel coordinates of building center in the DOP
    col_center = (easting - tie_e) / scale_x
    row_center = (tie_n - northing) / scale_y

    # Crop 50m x 50m -> ~250 x 250 pixels at 20cm GSD
    half_px = int(half_m / scale_x)
    row_start = int(round(row_center - half_px))
    row_end = int(round(row_center + half_px))
    col_start = int(round(col_center - half_px))
    col_end = int(round(col_center + half_px))

    h, w = rgb.shape[:2]
    row_start = max(0, row_start)
    row_end = min(h, row_end)
    col_start = max(0, col_start)
    col_end = min(w, col_end)

    rgb_crop = rgb[row_start:row_end, col_start:col_end]

    # Resize to exact 5× nDOM resolution to avoid shape mismatches
    # nDOM at 1m covers half_m*2 = 50 pixels → DOP should be 250×250
    target_px = int(half_m * 2 / scale_x)
    if rgb_crop.shape[0] != target_px or rgb_crop.shape[1] != target_px:
        rgb_crop = cv2.resize(rgb_crop, (target_px, target_px),
                              interpolation=cv2.INTER_AREA)

    # Convert to (bands, H, W) format expected by the pipeline
    dop_arr = np.transpose(rgb_crop, (2, 0, 1))  # (3, H, W)

    return dop_arr, rgb_crop


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — NRW (WCS download)
# ══════════════════════════════════════════════════════════════════════════════

def load_nrw_data(easting, northing, half_m=25):
    """Download NRW nDOM + DOP via WCS. Returns (ndom_arr, dop_arr, e_min, n_min)."""
    from single_building_estimate import download_chip
    from nrw_downloader import DOP_URL, NDOM_URL

    ndom_arr = download_chip(NDOM_URL, "nw_ndom", easting, northing, half_m)[0]
    dop_arr = download_chip(DOP_URL, "nw_dop", easting, northing, half_m)
    e_min = easting - half_m
    n_min = northing - half_m

    return ndom_arr, dop_arr, e_min, n_min


# ══════════════════════════════════════════════════════════════════════════════
# NIEDERSACHSEN ALKIS WFS
# ══════════════════════════════════════════════════════════════════════════════

NDS_ALKIS_WFS = 'https://opendata.lgln.niedersachsen.de/doorman/noauth/alkis_wfs_einfach'

def get_nds_alkis_mask(ndom_arr, easting, northing, e_min, n_min, gsd=1.0):
    """Query Niedersachsen ALKIS WFS and return mask + rect + polygon for closest building.

    Returns (alkis_mask, alkis_rect, alkis_poly) where alkis_poly is a list of
    (row, col) vertices in nDOM pixel coordinates (simplified, short edges removed).
    """
    import urllib.request
    import xml.etree.ElementTree as ET

    half = 30
    bbox = f'{easting-half},{northing-half},{easting+half},{northing+half}'
    url = (f'{NDS_ALKIS_WFS}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature'
           f'&TYPENAMES=ave:GebaeudeBauwerk&SRSNAME=EPSG:25832'
           f'&BBOX={bbox},EPSG:25832&COUNT=20')

    try:
        resp = urllib.request.urlopen(url, timeout=15).read().decode('utf-8')
    except Exception as ex:
        print(f"  ALKIS WFS query failed: {ex}")
        return None, None, None

    root = ET.fromstring(resp)
    gml_ns = '{http://www.opengis.net/gml/3.2}'

    buildings = []
    for pl in root.iter(f'{gml_ns}posList'):
        coords = [float(x) for x in pl.text.strip().split()]
        pts = np.array([(coords[i], coords[i+1]) for i in range(0, len(coords), 2)])
        centroid_e, centroid_n = pts[:, 0].mean(), pts[:, 1].mean()
        dist = np.hypot(centroid_e - easting, centroid_n - northing)
        area = 0.5 * abs(sum(pts[i, 0] * pts[(i+1) % len(pts), 1] -
                              pts[(i+1) % len(pts), 0] * pts[i, 1]
                              for i in range(len(pts))))
        if area > 30:
            buildings.append({'pts': pts, 'dist': dist, 'area': area})

    if not buildings:
        print(f"  No ALKIS buildings found")
        return None, None, None

    best = min(buildings, key=lambda b: b['dist'])
    print(f"  ALKIS: {len(buildings)} buildings, best={best['area']:.0f}m² at {best['dist']:.1f}m")

    h, w = ndom_arr.shape
    n_max = n_min + h * gsd
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_px = []
    pts_rc = []
    for pt_e, pt_n in best['pts']:
        col = (pt_e - e_min) / gsd
        row = (n_max - pt_n) / gsd
        pts_px.append([int(round(col)), int(round(row))])
        pts_rc.append([row, col])
    pts_cv = np.array(pts_px, dtype=np.int32)
    cv2.fillPoly(mask, [pts_cv], 1)
    alkis_mask = mask.astype(bool)

    # minAreaRect for wireframe
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alkis_rect = None
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        alkis_rect = [(float(pt[1]), float(pt[0])) for pt in box]

    # Simplify polygon: remove short edges < 1m
    pts_rc = np.array(pts_rc)
    simplified = [pts_rc[0]]
    for i in range(1, len(pts_rc)):
        d = np.hypot(pts_rc[i][0] - simplified[-1][0],
                     pts_rc[i][1] - simplified[-1][1]) * gsd
        if d >= 1.0:
            simplified.append(pts_rc[i])
    if (len(simplified) > 1 and
        np.hypot(simplified[-1][0] - simplified[0][0],
                 simplified[-1][1] - simplified[0][1]) * gsd < 1.0):
        simplified.pop()
    alkis_poly = np.array(simplified)
    print(f"  ALKIS polygon: {len(pts_rc)} -> {len(alkis_poly)} vertices (simplified)")

    return alkis_mask, alkis_rect, alkis_poly


# ══════════════════════════════════════════════════════════════════════════════
# MASK DETECTION (nDOM-based fallback)
# ══════════════════════════════════════════════════════════════════════════════

def detect_roof_ndom_only(ndom_arr, height_thresh=2.0, gsd=1.0):
    """nDOM-based roof detection: threshold + best component by height/distance/area.

    For Niedersachsen where NRW ALKIS WFS is not available.
    Scoring balances height, proximity to center, and area to avoid
    selecting small tall structures (chimneys) over the main building.
    """
    from scipy.ndimage import label as ndi_label, binary_closing, binary_fill_holes

    mask = ndom_arr > height_thresh
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)

    labeled, n = ndi_label(mask)
    if n == 0:
        return np.zeros_like(ndom_arr, dtype=bool), False

    # Pick the best component: balance height, distance from center, and area
    h, w = ndom_arr.shape
    cy, cx = h / 2, w / 2
    best_mask = None
    best_score = -np.inf
    min_area_m2 = 30.0  # minimum footprint area for a building

    for i in range(1, n + 1):
        comp = labeled == i
        area_px = comp.sum()
        area_m2 = area_px * gsd * gsd
        if area_m2 < min_area_m2:
            continue
        rows, cols = np.where(comp)
        cr, cc = rows.mean(), cols.mean()
        dist = np.hypot(cr - cy, cc - cx)
        mean_h = float(ndom_arr[comp].mean())
        # Score: height matters, but also reward area (log scale)
        # and penalize distance from center
        area_bonus = np.log(max(area_m2, 1.0)) * 1.5
        score = mean_h * 2.0 + area_bonus - dist * 0.5
        if score > best_score:
            best_score = score
            best_mask = comp

    if best_mask is None:
        return np.zeros_like(ndom_arr, dtype=bool), False

    return best_mask.astype(bool), True


def get_mask_rect(mask):
    """Compute minAreaRect from mask, return 4 corners as [(row, col), ...]."""
    u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return [(float(pt[1]), float(pt[0])) for pt in box]


# ══════════════════════════════════════════════════════════════════════════════
# ROOFMAPNET SUBPROCESS (copied from test_fusion_nrw.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_roofmapnet_subprocess(dop_path):
    """Run RoofMapNet in sam_env via subprocess, return lines + scores."""
    script = f"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join('{BASE}', 'RoofMapNet'))
import torch, cv2
from roofmapnet.config import C, M
from roofmapnet.models.regression import MultitaskHead
from roofmapnet.models.roofmapnet import RoofMapNet

IMG_SIZE = 512

C.update(C.from_yaml(filename=os.path.join('{BASE}', 'RoofMapNet', 'config', 'config.yaml')))
M.update(C.model)

model = RoofMapNet(
    depth=M.depth,
    head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
    num_stacks=M.num_stacks, num_blocks=M.num_blocks,
    num_classes=sum(sum(M.head_size, [])),
)
ckpt = os.path.join('{BASE}', 'RoofMapNet', 'pretrained_models', 'roofmapnet.pth')
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model.eval()

bgr = cv2.imread('{dop_path}')
rgb = bgr[:, :, ::-1].copy()
img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float64)
mean = np.array(M.image.mean)
std = np.array(M.image.stddev)
img = (img - mean) / std
img = np.rollaxis(img, 2).copy()
img_t = torch.from_numpy(img).float().unsqueeze(0)

feat_size = IMG_SIZE // 4
junc = torch.tensor([[-1000.0, -1000.0], [-1001.0, -1001.0]])
meta = dict(
    junc=junc, jtyp=torch.zeros(2, dtype=torch.uint8),
    Lpos=torch.zeros(3, 3, dtype=torch.uint8),
    Lneg=torch.zeros(3, 3, dtype=torch.uint8),
)
target = dict(
    jmap=torch.zeros(1, 1, feat_size, feat_size),
    joff=torch.zeros(1, 1, 2, feat_size, feat_size),
    lmap=torch.zeros(1, feat_size, feat_size),
)

with torch.no_grad():
    result = model(dict(image=img_t, meta=[meta], target=target, mode='validation'))

lines = result['preds']['lines'][0].cpu().numpy()
scores = result['preds']['score'][0].cpu().numpy()
mask = scores > 0.15
lines, scores = lines[mask], scores[mask]

seen = set()
ul, us = [], []
for l, s in zip(lines, scores):
    key = (round(l[0,0],1), round(l[0,1],1), round(l[1,0],1), round(l[1,1],1))
    if key not in seen:
        seen.add(key)
        ul.append(l.tolist())
        us.append(float(s))

print(json.dumps(dict(lines=ul, scores=us)))
"""
    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'sam_env', 'python', '-c', script],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("  RoofMapNet subprocess timed out")
        return np.zeros((0, 2, 2)), np.zeros(0)
    except Exception as ex:
        print(f"  RoofMapNet subprocess failed: {ex}")
        return np.zeros((0, 2, 2)), np.zeros(0)

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
        print(f"  RoofMapNet subprocess error: {stderr_tail}")
        return np.zeros((0, 2, 2)), np.zeros(0)

    for line in result.stdout.strip().split('\n'):
        if line.startswith('{'):
            data = json.loads(line)
            lines = np.array(data['lines']) if data['lines'] else np.zeros((0, 2, 2))
            scores = np.array(data['scores']) if data['scores'] else np.zeros(0)
            return lines, scores

    return np.zeros((0, 2, 2)), np.zeros(0)


def filter_rmn_lines(lines, scores, mask, gsd):
    """Filter RoofMapNet lines: score + mask coverage + length + light dedup."""
    if len(lines) == 0:
        return lines, scores

    keep = scores >= 0.3
    lines, scores = lines[keep], scores[keep]
    if len(lines) == 0:
        return lines, scores

    h, w = mask.shape
    sr, sc = h / FEAT_SIZE, w / FEAT_SIZE

    mask_buf = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    keep = []
    for i, ln in enumerate(lines):
        r1 = int(np.clip(round(ln[0, 0] * sr), 0, h - 1))
        c1 = int(np.clip(round(ln[0, 1] * sc), 0, w - 1))
        r2 = int(np.clip(round(ln[1, 0] * sr), 0, h - 1))
        c2 = int(np.clip(round(ln[1, 1] * sc), 0, w - 1))
        if mask_buf[r1, c1] and mask_buf[r2, c2]:
            keep.append(i)
    if not keep:
        return np.zeros((0, 2, 2)), np.zeros(0)
    lines, scores = lines[np.array(keep)], scores[np.array(keep)]

    lengths = np.hypot(
        (lines[:, 1, 0] - lines[:, 0, 0]) * sr * gsd,
        (lines[:, 1, 1] - lines[:, 0, 1]) * sc * gsd,
    )
    keep = lengths >= 1.5
    lines, scores = lines[keep], scores[keep]
    if len(lines) == 0:
        return lines, scores

    order = np.argsort(-scores)
    lines, scores = lines[order], scores[order]
    keep = np.ones(len(lines), dtype=bool)
    for i in range(len(lines)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(lines)):
            if not keep[j]:
                continue
            mid_dist = np.linalg.norm(
                (lines[i, 0] + lines[i, 1]) / 2 - (lines[j, 0] + lines[j, 1]) / 2
            )
            if mid_dist < 4.0:
                ai = np.degrees(np.arctan2(
                    lines[i, 1, 0] - lines[i, 0, 0],
                    lines[i, 1, 1] - lines[i, 0, 1])) % 180
                aj = np.degrees(np.arctan2(
                    lines[j, 1, 0] - lines[j, 0, 0],
                    lines[j, 1, 1] - lines[j, 0, 1])) % 180
                ad = min(abs(ai - aj), 180 - abs(ai - aj))
                if ad < 15:
                    keep[j] = False
    return lines[keep], scores[keep]


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION (from test_fusion_nrw.py)
# ══════════════════════════════════════════════════════════════════════════════

def visualize_fusion(img_rgb, fusion_result, out_path, ndom_shape, title='',
                     ndom_to_img=None):
    """Draw classified wireframe lines colored by type.

    ndom_to_img: optional (scale, offset_r, offset_c) tuple for non-aligned
                 images (e.g. drone orthophoto). Maps ndom pixel (r,c) to
                 image pixel (r*scale + offset_r, c*scale + offset_c).
    """
    h, w = img_rgb.shape[:2]
    h_nd, w_nd = ndom_shape

    if ndom_to_img is not None:
        nd_scale, off_r, off_c = ndom_to_img
        scale_r = nd_scale
        scale_c = nd_scale
    else:
        scale_r = h / h_nd
        scale_c = w / w_nd
        off_r = 0.0
        off_c = 0.0

    # Scale line thickness and font for high-res images
    ui_scale = max(1, int(round(min(h, w) / 300)))

    vis = img_rgb[:, :, ::-1].copy()

    colors = {
        'ridge': (0, 0, 255),
        'traufe': (255, 255, 0),
        'ortgang': (0, 255, 0),
        'kehle': (255, 0, 255),
        'grat': (0, 165, 255),
    }

    for cl in fusion_result.get('classified_lines', []):
        r1, c1, r2, c2 = cl['segment']
        pt1 = (int(c1 * scale_c + off_c), int(r1 * scale_r + off_r))
        pt2 = (int(c2 * scale_c + off_c), int(r2 * scale_r + off_r))
        color = colors.get(cl['type'], (200, 200, 200))
        thickness = (3 if cl['source'] == 'roofmapnet' else 2) * ui_scale
        cv2.line(vis, pt1, pt2, color, thickness)

    legend_y = 30 * ui_scale
    font_scale = 0.5 * ui_scale
    for ltype, color in colors.items():
        cv2.putText(vis, ltype, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 2 * ui_scale)
        legend_y += 22 * ui_scale

    if title:
        cv2.putText(vis, title, (10, h - 15 * ui_scale),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * ui_scale, (255, 255, 255), 2 * ui_scale)

    cv2.imwrite(out_path, vis)


def _ndom_to_img_scales(img_shape, ndom_shape, ndom_to_img):
    h, w = img_shape[:2]
    h_nd, w_nd = ndom_shape
    if ndom_to_img is not None:
        nd_scale, off_r, off_c = ndom_to_img
        return nd_scale, nd_scale, off_r, off_c
    return h / h_nd, w / w_nd, 0.0, 0.0


def visualize_masse(img_rgb, fusion_result, out_path, ndom_shape,
                    ndom_to_img=None):
    """Overlay: V10 structural lines with length labels."""
    scale_r, scale_c, off_r, off_c = _ndom_to_img_scales(
        img_rgb.shape, ndom_shape, ndom_to_img)
    h, w = img_rgb.shape[:2]
    ui = max(1, int(round(min(h, w) / 300)))
    vis = img_rgb[:, :, ::-1].copy()

    type_colors = {
        'ridge': (0, 0, 255),
        'traufe': (255, 255, 0),
        'ortgang': (0, 255, 0),
        'kehle': (255, 0, 255),
        'grat': (0, 165, 255),
    }
    type_labels = {
        'ridge': 'First', 'traufe': 'Traufe', 'ortgang': 'Ortgang',
        'kehle': 'Kehle', 'grat': 'Grat',
    }

    for cl in fusion_result.get('classified_lines', []):
        if cl.get('status') != 'accepted':
            continue
        r1, c1, r2, c2 = cl['segment']
        pt1 = (int(c1 * scale_c + off_c), int(r1 * scale_r + off_r))
        pt2 = (int(c2 * scale_c + off_c), int(r2 * scale_r + off_r))
        color = type_colors.get(cl['type'], (200, 200, 200))
        cv2.line(vis, pt1, pt2, color, max(2, ui + 1))
        mx, my = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
        label = f"{type_labels.get(cl['type'], cl['type'])} {cl['length_m']:.1f}m"
        cv2.putText(vis, label, (mx + 4, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * max(1, ui),
                    (255, 255, 255), max(1, ui))

    y0 = 15
    for i, (lbl, col) in enumerate([
        ("First", (0, 0, 255)), ("Traufe", (255, 255, 0)),
        ("Ortgang", (0, 255, 0)),
    ]):
        cv2.putText(vis, lbl, (5, y0 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * max(1, ui), col, 1)

    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Masse overlay saved: {out_path}")


def visualize_aufbauten(img_rgb, superstructures, skylights, out_path,
                        ndom_shape, ndom_to_img=None):
    """Overlay: PV, chimney, Dachfenster detections."""
    scale_r, scale_c, off_r, off_c = _ndom_to_img_scales(
        img_rgb.shape, ndom_shape, ndom_to_img)
    h, w = img_rgb.shape[:2]
    ui = max(1, int(round(min(h, w) / 300)))
    vis = img_rgb[:, :, ::-1].copy()

    cls_colors = {
        'PV Module': (0, 200, 200),
        'Dormer': (255, 128, 0),
        'Dachfenster': (200, 0, 200),
        'Chimney/Other': (0, 0, 200),
    }
    cls_labels = {
        'PV Module': 'PV', 'Dormer': 'Gaube',
        'Dachfenster': 'DF', 'Chimney/Other': 'Schorn.',
    }

    if superstructures is not None:
        for cls_name, cls_mask in superstructures.get('class_masks', {}).items():
            cnt = superstructures['counts'].get(cls_name, 0)
            if cnt == 0:
                continue
            color = cls_colors.get(cls_name, (128, 128, 128))
            contours, _ = cv2.findContours(
                cls_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            overlay = vis.copy()
            cv2.drawContours(overlay, contours, -1, color, -1)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            cv2.drawContours(vis, contours, -1, color, max(1, ui))
            short = cls_labels.get(cls_name, cls_name[:3])
            for cnt_i in contours:
                M = cv2.moments(cnt_i)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(vis, short, (cx - 8, cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3 * max(1, ui),
                                color, 1)

    y0 = 15
    legend_items = [
        ("PV Module", (0, 200, 200)), ("Schornstein", (0, 0, 200)),
        ("Dachfenster", (200, 0, 200)),
    ]
    for i, (lbl, col) in enumerate(legend_items):
        cv2.putText(vis, lbl, (5, y0 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * max(1, ui), col, 1)

    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Aufbauten overlay saved: {out_path}")


def visualize_rmn_classification(img_rgb, fusion_result, out_path, ndom_shape,
                                 vid='', ndom_to_img=None):
    """Draw ALL RoofMapNet lines color-coded by classification status.

    Accepted lines: thick, bright color by type
    Rejected lines: thin, gray, with reason label
    """
    h, w = img_rgb.shape[:2]
    h_nd, w_nd = ndom_shape

    if ndom_to_img is not None:
        nd_scale, off_r, off_c = ndom_to_img
        scale_r = nd_scale
        scale_c = nd_scale
    else:
        scale_r = h / h_nd
        scale_c = w / w_nd
        off_r = 0.0
        off_c = 0.0

    ui = max(1, int(round(min(h, w) / 300)))
    vis = img_rgb[:, :, ::-1].copy()

    type_colors = {
        'ridge': (0, 0, 255),
        'traufe': (255, 255, 0),
        'ortgang': (0, 255, 0),
        'kehle': (255, 0, 255),
        'grat': (0, 165, 255),
    }
    rejected_color = (120, 120, 120)

    all_lines = fusion_result.get('classified_lines', [])

    # Draw accepted lines only, thin, no labels
    for cl in all_lines:
        if cl.get('status') != 'accepted':
            continue
        r1, c1, r2, c2 = cl['segment']
        pt1 = (int(c1 * scale_c + off_c), int(r1 * scale_r + off_r))
        pt2 = (int(c2 * scale_c + off_c), int(r2 * scale_r + off_r))
        color = type_colors.get(cl['type'], (200, 200, 200))
        cv2.line(vis, pt1, pt2, color, max(1, ui))

    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ══════════════════════════════════════════════════════════════════════════════
# ERROR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def pct_error(pred, gt_val):
    """Percentage error; returns nan if gt is zero or nan."""
    try:
        if gt_val == 0 or (isinstance(gt_val, float) and np.isnan(gt_val)):
            return float('nan')
    except (TypeError, ValueError):
        return float('nan')
    return (pred - gt_val) / gt_val * 100


def abs_pct_error(pred, gt_val):
    """Absolute percentage error; returns nan if gt is zero."""
    return abs(pct_error(pred, gt_val))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — PROCESS ONE BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def process_building(vid, bldg, gt_row):
    """Run the full pipeline for one building and return result dict."""
    bldg = resolve_coords(vid, bldg)
    lat, lon = bldg['lat'], bldg['lon']
    gsd = bldg['gsd']
    state = bldg['state']
    e, n = latlon_to_utm32(lat, lon)
    half = TILE_M / 2

    print(f"\n{'='*70}")
    print(f"  {vid}: {bldg['address']}, {bldg['city']} ({state})")
    print(f"  UTM32: E={e:.1f}  N={n:.1f}  GSD={gsd}m")
    print(f"{'='*70}")

    result = {'id': vid, 'address': bldg['address'], 'city': bldg['city']}

    out_dir = os.path.join(BASE, 'results', 'validation', vid)
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load data (use cache if available) ──────────────────────────
    cache_ndom = os.path.join(out_dir, 'ndom.npy')
    cache_mask = os.path.join(out_dir, 'mask.npy')
    cache_dop = os.path.join(out_dir, 'dop.jpg')
    use_cache = (os.path.exists(cache_ndom) and os.path.exists(cache_mask)
                 and os.path.exists(cache_dop))

    try:
        if use_cache:
            print(f"  Using cached data from {out_dir}")
            ndom_arr = np.load(cache_ndom)
            mask = np.load(cache_mask)
            dop_bgr = cv2.imread(cache_dop)
            rgb_crop = dop_bgr[:, :, ::-1]
            dop_arr = np.transpose(rgb_crop, (2, 0, 1))
            e_min = e - half
            n_min = n - half
            print(f"  nDOM: {ndom_arr.shape} (range {ndom_arr.min():.1f}..{ndom_arr.max():.1f}m)")
            print(f"  DOP:  {dop_arr.shape}")
            print(f"  Mask: cached, {mask.sum()} px")
        elif state == 'Niedersachsen':
            print(f"  Loading Niedersachsen COG data...")
            ndom_arr, e_min, n_min = load_niedersachsen_ndom(vid, e, n, half)
            dop_arr, rgb_crop = load_niedersachsen_dop(vid, e, n, half)
            print(f"  nDOM: {ndom_arr.shape} (range {ndom_arr.min():.1f}..{ndom_arr.max():.1f}m)")
            print(f"  DOP:  {dop_arr.shape}")
        elif state == 'Nordrhein-Westfalen':
            print(f"  Downloading NRW WCS data...")
            ndom_arr, dop_arr, e_min, n_min = load_nrw_data(e, n, half)
            rgb_crop = np.transpose(dop_arr[:3], (1, 2, 0))  # (H, W, 3)
            print(f"  nDOM: {ndom_arr.shape} (range {ndom_arr.min():.1f}..{ndom_arr.max():.1f}m)")
            print(f"  DOP:  {dop_arr.shape}")
        else:
            print(f"  ERROR: unsupported state {state}")
            result['error'] = f"unsupported state {state}"
            return result
    except Exception as ex:
        print(f"  ERROR loading data: {ex}")
        traceback.print_exc()
        result['error'] = str(ex)
        return result

    # ── Step 2: Detect roof mask ─────────────────────────────────────────────
    merge_gap = bldg.get('merge_alkis', 0)
    merge = merge_gap > 0
    mu = bldg.get('max_units', None)
    if use_cache:
        found = mask.sum() > 0
        method = 'cached'
        alkis_mask = mask
        alkis_rect = get_mask_rect(mask) if found else None
        alkis_poly = None
    else:
        try:
            alkis_poly = None
            if state == 'Nordrhein-Westfalen':
                from building_detector import detect_roof, get_alkis_mask, get_alkis_rect
                mask, found, method = detect_roof(ndom_arr, dop_arr, e_min, n_min,
                                                  merge_alkis=merge,
                                                  merge_gap_m=merge_gap,
                                                  max_units=mu)
                if found:
                    alkis_mask = get_alkis_mask(ndom_arr, e_min, n_min,
                                               merge_alkis=merge,
                                               merge_gap_m=merge_gap,
                                               max_units=mu)
                    alkis_rect = get_alkis_rect(ndom_arr, e_min, n_min,
                                               merge_alkis=merge,
                                               merge_gap_m=merge_gap,
                                               max_units=mu)
                    if alkis_mask is None:
                        alkis_mask = mask
                    if alkis_rect is None:
                        alkis_rect = get_mask_rect(mask)
                else:
                    alkis_mask = mask
                    alkis_rect = None
                print(f"  Mask: {method}, {mask.sum()} px")
            else:
                alkis_mask, alkis_rect, alkis_poly = get_nds_alkis_mask(
                    ndom_arr, e, n, e_min, n_min, gsd=gsd)
                if alkis_mask is not None and alkis_mask.sum() > 0:
                    mask = alkis_mask & (ndom_arr > 1.5)
                    if mask.sum() < 10:
                        mask = alkis_mask
                    found = True
                    method = 'ALKIS + nDOM'
                else:
                    mask, found = detect_roof_ndom_only(ndom_arr, gsd=gsd)
                    method = 'nDOM threshold'
                    alkis_mask = mask
                    alkis_rect = get_mask_rect(mask) if found else None
                    alkis_poly = None
                print(f"  Mask: {method}, {mask.sum()} px, found={found}")
        except Exception as ex:
            print(f"  ERROR in roof detection: {ex}")
            traceback.print_exc()
            result['error'] = str(ex)
            return result

    if not found or mask.sum() == 0:
        print(f"  ERROR: no roof detected")
        result['error'] = 'no roof detected'
        return result

    # ── Step 3: Save DOP + run RoofMapNet BEFORE geometry analysis ─────────
    if not use_cache:
        np.save(os.path.join(out_dir, 'ndom.npy'), ndom_arr)
        np.save(os.path.join(out_dir, 'mask.npy'), mask)

    # Use drone orthophoto if available, otherwise save state DOP
    drone_ortho = os.path.join(out_dir, 'drone_ortho.jpg')
    dop_jpg_path = os.path.join(out_dir, 'dop.jpg')
    ndom_to_img = None
    if os.path.exists(drone_ortho):
        dop_jpg_path = drone_ortho
        drone_bgr = cv2.imread(drone_ortho)
        rgb_crop = drone_bgr[:, :, ::-1]  # BGR→RGB
        print(f"  Using drone orthophoto: {drone_ortho} ({rgb_crop.shape})")
        if alkis_poly is not None and len(alkis_poly) >= 4:
            align = auto_align_drone(alkis_poly, drone_bgr, gsd=gsd)
            if align is not None:
                ndom_to_img = align
                print(f"  nDOM→drone (auto): scale={align[0]:.1f} "
                      f"offset=({align[1]:.0f},{align[2]:.0f})")
            else:
                print(f"  WARNING: auto-alignment failed, using center estimate")
        if ndom_to_img is None:
            img_h, img_w = rgb_crop.shape[:2]
            mask_rows = np.where(mask)[0]
            mask_cols = np.where(mask)[1]
            if len(mask_rows) > 0:
                bldg_center_r = (mask_rows.min() + mask_rows.max()) / 2
                bldg_center_c = (mask_cols.min() + mask_cols.max()) / 2
                bldg_span = max(mask_rows.max()-mask_rows.min(),
                                mask_cols.max()-mask_cols.min()) + 1
                nd_scale = (min(img_h, img_w) * 0.85) / bldg_span
                dc_r, dc_c = img_h // 2, img_w // 2
                off_r = dc_r - bldg_center_r * nd_scale
                off_c = dc_c - bldg_center_c * nd_scale
                ndom_to_img = (nd_scale, off_r, off_c)
    else:
        bgr = rgb_crop[:, :, ::-1]
        cv2.imwrite(dop_jpg_path, bgr)
        print(f"  DOP saved: {dop_jpg_path}")

    rmn_cache_path = os.path.join(out_dir, 'rmn_cache.npz')
    if os.path.exists(rmn_cache_path):
        cached = np.load(rmn_cache_path)
        rmn_lines_raw = cached['lines']
        rmn_scores_raw = cached['scores']
        print(f"  RoofMapNet (cached): {len(rmn_lines_raw)} raw lines")
    else:
        print(f"  Running RoofMapNet...")
        rmn_lines_raw, rmn_scores_raw = run_roofmapnet_subprocess(dop_jpg_path)
        print(f"  RoofMapNet: {len(rmn_lines_raw)} raw lines")
        np.savez(rmn_cache_path, lines=rmn_lines_raw, scores=rmn_scores_raw)
        print(f"  RoofMapNet cached → {rmn_cache_path}")

    # Light pre-filter: score ≥ 0.3 + both endpoints near mask.
    # No NMS — the Satteldach classifier picks one best line per role.
    rmn_lines, rmn_scores = rmn_lines_raw, rmn_scores_raw
    if len(rmn_lines) > 0:
        keep = rmn_scores >= 0.3
        rmn_lines, rmn_scores = rmn_lines[keep], rmn_scores[keep]

        if len(rmn_lines) > 0:
            h_nd, w_nd = ndom_arr.shape
            sr, sc = h_nd / FEAT_SIZE, w_nd / FEAT_SIZE
            mask_buf = cv2.dilate(alkis_mask.astype(np.uint8),
                                  np.ones((9, 9), np.uint8)).astype(bool)
            keep2 = []
            for i, ln in enumerate(rmn_lines):
                r1 = int(np.clip(round(ln[0, 0] * sr), 0, h_nd - 1))
                c1 = int(np.clip(round(ln[0, 1] * sc), 0, w_nd - 1))
                r2 = int(np.clip(round(ln[1, 0] * sr), 0, h_nd - 1))
                c2 = int(np.clip(round(ln[1, 1] * sc), 0, w_nd - 1))
                rm = int(np.clip(round((ln[0, 0] + ln[1, 0]) / 2 * sr), 0, h_nd - 1))
                cm = int(np.clip(round((ln[0, 1] + ln[1, 1]) / 2 * sc), 0, w_nd - 1))
                if mask_buf[r1, c1] or mask_buf[r2, c2] or mask_buf[rm, cm]:
                    keep2.append(i)
            if keep2:
                rmn_lines = rmn_lines[np.array(keep2)]
                rmn_scores = rmn_scores[np.array(keep2)]
            else:
                rmn_lines = np.zeros((0, 2, 2))
                rmn_scores = np.zeros(0)

        print(f"  After score+mask filter: {len(rmn_lines)} lines")

    # ── Step 4: Roof geometry analysis (ridge-guided if RMN available) ──────
    try:
        geo = analyze_roof(mask, ndom_arr, dop_arr, gsd=gsd,
                           alkis_rect=alkis_rect,
                           rmn_lines=rmn_lines, rmn_scores=rmn_scores)
        ransac_segs = geo.get('line_segments', {})
        ransac_lengths = geo['lengths']
        planes = geo['planes']
        plane_labels = geo['plane_labels']
        ridge_angle = geo['ridge_angle']

        seg_method = 'guided' if geo.get('used_guided') else 'RANSAC'
        print(f"  Segmentation ({seg_method}): {geo['n_planes']} planes, "
              f"type={geo['roof_type'][0]}")
        print(f"  Lengths: first={ransac_lengths['first_m']:.1f}m  "
              f"traufe={ransac_lengths['traufe_m']:.1f}m  "
              f"ortgang={ransac_lengths['ortgang_m']:.1f}m  "
              f"kehle={ransac_lengths['kehle_m']:.1f}m")
    except Exception as ex:
        print(f"  ERROR in roof analysis: {ex}")
        traceback.print_exc()
        result['error'] = str(ex)
        return result

    # ── Step 5: Classify RoofMapNet lines ───────────────────────────────────
    roof_type = geo['roof_type'][0]

    fusion = classify_rmn_lines_v10(
        rmn_lines, rmn_scores,
        ndom_arr, mask, plane_labels, planes,
        ridge_angle, gsd=gsd,
    )

    f_lengths = fusion['lengths']
    all_lines = fusion.get('classified_lines', [])
    accepted = [c for c in all_lines if c.get('status') == 'accepted']
    rejected = [c for c in all_lines if c.get('status') != 'accepted']

    print(f"  RoofMapNet classification: {len(all_lines)} lines total")
    print(f"    Accepted: {len(accepted)}", end="")
    by_type = {}
    for c in accepted:
        by_type.setdefault(c['type'], []).append(c)
    if by_type:
        parts = [f"{len(v)} {k}" for k, v in sorted(by_type.items())]
        print(f" ({', '.join(parts)})")
    else:
        print()
    rej_types = {}
    for c in rejected:
        rej_types.setdefault(c.get('type', '?'), []).append(c)
    if rej_types:
        parts = [f"{len(v)} {k}" for k, v in sorted(rej_types.items())]
        print(f"    Rejected: {len(rejected)} ({', '.join(parts)})")

    # ── Step 7: Visualization ────────────────────────────────────────────────
    try:
        visualize_rmn_classification(
            rgb_crop, fusion,
            os.path.join(out_dir, 'fusion_overlay.jpg'),
            ndom_arr.shape, vid,
            ndom_to_img=ndom_to_img,
        )
    except Exception as ex:
        print(f"  WARNING: visualization failed: {ex}")
        traceback.print_exc()

    # ── Step 7b: Superstructure detection (SegFormer) ─────────────────────────
    superstructures = None
    try:
        from superstructure_detector import detect_superstructures
        h_nd, w_nd = ndom_arr.shape
        h_dop, w_dop = dop_arr.shape[1], dop_arr.shape[2]
        from scipy.ndimage import zoom as ndzoom
        scale_factor = h_dop / h_nd
        mask_dop = ndzoom(mask.astype(float), scale_factor, order=0) > 0.5
        mask_dop = mask_dop[:h_dop, :w_dop]
        dop_hwc = np.transpose(dop_arr[:3], (1, 2, 0))
        superstructures = detect_superstructures(dop_hwc, mask_dop, gsd=0.10)
        print(f"\n  Dachaufbauten (SegFormer):")
        for cls_name in sorted(superstructures['counts'].keys()):
            cnt = superstructures['counts'][cls_name]
            area = superstructures['areas_m2'][cls_name]
            if cnt > 0:
                print(f"    {cls_name}: {cnt} Stk., {area:.1f} m²")
        if all(v == 0 for v in superstructures['counts'].values()):
            print(f"    (keine Aufbauten erkannt)")
    except Exception as ex:
        print(f"  WARNING: superstructure detection failed: {ex}")
        traceback.print_exc()

    # ── Step 7c: Separate visualizations (Masse + Aufbauten) ─────────────────
    try:
        visualize_masse(
            rgb_crop, fusion,
            os.path.join(out_dir, 'masse_overlay.jpg'),
            ndom_arr.shape, ndom_to_img=ndom_to_img,
        )
    except Exception as ex:
        print(f"  WARNING: masse visualization failed: {ex}")
        traceback.print_exc()
    try:
        visualize_aufbauten(
            rgb_crop, superstructures, None,
            os.path.join(out_dir, 'aufbauten_overlay.jpg'),
            ndom_arr.shape, ndom_to_img=ndom_to_img,
        )
    except Exception as ex:
        print(f"  WARNING: aufbauten visualization failed: {ex}")
        traceback.print_exc()

    # ── Step 8: Extract predictions ──────────────────────────────────────────
    pred_area = geo['total_3d_m2']
    pred_pitch = max((p['pitch_deg'] for p in geo['planes']), default=0)
    pred_nplanes = geo['n_planes']
    pred_type = geo['roof_type'][0]

    # Select best lengths per metric. Polygon wireframe gives better ridge
    # for complex buildings; RANSAC is better for traufe/kehle at coarse GSD.
    if gsd >= 1.0:
        poly_first = f_lengths.get('first_m', 0) if fusion.get('is_t_shape') else 0
        pred_first = poly_first if poly_first > 0 else ransac_lengths.get('first_m', 0)
        pred_traufe = ransac_lengths.get('traufe_m', 0)
        pred_ortgang = ransac_lengths.get('ortgang_m', 0)
        pred_kehle = ransac_lengths.get('kehle_m', 0)
        pred_grat = ransac_lengths.get('grat_m', 0)
    else:
        def _best_length(key):
            f_val = f_lengths.get(key, 0)
            r_val = ransac_lengths.get(key, 0)
            if f_val > 0:
                return f_val
            return r_val
        pred_first = _best_length('first_m')
        pred_traufe = _best_length('traufe_m')
        pred_ortgang = _best_length('ortgang_m')
        pred_kehle = _best_length('kehle_m')
        pred_grat = _best_length('grat_m')

    # ── Step 9: Ground truth comparison ──────────────────────────────────────
    has_gt = gt_row is not None
    if has_gt:
        gt_area = float(gt_row['total_area_m2'])
        gt_pitch = float(gt_row['dominant_pitch_deg'])
        gt_first = float(gt_row['first_m'])
        gt_traufe = float(gt_row['traufe_m'])
        gt_kehle = float(gt_row['kehle_m'])
        gt_ortgang = float(gt_row['ortgang_m'])
        gt_grat = float(gt_row['grat_m'])
        gt_nplanes = int(gt_row['n_planes'])
        gt_type = str(gt_row['roof_type'])

        print(f"\n  GT roof type: {gt_type}  |  Predicted: {pred_type}")
        print(f"\n  {'Metric':<22} {'Ground Truth':>12} {'Predicted':>12} {'Error':>12}")
        print(f"  {'─'*60}")

        metrics = [
            ('Total 3D area (m2)', gt_area, pred_area, '%'),
            ('Dominant pitch (deg)', gt_pitch, pred_pitch, 'deg'),
            ('N planes', gt_nplanes, pred_nplanes, 'abs'),
            ('First (m)', gt_first, pred_first, '%'),
            ('Traufe (m)', gt_traufe, pred_traufe, '%'),
            ('Ortgang (m)', gt_ortgang, pred_ortgang, '%'),
            ('Kehle (m)', gt_kehle, pred_kehle, '%'),
            ('Grat (m)', gt_grat, pred_grat, '%'),
        ]

        for name, gt_val, pred_val, err_type in metrics:
            gt_str = f"{gt_val:.1f}" if isinstance(gt_val, float) else str(gt_val)
            pred_str = f"{pred_val:.1f}" if isinstance(pred_val, float) else str(pred_val)

            if err_type == '%':
                err = pct_error(pred_val, gt_val)
                err_str = f"{err:+.1f}%" if not np.isnan(err) else "n/a"
            elif err_type == 'deg':
                err = pred_val - gt_val
                err_str = f"{err:+.1f}deg"
            elif err_type == 'abs':
                err = pred_val - gt_val
                err_str = f"{err:+d}" if isinstance(err, int) else f"{err:+.0f}"
            else:
                err_str = ""

            print(f"  {name:<22} {gt_str:>12} {pred_str:>12} {err_str:>12}")

        gt_plane_str = gt_row.get('plane_areas_m2', '')
        gt_pitch_str = gt_row.get('pitch_values_deg', '')
        if isinstance(gt_plane_str, str) and gt_plane_str:
            print(f"\n  GT planes: {gt_plane_str}")
            print(f"  GT pitches: {gt_pitch_str}")
    else:
        gt_area = gt_pitch = gt_first = gt_traufe = gt_kehle = 0
        gt_ortgang = gt_grat = 0
        gt_nplanes = 0
        gt_type = 'unknown'
        print(f"\n  Predicted: {pred_type}  (no ground truth)")
        print(f"\n  {'Metric':<22} {'Predicted':>12}")
        print(f"  {'─'*36}")
        for name, pred_val in [
            ('Total 3D area (m2)', pred_area),
            ('Dominant pitch (deg)', pred_pitch),
            ('N planes', pred_nplanes),
            ('First (m)', pred_first),
            ('Traufe (m)', pred_traufe),
            ('Ortgang (m)', pred_ortgang),
            ('Kehle (m)', pred_kehle),
            ('Grat (m)', pred_grat),
        ]:
            pred_str = f"{pred_val:.1f}" if isinstance(pred_val, float) else str(pred_val)
            print(f"  {name:<22} {pred_str:>12}")

    # Plane detail
    if geo['planes']:
        print(f"\n  Detected planes:")
        for p in sorted(geo['planes'], key=lambda x: -x['area_3d_m2']):
            print(f"    Plane {p['label']}: {p['area_3d_m2']:>7.1f} m2  "
                  f"pitch={p['pitch_deg']:>5.1f}deg  aspect={p['aspect_deg']:>5.0f}deg")

    # ── Build result dict ────────────────────────────────────────────────────
    result.update({
        'state': state,
        'gsd': gsd,
        'method': method,
        'pred_roof_type': pred_type,
        'gt_roof_type': gt_type,
        'pred': {
            'total_area_m2': round(pred_area, 2),
            'dominant_pitch_deg': round(pred_pitch, 1),
            'n_planes': pred_nplanes,
            'first_m': round(pred_first, 2),
            'traufe_m': round(pred_traufe, 2),
            'ortgang_m': round(pred_ortgang, 2),
            'kehle_m': round(pred_kehle, 2),
            'grat_m': round(pred_grat, 2),
        },
        'n_classified_lines': len(accepted),
        'fusion_source': fusion.get('source', 'unknown'),
        'ransac_lengths': {k: round(v, 2) for k, v in ransac_lengths.items()},
        'fusion_lengths': {k: round(v, 2) for k, v in f_lengths.items()},
    })
    if has_gt:
        result['gt'] = {
            'total_area_m2': gt_area,
            'dominant_pitch_deg': gt_pitch,
            'n_planes': gt_nplanes,
            'first_m': gt_first,
            'traufe_m': gt_traufe,
            'ortgang_m': gt_ortgang,
            'kehle_m': gt_kehle,
            'grat_m': gt_grat,
        }
        result['errors_pct'] = {
            'total_area_m2': round(pct_error(pred_area, gt_area), 2),
            'first_m': round(pct_error(pred_first, gt_first), 2)
                       if gt_first > 0 else None,
            'traufe_m': round(pct_error(pred_traufe, gt_traufe), 2)
                        if gt_traufe > 0 else None,
            'ortgang_m': round(pct_error(pred_ortgang, gt_ortgang), 2)
                         if gt_ortgang > 0 else None,
            'kehle_m': round(pct_error(pred_kehle, gt_kehle), 2)
                       if gt_kehle > 0 else None,
            'grat_m': round(pct_error(pred_grat, gt_grat), 2)
                      if gt_grat > 0 else None,
        }

    # Superstructure results
    if superstructures is not None:
        result['superstructures'] = {
            'counts': superstructures['counts'],
            'areas_m2': {k: round(v, 2) for k, v in superstructures['areas_m2'].items()},
        }

    # Save result JSON
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Save measurements text file
    with open(os.path.join(out_dir, 'measurements.txt'), 'w') as f:
        f.write(f"{vid}: {bldg['address']}, {bldg['city']}\n")
        f.write(f"{'─' * 40}\n")
        f.write(f"Dachtyp:           {pred_type}\n")
        f.write(f"Gesamtfläche:      {pred_area:.1f} m²\n")
        f.write(f"Dachneigung:       {pred_pitch:.1f}°\n")
        f.write(f"Anzahl Ebenen:     {pred_nplanes}\n\n")
        f.write(f"First:             {pred_first:.1f} m\n")
        f.write(f"Traufe (gesamt):   {pred_traufe:.1f} m\n")
        f.write(f"Ortgang (gesamt):  {pred_ortgang:.1f} m\n")
        f.write(f"Kehle:             {pred_kehle:.1f} m\n")
        f.write(f"Grat:              {pred_grat:.1f} m\n\n")
        f.write(f"Einzellinien:\n")
        for cl in accepted:
            f.write(f"  {cl['type']:<10} {cl['length_m']:>6.1f} m\n")
        f.write(f"\nDachaufbauten:\n")
        if superstructures is not None:
            for cls_name in sorted(superstructures['counts'].keys()):
                cnt = superstructures['counts'][cls_name]
                area = superstructures['areas_m2'][cls_name]
                if cnt > 0:
                    f.write(f"  {cls_name:<15} {cnt} Stk.  {area:.1f} m²\n")
            if all(v == 0 for v in superstructures['counts'].values()):
                f.write(f"  (keine erkannt)\n")

    print(f"\n  Saved to {out_dir}/")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results):
    """Print summary table with MAPE per metric across all buildings."""
    valid = [r for r in results if 'error' not in r and 'gt' in r]
    if not valid:
        print("\nNo valid results for summary.")
        return

    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: Predicted vs Airteam Ground Truth  (n={len(valid)} buildings)")
    print(f"{'='*80}")

    metric_names = [
        ('total_area_m2', 'Total 3D area (m2)'),
        ('first_m', 'First (m)'),
        ('traufe_m', 'Traufe (m)'),
        ('ortgang_m', 'Ortgang (m)'),
        ('kehle_m', 'Kehle (m)'),
        ('grat_m', 'Grat (m)'),
    ]

    print(f"\n  {'Metric':<22} ", end='')
    for r in valid:
        print(f"  {r['id']:>8}", end='')
    print(f"  {'MAPE':>8}")
    print(f"  {'─'*(24 + 10*len(valid) + 10)}")

    # Pitch error (absolute degrees, not percentage)
    print(f"  {'Dominant pitch (deg)':<22} ", end='')
    pitch_errors = []
    for r in valid:
        gt_p = r['gt']['dominant_pitch_deg']
        pr_p = r['pred']['dominant_pitch_deg']
        err = pr_p - gt_p
        pitch_errors.append(abs(err))
        print(f"  {err:>+7.1f}d", end='')
    mae_pitch = np.mean(pitch_errors) if pitch_errors else float('nan')
    print(f"  {mae_pitch:>7.1f}d")

    # N planes (absolute difference)
    print(f"  {'N planes':<22} ", end='')
    plane_diffs = []
    for r in valid:
        diff = r['pred']['n_planes'] - r['gt']['n_planes']
        plane_diffs.append(abs(diff))
        print(f"  {diff:>+8d}", end='')
    mae_planes = np.mean(plane_diffs) if plane_diffs else float('nan')
    print(f"  {mae_planes:>8.1f}")

    # Percentage metrics
    for key, label in metric_names:
        print(f"  {label:<22} ", end='')
        abs_errors = []
        for r in valid:
            gt_val = r['gt'][key]
            pr_val = r['pred'][key]
            if gt_val > 0:
                err = pct_error(pr_val, gt_val)
                abs_errors.append(abs(err))
                print(f"  {err:>+7.1f}%", end='')
            else:
                print(f"  {'n/a':>8}", end='')
        mape = np.mean(abs_errors) if abs_errors else float('nan')
        print(f"  {mape:>7.1f}%")

    # Overall MAPE (across all length metrics)
    all_abs = []
    for r in valid:
        for key, _ in metric_names:
            gt_val = r['gt'][key]
            pr_val = r['pred'][key]
            if gt_val > 0:
                all_abs.append(abs(pct_error(pr_val, gt_val)))
    overall_mape = np.mean(all_abs) if all_abs else float('nan')

    print(f"\n  Overall MAPE (length metrics): {overall_mape:.1f}%")
    print(f"  Mean pitch error (MAE): {mae_pitch:.1f} degrees")
    print(f"  Mean plane count error (MAE): {mae_planes:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load ground truth
    gt_path = os.path.join(BASE, 'data', 'validation_buildings.csv')
    if not os.path.exists(gt_path):
        print(f"ERROR: ground truth file not found at {gt_path}")
        sys.exit(1)

    gt = pd.read_csv(gt_path)
    gt = gt.set_index('id')

    # Parse command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].upper() == 'ALL':
            test_ids = list(BUILDINGS.keys())
        else:
            test_ids = [a.upper() for a in sys.argv[1:]]
    else:
        test_ids = ['V1']

    # Validate requested IDs
    test_ids = [vid for vid in test_ids if vid in BUILDINGS]
    if not test_ids:
        print(f"ERROR: no valid building IDs. Available: {sorted(BUILDINGS.keys())}")
        sys.exit(1)

    print(f"Validation buildings: {', '.join(test_ids)}")
    print(f"Ground truth: {gt_path}")

    results = []
    for vid in test_ids:
        bldg = BUILDINGS[vid]
        gt_row = gt.loc[vid] if vid in gt.index else None
        try:
            result = process_building(vid, bldg, gt_row)
            results.append(result)
        except Exception as ex:
            print(f"\n  FATAL ERROR processing {vid}: {ex}")
            traceback.print_exc()
            results.append({'id': vid, 'error': str(ex)})

    # Summary
    print_summary(results)

    # Save all results
    summary_dir = os.path.join(BASE, 'results', 'validation')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        # Filter out non-serializable items
        clean = []
        for r in results:
            clean.append({k: v for k, v in r.items()
                         if not isinstance(v, np.ndarray)})
        json.dump(clean, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
