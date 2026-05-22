"""
Robust building detection — precision pipeline:

1. ALKIS+SAM (primary)
   ALKIS WFS identifies the target building.  SAM segments the roof on the
   DOP (10 cm/px).  Output is refined at DOP resolution: constrained to
   ALKIS polygon + eave buffer, height-filtered, NDVI-filtered, then
   downsampled to nDOM (0.5 m/px).

2. ALKIS+GrabCut (secondary)
   Same ALKIS identification + DOP-resolution GrabCut + same refinement.

3. ALKIS+nDOM (tertiary)
   nDOM height segmentation constrained to ALKIS polygon + buffer.

4. NDVI+nDOM (quaternary — no ALKIS)
   Pure nDOM + NIR fallback when ALKIS is unavailable.

Returns a binary pixel mask at nDOM resolution (0.5 m/px).
"""

import numpy as np
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from matplotlib.path import Path as MplPath
from scipy.ndimage import (
    binary_closing, binary_dilation, binary_erosion, binary_opening,
    binary_fill_holes, label as ndi_label, zoom,
)

ALKIS_WFS  = "https://www.wfs.nrw.de/geobasis/wfs_nw_alkis_vereinfacht"
GML_NS     = "http://www.opengis.net/gml/3.2"
NDOM_RES   = 0.5   # m/px for nDOM
DOP_RES    = 0.10  # m/px for DOP
SCALE      = int(NDOM_RES / DOP_RES)   # 5  — nDOM px → DOP px
MIN_AREA   = 20.0  # m²
MAX_AREA   = 350.0 # m²


# ── ALKIS helpers ─────────────────────────────────────────────────────────────

GARAGE_FUNCTIONS = {'Garage', 'Gebäude zum Parken', 'Tiefgarage',
                    'Carport', 'Parkhaus'}

def _query_alkis(e_min, e_max, n_min, n_max, timeout=15,
                 exclude_garages=True):
    url = (f"{ALKIS_WFS}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
           f"&TYPENAMES=ave:GebaeudeBauwerk"
           f"&BBOX={int(e_min)},{int(n_min)},{int(e_max)},{int(n_max)},EPSG:25832"
           f"&SRSNAME=EPSG:25832")
    try:
        r = requests.get(url, timeout=timeout)
    except Exception:
        return []
    if not r.ok:
        return []
    root = ET.fromstring(r.content)
    AVE_NS = ("http://repository.gdi-de.org/schemas/adv/produkt/"
              "alkis-vereinfacht/2.0")
    rings = []
    for member in root.findall(f".//{{{AVE_NS}}}GebaeudeBauwerk"):
        funktion = ''
        fk_el = member.find(f"{{{AVE_NS}}}funktion")
        if fk_el is None:
            fk_el = member.find("funktion")
        if fk_el is not None and fk_el.text:
            funktion = fk_el.text.strip()
        if exclude_garages and funktion in GARAGE_FUNCTIONS:
            continue
        for poly in member.findall(f".//{{{GML_NS}}}Polygon"):
            el = poly.find(f".//{{{GML_NS}}}posList")
            if el is None or not el.text:
                continue
            nums = list(map(float, el.text.split()))
            rings.append(np.array(nums).reshape(-1, 2))
    return rings


def _rasterize_polygon(ring_utm, e_min, n_max, res, h, w):
    px = (ring_utm[:, 0] - e_min) / res
    py = (n_max - ring_utm[:, 1]) / res
    path = MplPath(np.column_stack([px, py]))
    cols_g, rows_g = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return path.contains_points(np.column_stack([cols_g.ravel(), rows_g.ravel()])).reshape(h, w)


def _select_best_alkis(rings, ndom_arr, e_min, n_min):
    """Score all ALKIS rings, return (best_mask_ndom, centroid_row, centroid_col)."""
    h, w  = ndom_arr.shape
    e_max = e_min + w * NDOM_RES
    n_max = n_min + h * NDOM_RES
    cx, cy = w / 2, h / 2

    min_px = int(MIN_AREA / NDOM_RES ** 2)
    max_px = int(MAX_AREA / NDOM_RES ** 2)
    best_m, best_score, best_cr, best_cc = None, -np.inf, None, None

    for ring in rings:
        m = _rasterize_polygon(ring, e_min, n_max, NDOM_RES, h, w)
        n_px = m.sum()
        if not (min_px <= n_px <= max_px):
            continue
        rows, cols = np.where(m)
        if len(rows) == 0:
            continue
        cr, cc = rows.mean(), cols.mean()
        dist = np.hypot(cr - cy, cc - cx)
        vals = ndom_arr[m]
        mean_h = float(vals[vals > 0].mean()) if (vals > 0).any() else 0.0
        score = mean_h * 3.0 - dist * 0.5
        if score > best_score:
            best_score, best_m, best_cr, best_cc = score, m, cr, cc

    return best_m, best_cr, best_cc


def _rings_share_wall(ring_a, ring_b, tol_m=0.5, min_shared_pts=2):
    """Check if two ALKIS polygon rings share a wall (common boundary edge).

    Doppelhaus/Reihenhaus units share party-wall edges in ALKIS — their
    polygon vertices overlap along the shared boundary.  Detached garages
    and neighboring houses do NOT share edges.

    Returns True if at least *min_shared_pts* vertices of ring_b are within
    *tol_m* of the boundary of ring_a (or vice versa).
    """
    from scipy.spatial import cKDTree
    pts_a = ring_a[:-1] if np.allclose(ring_a[0], ring_a[-1]) else ring_a
    pts_b = ring_b[:-1] if np.allclose(ring_b[0], ring_b[-1]) else ring_b
    tree_a = cKDTree(pts_a)
    dists, _ = tree_a.query(pts_b)
    if np.sum(dists <= tol_m) >= min_shared_pts:
        return True
    tree_b = cKDTree(pts_b)
    dists2, _ = tree_b.query(pts_a)
    return np.sum(dists2 <= tol_m) >= min_shared_pts


def _merge_alkis_nearby(rings, ndom_arr, e_min, n_min, max_gap_m=3.0,
                        max_height_diff_m=3.0, max_units=None):
    """Merge adjacent ALKIS polygons that share a wall into a single mask.

    For Doppelhaus / Reihenhaus where each unit is a separate ALKIS polygon.
    Uses shared-wall detection: only merges polygons whose ALKIS boundaries
    share at least 2 vertices within 0.5 m (party wall).  This prevents
    merging detached garages or neighboring houses that are merely close.

    max_units: if set, stop after merging this many polygons total (seed
    counts as 1).  Prioritises polygons closest to the tile center.
    """
    h, w = ndom_arr.shape
    n_max = n_min + h * NDOM_RES
    cx, cy = w / 2, h / 2

    min_px = int(MIN_AREA / NDOM_RES ** 2)

    valid_rings = []
    masks = []
    mean_heights = []
    centroids = []
    for ring in rings:
        m = _rasterize_polygon(ring, e_min, n_max, NDOM_RES, h, w)
        if m.sum() >= min_px:
            vals = ndom_arr[m]
            mh = float(vals[vals > 0].mean()) if (vals > 0).any() else 0.0
            rows, cols = np.where(m)
            cr, cc = rows.mean(), cols.mean()
            valid_rings.append(ring)
            masks.append(m)
            mean_heights.append(mh)
            centroids.append((cr, cc))

    if not masks:
        return None, None, None

    dists = [np.hypot(cr - cy, cc - cx) for cr, cc in centroids]

    seed_idx = int(np.argmin(dists))
    merged = masks[seed_idx].copy()
    used = {seed_idx}

    changed = True
    while changed:
        if max_units is not None and len(used) >= max_units:
            break
        changed = False
        candidates = []
        for i in range(len(masks)):
            if i in used:
                continue
            shares_wall = any(
                _rings_share_wall(valid_rings[j], valid_rings[i], tol_m=0.5)
                for j in used
            )
            if shares_wall:
                candidates.append(i)
        if not candidates:
            break
        candidates.sort(key=lambda i: dists[i])
        best = candidates[0]
        merged = merged | masks[best]
        used.add(best)
        changed = True

    rows, cols = np.where(merged)
    if len(rows) == 0:
        return None, None, None
    return merged, rows.mean(), cols.mean()


# ── Precision mask refinement at DOP resolution ──────────────────────────────

def _refine_mask_hires(raw_mask_dop, alkis_mask_ndom, ndom_arr, dop_arr,
                       centroid_ndom, eave_buffer_m=0.7):
    """
    Precision post-processing at DOP resolution (10 cm/px).

    1. Constrain to ALKIS polygon + eave overhang buffer
    2. Adaptive nDOM height filter
    3. NDVI vegetation removal
    4. Connected component from building centroid
    5. Morphological cleanup + hole filling
    6. Downsample to nDOM resolution
    """
    h_nd, w_nd = ndom_arr.shape
    h_dop, w_dop = dop_arr.shape[1], dop_arr.shape[2]

    # 1. ALKIS at DOP resolution + eave buffer
    alkis_dop = zoom(alkis_mask_ndom.astype(float), SCALE, order=0) > 0.5
    alkis_dop = alkis_dop[:h_dop, :w_dop]
    buf_px = max(1, int(round(eave_buffer_m / DOP_RES)))
    constraint = binary_dilation(alkis_dop, iterations=buf_px)
    mask = raw_mask_dop[:h_dop, :w_dop] & constraint

    # 2. Adaptive nDOM height filter (interpolated to DOP resolution)
    ndom_dop = zoom(ndom_arr.astype(float), SCALE, order=1)[:h_dop, :w_dop]
    alkis_heights = ndom_arr[alkis_mask_ndom & (ndom_arr > 0)]
    if alkis_heights.size > 10:
        h_thresh = max(1.5, float(np.percentile(alkis_heights, 10)) * 0.4)
    else:
        h_thresh = 2.0
    mask = mask & (ndom_dop > h_thresh)

    # 3. NDVI vegetation filter at DOP resolution
    if dop_arr.shape[0] >= 4:
        R = dop_arr[0, :h_dop, :w_dop].astype(float)
        NIR = dop_arr[3, :h_dop, :w_dop].astype(float)
        ndvi = (NIR - R) / (NIR + R + 1e-6)
        mask = mask & (ndvi < 0.20)

    if not mask.any():
        return None

    # 4. Connected component from centroid
    labeled, n = ndi_label(mask)
    cr_d = int(np.clip(round(centroid_ndom[0] * SCALE), 0, h_dop - 1))
    cc_d = int(np.clip(round(centroid_ndom[1] * SCALE), 0, w_dop - 1))
    target = int(labeled[cr_d, cc_d])
    if target == 0:
        ar, ac = np.where(alkis_dop)
        for r, c in zip(ar[::5], ac[::5]):
            lbl = int(labeled[r, c])
            if lbl > 0:
                target = lbl
                break
    if target > 0:
        mask = labeled == target
    elif n > 0:
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        mask = labeled == (np.argmax(sizes) + 1)

    if not mask.any():
        return None

    # 5. Morphological cleanup at DOP resolution (smooth boundaries)
    mask = binary_closing(mask, iterations=3)
    mask = binary_fill_holes(mask)
    labeled2, n2 = ndi_label(mask)
    if n2 > 1:
        sizes = [(labeled2 == i).sum() for i in range(1, n2 + 1)]
        mask = labeled2 == (np.argmax(sizes) + 1)

    # 6. Downsample to nDOM resolution
    mask_ndom = zoom(mask.astype(float), 1.0 / SCALE, order=1) > 0.4
    mask_ndom = mask_ndom[:h_nd, :w_nd]

    return mask_ndom.astype(bool) if mask_ndom.any() else None


def _alkis_only_mask(alkis_mask_ndom, ndom_arr, dop_arr, centroid_ndom,
                     eave_buffer_m=0.5):
    """
    Fallback when SAM/GrabCut both fail: use ALKIS polygon + small buffer,
    filtered by nDOM height and NDVI.
    """
    h_nd, w_nd = ndom_arr.shape
    h_dop, w_dop = dop_arr.shape[1], dop_arr.shape[2]

    alkis_dop = zoom(alkis_mask_ndom.astype(float), SCALE, order=0) > 0.5
    alkis_dop = alkis_dop[:h_dop, :w_dop]
    buf_px = max(1, int(round(eave_buffer_m / DOP_RES)))
    mask = binary_dilation(alkis_dop, iterations=buf_px)

    ndom_dop = zoom(ndom_arr.astype(float), SCALE, order=1)[:h_dop, :w_dop]
    alkis_heights = ndom_arr[alkis_mask_ndom & (ndom_arr > 0)]
    h_thresh = max(1.5, float(np.percentile(alkis_heights, 10)) * 0.4) \
        if alkis_heights.size > 10 else 2.0
    mask = mask & (ndom_dop > h_thresh)

    if dop_arr.shape[0] >= 4:
        R = dop_arr[0, :h_dop, :w_dop].astype(float)
        NIR = dop_arr[3, :h_dop, :w_dop].astype(float)
        ndvi = (NIR - R) / (NIR + R + 1e-6)
        mask = mask & (ndvi < 0.20)

    if not mask.any():
        return None

    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)

    mask_ndom = zoom(mask.astype(float), 1.0 / SCALE, order=1) > 0.4
    mask_ndom = mask_ndom[:h_nd, :w_nd]
    return mask_ndom.astype(bool) if mask_ndom.any() else None


# ── SAM segmentation on DOP ───────────────────────────────────────────────────

def _sam_roof(dop_arr, alkis_mask_ndom, centroid_ndom, tile_key=None):
    """
    Use Meta SAM (Segment Anything) to get a pixel-precise roof boundary on the
    full-resolution DOP (10 cm/px).

    Prompts:
      * foreground points: 5 points sampled from the eroded ALKIS polygon interior
      * bounding box: ALKIS polygon bbox + 2 m buffer (DOP pixels, XYXY)
      * multimask: 3 candidates → pick highest IoU with the ALKIS polygon

    Returns (mask_ndom, method_str) or (None, None) if SAM unavailable or fails.
    """
    try:
        from sam_client import get_sam_client
    except Exception:
        return None, None

    sam = get_sam_client()
    if not sam.available:
        return None, None

    h_nd, w_nd = alkis_mask_ndom.shape
    h_dop, w_dop = dop_arr.shape[1], dop_arr.shape[2]

    # DOP RGB (H, W, 3) uint8 — SAM expects RGB
    rgb = np.transpose(dop_arr[:3], (1, 2, 0)).astype(np.uint8)

    # ── Prompts from ALKIS polygon ───────────────────────────────────────────
    alkis_inner = binary_erosion(alkis_mask_ndom, iterations=2)
    if not alkis_inner.any():
        alkis_inner = alkis_mask_ndom

    rows_in, cols_in = np.where(alkis_inner)

    # Sample up to 6 well-separated interior points
    n_pts = min(6, len(rows_in))
    if n_pts == 0:
        return None, None
    idx = np.linspace(0, len(rows_in) - 1, n_pts).astype(int)
    points_nd = list(zip(cols_in[idx], rows_in[idx]))
    points_dop = [(int(c * SCALE + SCALE // 2), int(r * SCALE + SCALE // 2))
                  for c, r in points_nd]
    labels = [1] * len(points_dop)

    # Bounding box in DOP pixel coords
    rows_full, cols_full = np.where(alkis_mask_ndom)
    pad = 4   # nDOM px (= 2 m)
    r0 = max(0, rows_full.min() - pad) * SCALE
    c0 = max(0, cols_full.min() - pad) * SCALE
    r1 = min(h_nd, rows_full.max() + pad + 1) * SCALE
    c1 = min(w_nd, cols_full.max() + pad + 1) * SCALE
    box = [c0, r0, c1, r1]   # XYXY

    # ── Run SAM (returns the highest-score mask by default) ──────────────────
    mask_dop, info = sam.segment(
        rgb,
        points=points_dop,
        labels=labels,
        box=box,
        multimask=True,
        select="highest_score",
        image_key=tile_key,
    )
    if mask_dop is None:
        return None

    return mask_dop.astype(bool)


# ── GrabCut segmentation on DOP ───────────────────────────────────────────────

def _grabcut_roof(dop_arr, alkis_mask_ndom, centroid_ndom):
    """
    Run GrabCut on the DOP image using the ALKIS polygon as a foreground prior.

    Parameters
    ----------
    dop_arr        : (4, H_dop, W_dop) uint8 RGBI
    alkis_mask_ndom: (H_nd, W_nd) bool  ALKIS polygon at nDOM resolution
    centroid_ndom  : (row, col) float   ALKIS centroid in nDOM pixel coords

    Returns
    -------
    mask_ndom : (H_nd, W_nd) bool  or None on failure
    """
    try:
        import cv2
    except ImportError:
        return None

    h_nd, w_nd = alkis_mask_ndom.shape
    h_dop, w_dop = dop_arr.shape[1], dop_arr.shape[2]

    # DOP RGB (H_dop, W_dop, 3) — GrabCut needs BGR uint8
    rgb = np.transpose(dop_arr[:3], (1, 2, 0)).astype(np.uint8)
    bgr = rgb[:, :, ::-1].copy()

    # ── bounding rect: ALKIS polygon + 2 m (20 px @ 10 cm) buffer ───────────
    rows_nd, cols_nd = np.where(alkis_mask_ndom)
    r_min_nd = max(0, int(rows_nd.min()) - 4)   # 4 nDOM px = 2 m
    r_max_nd = min(h_nd - 1, int(rows_nd.max()) + 4)
    c_min_nd = max(0, int(cols_nd.min()) - 4)
    c_max_nd = min(w_nd - 1, int(cols_nd.max()) + 4)

    # Scale to DOP pixels
    r_min_d = r_min_nd * SCALE
    r_max_d = min(h_dop - 1, r_max_nd * SCALE + SCALE - 1)
    c_min_d = c_min_nd * SCALE
    c_max_d = min(w_dop - 1, c_max_nd * SCALE + SCALE - 1)
    rect = (c_min_d, r_min_d, c_max_d - c_min_d, r_max_d - r_min_d)

    if rect[2] < 10 or rect[3] < 10:
        return None

    # ── GrabCut seed mask from ALKIS polygon ─────────────────────────────────
    gc_mask = np.full((h_dop, w_dop), cv2.GC_PR_BGD, dtype=np.uint8)

    # Pixels strictly inside ALKIS polygon → probable foreground
    # (erode by 1 nDOM px = 5 DOP px to avoid touching boundary)
    alkis_inner = binary_erosion(alkis_mask_ndom, iterations=1)
    alkis_dop = zoom(alkis_inner.astype(np.uint8), SCALE, order=0)
    alkis_dop = alkis_dop[:h_dop, :w_dop].astype(bool)
    gc_mask[alkis_dop] = cv2.GC_FGD

    # Narrow band around the ALKIS boundary → let GrabCut decide
    alkis_full_dop = zoom(alkis_mask_ndom.astype(np.uint8), SCALE, order=0)
    alkis_full_dop = alkis_full_dop[:h_dop, :w_dop].astype(bool)
    boundary_dop = alkis_full_dop & ~alkis_dop
    gc_mask[boundary_dop] = cv2.GC_PR_FGD

    # ── Run GrabCut ───────────────────────────────────────────────────────────
    bgr_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc_mask, rect, bgr_model, fgd_model,
                    iterCount=5, mode=cv2.GC_INIT_WITH_MASK)
    except Exception:
        try:
            # Fallback: init with rect only (ignores seed mask)
            gc_mask2 = np.zeros((h_dop, w_dop), dtype=np.uint8)
            bgr_m2 = np.zeros((1, 65), np.float64)
            fgd_m2 = np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr, gc_mask2, rect, bgr_m2, fgd_m2,
                        iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
            gc_mask = gc_mask2
        except Exception:
            return None

    # GrabCut result: FGD or PR_FGD pixels are foreground
    fg_dop = np.isin(gc_mask, [cv2.GC_FGD, cv2.GC_PR_FGD])

    # ── Select connected component containing ALKIS centroid ─────────────────
    labeled, n = ndi_label(fg_dop)
    if n == 0:
        return None

    cr_nd, cc_nd = centroid_ndom
    cr_d = int(np.clip(round(cr_nd * SCALE), 0, h_dop - 1))
    cc_d = int(np.clip(round(cc_nd * SCALE), 0, w_dop - 1))
    target_lbl = int(labeled[cr_d, cc_d])

    if target_lbl == 0:
        # Scan ALKIS pixels for a foreground label
        for r, c in zip(rows_nd[::3], cols_nd[::3]):
            lbl = int(labeled[min(r * SCALE, h_dop - 1),
                               min(c * SCALE, w_dop - 1)])
            if lbl > 0:
                target_lbl = lbl
                break

    if target_lbl == 0:
        return None

    fg_selected = labeled == target_lbl  # (H_dop, W_dop) bool

    return fg_selected.astype(bool)


# ── NDVI / vegetation helpers ─────────────────────────────────────────────────

def _ndvi(dop_arr):
    if dop_arr.shape[0] < 4:
        return None
    r_ds  = zoom(dop_arr[0].astype(float), DOP_RES / NDOM_RES, order=1)
    nir_ds = zoom(dop_arr[3].astype(float), DOP_RES / NDOM_RES, order=1)
    eps = 1e-6
    return (nir_ds - r_ds) / (nir_ds + r_ds + eps)


def _not_vegetation_mask(dop_arr, h, w):
    ndvi = _ndvi(dop_arr)
    if ndvi is None:
        return np.ones((h, w), dtype=bool)
    hn, wn = ndvi.shape
    ndvi_m = ndvi[:h, :w] if (hn >= h and wn >= w) else \
        np.pad(ndvi, ((0, max(0, h - hn)), (0, max(0, w - wn))))[:h, :w]
    return ndvi_m < 0.20


# ── nDOM segmentation fallback ────────────────────────────────────────────────

def _ndom_roof_mask(ndom_arr, dop_arr, alkis_mask, centroid):
    """
    Derive roof mask from nDOM + NDVI, tightly clipped to ALKIS + 1 px buffer.
    Uses relaxed NDVI threshold inside ALKIS polygon (moss/lichen tolerance).
    """
    h, w = ndom_arr.shape
    min_px = int(MIN_AREA / NDOM_RES ** 2)
    eave_px = 2  # 1 m at 0.5 m/px

    alkis_heights = ndom_arr[alkis_mask & (ndom_arr > 0)]
    h_thresh = max(1.5, float(np.percentile(alkis_heights, 10)) * 0.4) \
        if alkis_heights.size > 0 else 1.5

    not_veg   = _not_vegetation_mask(dop_arr, h, w)
    candidate = binary_closing((ndom_arr > h_thresh) & not_veg, iterations=1)
    labeled, n = ndi_label(candidate)
    if n == 0:
        return None

    alkis_expanded = binary_dilation(alkis_mask, iterations=eave_px)
    cr_a, cc_a = centroid
    seed_r = int(np.clip(round(cr_a), 0, h - 1))
    seed_c = int(np.clip(round(cc_a), 0, w - 1))
    target_lbl = int(labeled[seed_r, seed_c])

    if target_lbl == 0:
        ar, ac = np.where(alkis_mask)
        for r, c in zip(ar[::3], ac[::3]):
            lbl = int(labeled[r, c])
            if lbl > 0:
                target_lbl = lbl
                break

    if target_lbl > 0:
        comp = (labeled == target_lbl) & alkis_expanded
        if comp.sum() >= min_px // 2:
            return comp

    best, best_ov = None, 0
    for bid in range(1, n + 1):
        comp = (labeled == bid) & alkis_expanded
        ov = int((comp & alkis_mask).sum())
        if ov > best_ov:
            best_ov, best = ov, comp
    return best if (best is not None and best.sum() >= min_px // 2) else None


# ── Detectors ─────────────────────────────────────────────────────────────────

def detect_with_alkis(ndom_arr, dop_arr, e_min, n_min, tile_key=None,
                      merge_alkis=False, merge_gap_m=3.0, max_units=None):
    """
    Precision pipeline:
      1. ALKIS → building ID
      2. SAM/GrabCut on DOP → raw mask at 10 cm/px
      3. Refine at DOP resolution (ALKIS constraint + height + NDVI)
      4. Downsample to nDOM resolution
    Returns (mask, method_str) or (None, None).
    """
    h, w  = ndom_arr.shape
    e_max = e_min + w * NDOM_RES
    n_max = n_min + h * NDOM_RES

    rings = _query_alkis(e_min, e_max, n_min, n_max)
    if not rings:
        return None, None

    if merge_alkis:
        best_alkis, cr_a, cc_a = _merge_alkis_nearby(
            rings, ndom_arr, e_min, n_min, max_gap_m=merge_gap_m,
            max_units=max_units)
    else:
        best_alkis, cr_a, cc_a = _select_best_alkis(
            rings, ndom_arr, e_min, n_min)
    if best_alkis is None:
        return None, None

    centroid = (cr_a, cc_a)
    min_px = int(MIN_AREA / NDOM_RES ** 2)

    # ── 1. Try SAM → refine at DOP resolution ────────────────────────────────
    sam_dop = _sam_roof(dop_arr, best_alkis, centroid, tile_key=tile_key)
    if sam_dop is not None and sam_dop.any():
        mask = _refine_mask_hires(sam_dop, best_alkis, ndom_arr, dop_arr,
                                  centroid)
        if mask is not None and mask.sum() >= min_px // 2:
            return mask, "ALKIS+SAM"

    # ── 2. Fallback: GrabCut → refine at DOP resolution ──────────────────────
    gc_dop = _grabcut_roof(dop_arr, best_alkis, centroid)
    if gc_dop is not None and gc_dop.any():
        mask = _refine_mask_hires(gc_dop, best_alkis, ndom_arr, dop_arr,
                                  centroid)
        if mask is not None and mask.sum() >= min_px // 2:
            return mask, "ALKIS+GrabCut"

    # No fallback — if SAM and GrabCut both fail, report failure
    return None, None


def detect_with_ndvi(ndom_arr, dop_arr):
    """
    Tertiary fallback: nDOM threshold + NDVI + centroid proximity.
    Returns (mask, method_str) or (None, None).
    """
    h, w   = ndom_arr.shape
    cx, cy = w / 2, h / 2

    not_veg   = _not_vegetation_mask(dop_arr, h, w)
    candidate = (ndom_arr > 3.0) & not_veg
    labeled, n = ndi_label(candidate)
    if n == 0:
        return None, None

    min_px = int(MIN_AREA / NDOM_RES ** 2)
    max_px = int(MAX_AREA / NDOM_RES ** 2)
    best, best_dist = None, np.inf

    for bid in range(1, n + 1):
        m = labeled == bid
        if not (min_px <= m.sum() <= max_px):
            continue
        rows, cols = np.where(m)
        dist = np.hypot(rows.mean() - cy, cols.mean() - cx)
        if dist < best_dist:
            best_dist, best = dist, m

    return (best, "NDVI+nDOM fallback") if best is not None else (None, None)


# ── Public entry point ────────────────────────────────────────────────────────

def detect_roof(ndom_arr, dop_arr, e_min, n_min, tile_key=None,
                merge_alkis=False, merge_gap_m=3.0, max_units=None):
    """
    Detect the single/semi-detached house roof at the tile center.

    Parameters
    ----------
    ndom_arr : (H, W) float32   nDOM — height above ground in metres
    dop_arr  : (4, H_dop, W_dop) uint8  RGBI aerial photo (10 cm/px)
    e_min    : float  UTM32N easting of left tile edge
    n_min    : float  UTM32N northing of bottom tile edge
    tile_key : str   optional key for SAM image-embedding cache
    merge_alkis : bool  merge adjacent ALKIS polygons (Doppelhaus/Reihenhaus)

    Returns
    -------
    mask   : (H, W) bool
    found  : bool
    method : str  — 'ALKIS+SAM', 'ALKIS+GrabCut', 'ALKIS+nDOM',
                    'NDVI+nDOM fallback', 'none'
    """
    mask, method = detect_with_alkis(ndom_arr, dop_arr, e_min, n_min,
                                     tile_key=tile_key,
                                     merge_alkis=merge_alkis,
                                     merge_gap_m=merge_gap_m,
                                     max_units=max_units)
    if mask is not None and mask.sum() > 0:
        return mask, True, method

    mask, method = detect_with_ndvi(ndom_arr, dop_arr)
    if mask is not None and mask.sum() > 0:
        return mask, True, method

    return np.zeros_like(ndom_arr, dtype=bool), False, "none"


def get_alkis_mask(ndom_arr, e_min, n_min, eave_buffer_m=0.5,
                   merge_alkis=False, merge_gap_m=3.0, max_units=None):
    """Return ALKIS building mask at nDOM resolution with eave buffer.

    Covers the full building footprint from cadastral data, independent
    of SAM/nDOM segmentation.  Use this for RoofMapNet line filtering
    and boundary classification.
    """
    h, w = ndom_arr.shape
    e_max = e_min + w * NDOM_RES
    n_max = n_min + h * NDOM_RES

    rings = _query_alkis(e_min, e_max, n_min, n_max)
    if not rings:
        return None

    if merge_alkis:
        best_alkis, _, _ = _merge_alkis_nearby(
            rings, ndom_arr, e_min, n_min, max_gap_m=merge_gap_m,
            max_units=max_units)
    else:
        best_alkis, _, _ = _select_best_alkis(rings, ndom_arr, e_min, n_min)
    if best_alkis is None:
        return None

    buf_px = max(1, int(round(eave_buffer_m / NDOM_RES)))
    return binary_dilation(best_alkis, iterations=buf_px)


def get_alkis_rect(ndom_arr, e_min, n_min, eave_buffer_m=0.5,
                   merge_alkis=False, merge_gap_m=3.0, max_units=None):
    """Return ALKIS-based building rectangle (4 corners in nDOM pixel coords).

    Uses the ALKIS cadastral polygon (+ small eave buffer) to compute a
    minAreaRect.  This is more reliable than deriving the building boundary
    from the nDOM mask, which can be incomplete due to NDVI / SAM issues.

    Returns list of 4 (row, col) tuples or None.
    """
    import cv2
    h, w = ndom_arr.shape
    e_max = e_min + w * NDOM_RES
    n_max = n_min + h * NDOM_RES

    rings = _query_alkis(e_min, e_max, n_min, n_max)
    if not rings:
        return None

    if merge_alkis:
        best_alkis, _, _ = _merge_alkis_nearby(
            rings, ndom_arr, e_min, n_min, max_gap_m=merge_gap_m,
            max_units=max_units)
    else:
        best_alkis, _, _ = _select_best_alkis(rings, ndom_arr, e_min, n_min)
    if best_alkis is None:
        return None

    buf_px = max(1, int(round(eave_buffer_m / NDOM_RES)))
    dilated = binary_dilation(best_alkis, iterations=buf_px)

    u8 = dilated.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return [(float(pt[1]), float(pt[0])) for pt in box]
