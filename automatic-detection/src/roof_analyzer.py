"""
High-precision roof geometry analyzer using RANSAC plane fitting.

Replaces: roof_geometry.py, roof_segmentation.py, feature_extractor.py

Pipeline:
  1. Sequential RANSAC on 3D point cloud from nDOM → plane segmentation
  2. Post-RANSAC merge of over-segmented planes
  3. Rule-based roof type classification from plane geometry
  4. Geometric structural line detection with PCA-based length measurement
  5. nDOM-residual dormer detection + DOP-based skylight detection
  6. Backward-compatible 8-feature extraction for cost model
"""

import numpy as np
import cv2
from scipy.ndimage import (
    gaussian_filter, label as ndi_label,
    binary_dilation, binary_erosion, binary_opening, binary_closing,
    uniform_filter, maximum_filter,
)

NDOM_RES = 0.5
DOP_RES = 0.10
SCALE = int(round(NDOM_RES / DOP_RES))

ROOF_TYPES = {
    'Flachdach':         'Flat roof',
    'Pultdach':          'Mono-pitch / Shed roof',
    'Satteldach':        'Gable roof',
    'Walmdach':          'Hip roof',
    'Krüppelwalmdach':   'Jerkinhead / Clipped gable',
    'Mansarddach':       'Mansard roof',
    'Zeltdach':          'Pyramid / Tent roof',
    'Komplexdach':       'Complex roof',
}


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _angular_diff(a1, a2):
    d = abs(a1 - a2) % 360
    return min(d, 360 - d)


def _fit_plane_lstsq(points):
    """Least-squares plane fit z = a*x + b*y + c. Returns ((a,b,c), normal)."""
    A = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    coeffs, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    return (a, b, c), normal


def _pca_line_length(rows, cols, gsd):
    """2D projected length of roughly-collinear pixels via PCA projection."""
    n = len(rows)
    if n == 0:
        return 0.0
    if n == 1:
        return gsd
    coords = np.column_stack([cols.astype(float), rows.astype(float)]) * gsd
    if n == 2:
        return float(np.linalg.norm(coords[1] - coords[0])) + gsd
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.ndim < 2 or cov.shape != (2, 2):
        return float(n) * gsd
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    proj = centered @ direction
    return float(proj.max() - proj.min()) + gsd


def _pca_line_length_3d(rows, cols, ndom, gsd):
    """3D slope-corrected length via PCA + linear height regression."""
    length_2d = _pca_line_length(rows, cols, gsd)
    if length_2d < gsd * 1.5 or len(rows) < 3:
        return length_2d
    coords = np.column_stack([cols.astype(float), rows.astype(float)]) * gsd
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        return length_2d
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    proj = centered @ direction
    z = ndom[rows, cols].astype(float)
    if np.std(proj) < 1e-6:
        return length_2d
    slope = np.polyfit(proj, z, 1)[0]
    return length_2d * np.sqrt(1 + slope ** 2)


def _measure_segments(line_mask, gsd, ndom=None, use_3d=False):
    """Total length of all connected segments in a line mask."""
    if not line_mask.any():
        return 0.0
    labeled, n = ndi_label(line_mask)
    total = 0.0
    for i in range(1, n + 1):
        seg = labeled == i
        sr, sc = np.where(seg)
        if len(sr) < 2:
            total += gsd
            continue
        if use_3d and ndom is not None:
            total += _pca_line_length_3d(sr, sc, ndom, gsd)
        else:
            total += _pca_line_length(sr, sc, gsd)
    return round(total, 2)


def _pixels_to_segments(line_mask):
    """Convert boolean line mask to geometric line segments via PCA.
    Returns list of (r1, c1, r2, c2) in pixel coordinates."""
    if not line_mask.any():
        return []
    labeled, n = ndi_label(line_mask)
    segments = []
    for i in range(1, n + 1):
        seg = labeled == i
        rows, cols = np.where(seg)
        if len(rows) < 2:
            continue
        coords = np.column_stack([cols.astype(float), rows.astype(float)])
        if len(coords) == 2:
            segments.append((float(rows[0]), float(cols[0]),
                             float(rows[1]), float(cols[1])))
            continue
        mean = coords.mean(axis=0)
        centered = coords - mean
        cov = np.cov(centered.T)
        if cov.ndim < 2:
            continue
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        proj = centered @ direction
        p1 = mean + direction * proj.min()
        p2 = mean + direction * proj.max()
        segments.append((float(p1[1]), float(p1[0]),
                         float(p2[1]), float(p2[0])))
    return segments


def _mask_polygon(mask, epsilon_frac=0.012):
    """Simplified polygon boundary of a boolean mask via approxPolyDP.
    Returns list of (row, col) vertices."""
    u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    eps = epsilon_frac * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    return [(float(pt[0][1]), float(pt[0][0])) for pt in approx]


def _rect_corners(mask, dop_arr=None, gsd=NDOM_RES):
    """Minimum-area oriented rectangle, refined at DOP resolution when available."""
    if dop_arr is not None and dop_arr.shape[0] >= 3:
        result = _rect_corners_hires(mask, dop_arr, gsd)
        if result:
            return result
    u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return [(float(pt[1]), float(pt[0])) for pt in box]


def _rect_corners_hires(mask, dop_arr, gsd):
    """Compute minAreaRect at DOP resolution with erosion + NDVI filtering.
    Erodes the nDOM mask by 1 pixel to remove boundary overshoot,
    upscales to DOP 10cm for smooth contours, applies NDVI vegetation filter."""
    from scipy.ndimage import zoom as ndzoom

    h_dop, w_dop = dop_arr.shape[1], dop_arr.shape[2]
    h_nd, w_nd = mask.shape
    scale = max(1, round(h_dop / h_nd))

    mask_er = binary_erosion(mask, iterations=1)
    if mask_er.sum() < 6:
        mask_er = mask

    md = ndzoom(mask_er.astype(float), scale, order=0) > 0.5
    md = md[:h_dop, :w_dop]
    if md.sum() < 50:
        return []

    if dop_arr.shape[0] >= 4:
        R = dop_arr[0, :h_dop, :w_dop].astype(float)
        NIR = dop_arr[3, :h_dop, :w_dop].astype(float)
        ndvi = (NIR - R) / (NIR + R + 1e-6)
        md = md & (ndvi < 0.2)

    md = binary_opening(md, iterations=1)

    labeled, n = ndi_label(md)
    if n == 0:
        return []
    sizes = [int((labeled == i).sum()) for i in range(1, n + 1)]
    mf = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mf, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return [(float(pt[1]) / scale, float(pt[0]) / scale) for pt in box]


def _classify_rect_edges(rect_verts, ridge_angle):
    """Classify 4 edges of a rectangle as traufe or ortgang.
    Guarantees exactly 2 traufe + 2 ortgang for rectangular buildings."""
    result = {'traufe': [], 'ortgang': []}
    n = len(rect_verts)
    if n < 3:
        return result
    for i in range(n):
        r1, c1 = rect_verts[i]
        r2, c2 = rect_verts[(i + 1) % n]
        edge_angle = float(np.degrees(np.arctan2(r2 - r1, c2 - c1))) % 180
        diff = abs(edge_angle - ridge_angle)
        diff = min(diff, 180 - diff)
        if diff < 45:
            result['traufe'].append((r1, c1, r2, c2))
        else:
            result['ortgang'].append((r1, c1, r2, c2))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# RANSAC PLANE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def segment_planes(mask, ndom, gsd=NDOM_RES, max_planes=8,
                   min_plane_m2=6.0, inlier_thresh=0.15, n_trials=300):
    """
    Sequential RANSAC plane fitting on the 3D point cloud from nDOM.
    Each pixel (r, c) in mask → 3D point (c*gsd, r*gsd, ndom[r,c]).
    Returns (planes_list, plane_labels H×W int).
    """
    rows_all, cols_all = np.where(mask)
    n_pts = len(rows_all)
    if n_pts < 10:
        lbl = np.zeros(mask.shape, dtype=int)
        if n_pts > 0:
            lbl[mask] = 1
            return [_single_plane(mask, ndom, gsd, 1)], lbl
        return [], lbl

    x = cols_all.astype(float) * gsd
    y = rows_all.astype(float) * gsd
    z = ndom[rows_all, cols_all].astype(float)
    points = np.column_stack([x, y, z])

    min_px = max(4, int(min_plane_m2 / gsd ** 2))
    available = np.ones(n_pts, dtype=bool)
    plane_labels = np.zeros(mask.shape, dtype=int)
    planes = []
    lbl_ctr = 0
    rng = np.random.RandomState(42)

    for _ in range(max_planes):
        avail_idx = np.where(available)[0]
        if len(avail_idx) < min_px:
            break

        pts = points[avail_idx]
        best_count = 0
        best_inliers = None

        for _ in range(n_trials):
            si = rng.choice(len(pts), 3, replace=False)
            p1, p2, p3 = pts[si[0]], pts[si[1]], pts[si[2]]
            v1, v2 = p2 - p1, p3 - p1
            normal = np.cross(v1, v2)
            nrm = np.linalg.norm(normal)
            if nrm < 1e-12:
                continue
            normal /= nrm
            if abs(normal[2]) < 0.25:
                continue
            d = -np.dot(normal, p1)
            dist = np.abs(pts @ normal + d)
            inl = dist < inlier_thresh
            count = int(inl.sum())
            if count > best_count:
                best_count = count
                best_inliers = avail_idx[inl]

        if best_inliers is None or len(best_inliers) < min_px:
            break

        (a, b, c), normal = _fit_plane_lstsq(points[best_inliers])
        z_pred = a * points[best_inliers, 0] + b * points[best_inliers, 1] + c
        resid = np.abs(points[best_inliers, 2] - z_pred)
        refined = resid < inlier_thresh * 1.5
        best_inliers = best_inliers[refined]
        if len(best_inliers) < min_px:
            break
        (a, b, c), normal = _fit_plane_lstsq(points[best_inliers])

        inl_rows = rows_all[best_inliers]
        inl_cols = cols_all[best_inliers]
        tmp = np.zeros(mask.shape, dtype=bool)
        tmp[inl_rows, inl_cols] = True
        cc_labeled, n_cc = ndi_label(tmp)

        for ci in range(1, n_cc + 1):
            cmask = cc_labeled == ci
            cr, cc_ = np.where(cmask)
            if len(cr) < min_px:
                continue
            lbl_ctr += 1
            cp = np.column_stack([
                cc_.astype(float) * gsd,
                cr.astype(float) * gsd,
                ndom[cr, cc_].astype(float),
            ])
            if len(cp) >= 4:
                (ca, cb, ccc), cn = _fit_plane_lstsq(cp)
            else:
                (ca, cb, ccc), cn = (a, b, c), normal.copy()

            plane_labels[cr, cc_] = lbl_ctr
            planes.append(_plane_dict(lbl_ctr, cmask, (ca, cb, ccc), cn,
                                      ndom, gsd, cr, cc_))

        available[best_inliers] = False

    _assign_remaining(mask, ndom, gsd, planes, plane_labels)
    planes, plane_labels = _merge_similar(planes, plane_labels, ndom, gsd)
    planes, plane_labels = _filter_annexes(planes, plane_labels, ndom, gsd)

    for p in planes:
        p['mask'] = (plane_labels == p['label'])
        mr, mc = np.where(p['mask'])
        fp = len(mr) * gsd ** 2
        pitch_rad = np.arccos(np.clip(abs(p['normal'][2]), 0, 1))
        p['footprint_m2'] = round(fp, 1)
        p['area_3d_m2'] = round(fp / max(np.cos(pitch_rad), 0.1), 1)
        if len(mr):
            p['centroid_r'] = float(mr.mean())
            p['centroid_c'] = float(mc.mean())

    planes.sort(key=lambda p: p['footprint_m2'], reverse=True)
    return planes, plane_labels


def _plane_dict(label, pmask, coeffs, normal, ndom, gsd, rows, cols):
    a, b, c = coeffs
    pitch_rad = np.arccos(np.clip(abs(normal[2]), 0, 1))
    pitch_deg = float(np.degrees(pitch_rad))
    aspect = float(np.degrees(np.arctan2(-a, b))) % 360
    z = ndom[rows, cols].astype(float)
    fp = len(rows) * gsd ** 2
    return {
        'label': label,
        'mask': pmask,
        'coefficients': coeffs,
        'normal': normal.copy(),
        'pitch_deg': round(pitch_deg, 1),
        'aspect_deg': round(aspect, 0),
        'footprint_m2': round(fp, 1),
        'area_3d_m2': round(fp / max(np.cos(pitch_rad), 0.1), 1),
        'centroid_r': float(rows.mean()),
        'centroid_c': float(cols.mean()),
        'z_mean': float(z.mean()),
        'z_min': float(z.min()),
        'z_max': float(z.max()),
    }


def _single_plane(mask, ndom, gsd, label):
    rows, cols = np.where(mask)
    pts = np.column_stack([
        cols.astype(float) * gsd, rows.astype(float) * gsd,
        ndom[rows, cols].astype(float),
    ])
    if len(pts) >= 4:
        coeffs, normal = _fit_plane_lstsq(pts)
    else:
        z_mean = float(ndom[rows, cols].mean()) if len(rows) else 0.0
        coeffs = (0.0, 0.0, z_mean)
        normal = np.array([0.0, 0.0, 1.0])
    return _plane_dict(label, mask.copy(), coeffs, normal, ndom, gsd, rows, cols)


def _assign_remaining(mask, ndom, gsd, planes, plane_labels):
    if not planes:
        return
    unassigned = mask & (plane_labels == 0)
    if not unassigned.any():
        return
    ur, uc = np.where(unassigned)
    z_vals = ndom[ur, uc].astype(float)
    best_label = np.full(len(ur), planes[0]['label'], dtype=int)
    best_dist = np.full(len(ur), np.inf)
    for p in planes:
        a, b, c = p['coefficients']
        z_pred = a * uc.astype(float) * gsd + b * ur.astype(float) * gsd + c
        dist = np.abs(z_vals - z_pred)
        better = dist < best_dist
        best_dist[better] = dist[better]
        best_label[better] = p['label']
    for i in range(len(ur)):
        plane_labels[ur[i], uc[i]] = best_label[i]


def _filter_annexes(planes, plane_labels, ndom, gsd,
                    min_gap_m=1.5, max_annex_frac=0.35):
    """
    Remove terrace / carport / annex roofs that are significantly lower
    than the main house roof.  A plane is considered an annex if:
      - its z_max is > min_gap_m below the area-weighted z_mean of the
        remaining planes (i.e. even its highest point doesn't reach the
        main roof's average height), AND
      - its footprint is less than max_annex_frac of the total footprint.
    """
    if len(planes) < 2:
        return planes, plane_labels

    total_fp = sum(p['footprint_m2'] for p in planes) or 1.0

    keep = []
    remove_labels = set()
    for p in planes:
        if p['footprint_m2'] / total_fp >= max_annex_frac:
            keep.append(p)
            continue

        others = [q for q in planes if q['label'] != p['label']]
        if not others:
            keep.append(p)
            continue
        other_fp = sum(q['footprint_m2'] for q in others) or 1.0
        other_z_mean = sum(q['z_mean'] * q['footprint_m2'] for q in others) / other_fp

        if p['z_max'] < other_z_mean - min_gap_m:
            remove_labels.add(p['label'])
        else:
            keep.append(p)

    if not remove_labels:
        return planes, plane_labels

    for lbl in remove_labels:
        plane_labels[plane_labels == lbl] = 0

    return keep, plane_labels


def _merge_similar(planes, plane_labels, ndom, gsd,
                   angle_thresh=15.0, height_thresh=0.35):
    """Merge spatially adjacent planes with similar normals."""
    changed = True
    while changed:
        changed = False
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                pi, pj = planes[i], planes[j]
                cos_a = np.clip(np.dot(pi['normal'], pj['normal']), -1, 1)
                angle = np.degrees(np.arccos(abs(cos_a)))
                if angle > angle_thresh:
                    continue
                adj = binary_dilation(pi['mask'], iterations=1) & pj['mask']
                if not adj.any():
                    continue
                br, bc = np.where(adj)
                a, b, c = pi['coefficients']
                z_pred = a * bc.astype(float) * gsd + b * br.astype(float) * gsd + c
                z_actual = ndom[br, bc].astype(float)
                if np.median(np.abs(z_actual - z_pred)) > height_thresh:
                    continue
                merged = pi['mask'] | pj['mask']
                mr, mc = np.where(merged)
                pts = np.column_stack([
                    mc.astype(float) * gsd, mr.astype(float) * gsd,
                    ndom[mr, mc].astype(float),
                ])
                coeffs, normal = _fit_plane_lstsq(pts)
                pitch_rad = np.arccos(np.clip(abs(normal[2]), 0, 1))
                a_new, b_new, c_new = coeffs
                fp = len(mr) * gsd ** 2
                pi['mask'] = merged
                pi['coefficients'] = coeffs
                pi['normal'] = normal
                pi['pitch_deg'] = round(float(np.degrees(pitch_rad)), 1)
                pi['aspect_deg'] = round(float(np.degrees(np.arctan2(-a_new, b_new))) % 360, 0)
                pi['footprint_m2'] = round(fp, 1)
                pi['area_3d_m2'] = round(fp / max(np.cos(pitch_rad), 0.1), 1)
                pi['centroid_r'] = float(mr.mean())
                pi['centroid_c'] = float(mc.mean())
                pi['z_mean'] = float(ndom[mr, mc].mean())
                pi['z_min'] = float(ndom[mr, mc].min())
                pi['z_max'] = float(ndom[mr, mc].max())
                plane_labels[plane_labels == pj['label']] = pi['label']
                planes.pop(j)
                changed = True
                break
            if changed:
                break
    return planes, plane_labels


# ══════════════════════════════════════════════════════════════════════════════
# RIDGE-GUIDED PLANE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def segment_planes_guided(mask, ndom, ridge_lines_ndom, gsd=NDOM_RES,
                          min_plane_m2=6.0, line_width_px=1):
    """Segment roof planes using RoofMapNet ridge lines as cutting boundaries.

    Instead of blind RANSAC, draws ridge lines as barriers on the mask,
    then labels connected components. Each region gets its own plane fit.

    Parameters
    ----------
    mask : (H, W) bool — roof mask.
    ndom : (H, W) float — height model.
    ridge_lines_ndom : list of (2, 2) arrays — ridge lines in nDOM pixel coords.
    gsd : float — ground sampling distance.
    min_plane_m2 : float — minimum plane area to keep.
    line_width_px : int — width of cutting lines in pixels.

    Returns
    -------
    (planes_list, plane_labels) same format as segment_planes().
    """
    h, w = mask.shape
    cut_mask = mask.copy()

    for ln in ridge_lines_ndom:
        r1 = int(np.clip(round(ln[0, 0]), 0, h - 1))
        c1 = int(np.clip(round(ln[0, 1]), 0, w - 1))
        r2 = int(np.clip(round(ln[1, 0]), 0, h - 1))
        c2 = int(np.clip(round(ln[1, 1]), 0, w - 1))
        cv2.line(cut_mask.astype(np.uint8), (c1, r1), (c2, r2), 0, line_width_px)
        cut_mask_tmp = cut_mask.astype(np.uint8)
        cv2.line(cut_mask_tmp, (c1, r1), (c2, r2), 0, line_width_px)
        cut_mask = cut_mask_tmp.astype(bool)

    labeled, n_regions = ndi_label(cut_mask)

    min_px = max(4, int(min_plane_m2 / gsd ** 2))
    plane_labels = np.zeros((h, w), dtype=int)
    planes = []
    lbl_ctr = 0

    for reg_id in range(1, n_regions + 1):
        region = labeled == reg_id
        if region.sum() < min_px:
            continue
        lbl_ctr += 1
        rows, cols = np.where(region)
        pts = np.column_stack([
            cols.astype(float) * gsd,
            rows.astype(float) * gsd,
            ndom[rows, cols].astype(float),
        ])
        if len(pts) >= 4:
            coeffs, normal = _fit_plane_lstsq(pts)
        else:
            z_mean = float(ndom[rows, cols].mean())
            coeffs = (0.0, 0.0, z_mean)
            normal = np.array([0.0, 0.0, 1.0])

        plane_labels[rows, cols] = lbl_ctr
        planes.append(_plane_dict(lbl_ctr, region, coeffs, normal,
                                  ndom, gsd, rows, cols))

    # Assign boundary/cut pixels to the nearest plane
    _assign_remaining(mask, ndom, gsd, planes, plane_labels)

    # Merge planes with very similar normals that touch
    planes, plane_labels = _merge_similar(planes, plane_labels, ndom, gsd)
    planes, plane_labels = _filter_annexes(planes, plane_labels, ndom, gsd)

    for p in planes:
        p['mask'] = (plane_labels == p['label'])
        mr, mc = np.where(p['mask'])
        fp = len(mr) * gsd ** 2
        pitch_rad = np.arccos(np.clip(abs(p['normal'][2]), 0, 1))
        p['footprint_m2'] = round(fp, 1)
        p['area_3d_m2'] = round(fp / max(np.cos(pitch_rad), 0.1), 1)
        if len(mr):
            p['centroid_r'] = float(mr.mean())
            p['centroid_c'] = float(mc.mean())

    planes.sort(key=lambda p: p['footprint_m2'], reverse=True)
    return planes, plane_labels


# ══════════════════════════════════════════════════════════════════════════════
# ROOF TYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_roof_type(planes, ridge_length, total_footprint):
    """
    Classify the roof type from plane geometry.
    Returns (german_name, english_name).
    """
    n = len(planes)
    if n == 0:
        return 'Unbekannt', 'Unknown'

    pitches = np.array([p['pitch_deg'] for p in planes])
    aspects = np.array([p['aspect_deg'] for p in planes])
    areas = np.array([p['footprint_m2'] for p in planes])
    total = float(areas.sum()) or 1.0

    if n == 1:
        if pitches[0] < 10:
            return 'Flachdach', 'Flat roof'
        if pitches[0] > 20 and ridge_length > 0.5:
            return 'Satteldach', 'Gable roof'
        return 'Pultdach', 'Mono-pitch / Shed roof'

    if n == 2:
        ad = _angular_diff(aspects[0], aspects[1])
        pd = abs(pitches[0] - pitches[1])
        if ad < 40 and pd > 15:
            return 'Mansarddach', 'Mansard roof'
        return 'Satteldach', 'Gable roof'

    # n >= 3: check for offset Satteldach (2 aspect groups facing opposite)
    if n >= 3:
        used = [False] * n
        groups = []
        for i in range(n):
            if used[i]:
                continue
            g = [i]
            for j in range(i + 1, n):
                if not used[j] and _angular_diff(aspects[i], aspects[j]) < 35:
                    g.append(j)
                    used[j] = True
            groups.append(g)

        if len(groups) == 2:
            a1 = aspects[groups[0][0]]
            a2 = aspects[groups[1][0]]
            if _angular_diff(a1, a2) > 100:
                for g in groups:
                    if len(g) >= 2:
                        g_pitches = pitches[g]
                        if g_pitches.max() - g_pitches.min() > 15:
                            return 'Mansarddach', 'Mansard roof'
                return 'Satteldach', 'Gable roof'

    if n == 3:
        sorted_a = np.sort(areas)
        if sorted_a[0] / total < 0.12:
            return 'Krüppelwalmdach', 'Jerkinhead / Clipped gable'
        return 'Komplexdach', 'Complex roof'

    if n == 4:
        for i in range(4):
            for j in range(i + 1, 4):
                ad = _angular_diff(aspects[i], aspects[j])
                pd = abs(pitches[i] - pitches[j])
                if ad < 35 and pd > 15:
                    return 'Mansarddach', 'Mansard roof'

        sorted_a = np.sort(areas)
        area_ratios = sorted_a / total
        cv_area = float(np.std(areas) / np.mean(areas))

        if cv_area < 0.25:
            ridge_ratio = ridge_length / max(np.sqrt(total_footprint), 1.0)
            if ridge_ratio < 0.12:
                return 'Zeltdach', 'Pyramid / Tent roof'

        if area_ratios[0] + area_ratios[1] < 0.28:
            return 'Krüppelwalmdach', 'Jerkinhead / Clipped gable'

        return 'Walmdach', 'Hip roof'

    return 'Komplexdach', 'Complex roof'


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL LINE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_structural_lines(planes, plane_labels, mask, ndom, gsd=NDOM_RES):
    """
    Detect and classify roof structural lines.
    Internal: ridge (First), valley (Kehle), hip (Grat) — from plane adjacency.
    External: eave (Traufe), gable end (Ortgang) — from mask boundary.
    Returns (lines_dict, lengths_dict, ridge_angle_deg).
    """
    h, w = mask.shape
    n_planes = len(planes)
    smooth = gaussian_filter(ndom.astype(float), sigma=1.0)

    is_ridge = np.zeros((h, w), dtype=bool)
    is_valley = np.zeros((h, w), dtype=bool)
    is_grat = np.zeros((h, w), dtype=bool)

    # ── Internal plane boundaries ──
    boundary_map = {}
    padded = np.pad(plane_labels, 1, mode='constant', constant_values=0)
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        shifted = padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        diff_mask = (shifted != plane_labels) & (plane_labels > 0) & (shifted > 0)
        if not diff_mask.any():
            continue
        dr_arr, dc_arr = np.where(diff_mask)
        for ri in range(len(dr_arr)):
            r, c = dr_arr[ri], dc_arr[ri]
            l1 = plane_labels[r, c]
            l2 = shifted[r, c]
            key = (min(l1, l2), max(l1, l2))
            if key not in boundary_map:
                boundary_map[key] = np.zeros((h, w), dtype=bool)
            boundary_map[key][r, c] = True

    plane_by_label = {p['label']: p for p in planes}

    for (l1, l2), bmask in boundary_map.items():
        br, bc = np.where(bmask)
        if len(br) == 0:
            continue

        p1 = plane_by_label.get(l1)
        p2 = plane_by_label.get(l2)
        if p1 is None or p2 is None:
            is_grat[br, bc] = True
            continue

        if _angular_diff(p1['aspect_deg'], p2['aspect_deg']) < 35:
            continue

        a1, b1, _ = p1['coefficients']
        a2, b2, _ = p2['coefficients']

        perp = np.array([
            p2['centroid_c'] - p1['centroid_c'],
            p2['centroid_r'] - p1['centroid_r'],
        ])
        pnorm = np.linalg.norm(perp)
        if pnorm < 1e-6:
            is_grat[br, bc] = True
            continue
        perp /= pnorm

        d1 = a1 * perp[0] + b1 * perp[1]
        d2 = a2 * perp[0] + b2 * perp[1]

        min_grad = 0.02
        if d1 > min_grad and d2 < -min_grad:
            is_ridge[br, bc] = True
        elif d1 < -min_grad and d2 > min_grad:
            is_valley[br, bc] = True
        elif n_planes <= 2:
            z_boundary = float(np.mean(smooth[br, bc]))
            near1 = binary_dilation(bmask, iterations=2) & (plane_labels == l1) & ~bmask
            near2 = binary_dilation(bmask, iterations=2) & (plane_labels == l2) & ~bmask
            z1 = float(np.mean(smooth[near1])) if near1.any() else z_boundary
            z2 = float(np.mean(smooth[near2])) if near2.any() else z_boundary
            if z_boundary >= max(z1, z2) - 0.1:
                is_ridge[br, bc] = True
            else:
                is_grat[br, bc] = True
        else:
            is_grat[br, bc] = True

    # ── Height-based ridge fallback ──
    # If plane-boundary detection found no ridge but the roof is pitched,
    # detect the ridge from nDOM height maxima (top ~10% of height values).
    max_pitch = max((p['pitch_deg'] for p in planes), default=0)
    if not is_ridge.any() and max_pitch > 12:
        is_ridge = _detect_ridge_from_height(mask, ndom, gsd)

    # ── Ridge direction ──
    ridge_angle = _compute_ridge_angle(is_ridge, planes, mask, ndom)

    # ── External boundary: Traufe vs Ortgang ──
    is_traufe = np.zeros((h, w), dtype=bool)
    is_ortgang = np.zeros((h, w), dtype=bool)

    contour_mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=len)
        pts = contour.reshape(-1, 2)
        n_pts = len(pts)
        if n_pts > 4:
            window = max(2, min(5, n_pts // 10))
            for i in range(n_pts):
                c, r = int(pts[i][0]), int(pts[i][1])
                if not (0 <= r < h and 0 <= c < w):
                    continue
                p_prev = pts[(i - window) % n_pts]
                p_next = pts[(i + window) % n_pts]
                tang = (p_next - p_prev).astype(float)
                tnorm = np.linalg.norm(tang)
                if tnorm < 1e-6:
                    continue
                tang_angle = float(np.degrees(np.arctan2(tang[1], tang[0]))) % 180
                diff = abs(tang_angle - ridge_angle)
                diff = min(diff, 180 - diff)
                if diff < 45:
                    is_traufe[r, c] = True
                else:
                    is_ortgang[r, c] = True

    # ── Lengths ──
    lines = {
        'ridge': is_ridge, 'valley': is_valley, 'grat': is_grat,
        'traufe': is_traufe, 'ortgang': is_ortgang,
    }
    lengths = {
        'first_m': _measure_segments(is_ridge, gsd),
        'kehle_m': _measure_segments(is_valley, gsd, ndom, use_3d=True),
        'grat_m': _measure_segments(is_grat, gsd, ndom, use_3d=True),
        'traufe_m': _measure_segments(is_traufe, gsd),
        'ortgang_m': _measure_segments(is_ortgang, gsd),
    }
    return lines, lengths, ridge_angle


def _detect_ridge_from_height(mask, ndom, gsd):
    """Detect ridge line from nDOM height maxima when plane-boundary fails."""
    vals = ndom[mask]
    if vals.size < 10:
        return np.zeros(mask.shape, dtype=bool)

    thresh = np.percentile(vals, 85)
    ridge_mask = mask & (ndom >= thresh)

    if ridge_mask.sum() > 40:
        ridge_mask = binary_erosion(ridge_mask, iterations=1)
        ridge_mask = binary_dilation(ridge_mask, iterations=1) & mask

    labeled, n = ndi_label(ridge_mask)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    ridge_mask = labeled == (np.argmax(sizes) + 1)

    rr, rc = np.where(ridge_mask)
    if len(rr) < 3:
        return ridge_mask

    coords = np.column_stack([rc.astype(float), rr.astype(float)])
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        return ridge_mask

    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    proj = centered @ direction

    perp = eigvecs[:, np.argmin(eigvals)]
    perp_proj = np.abs(centered @ perp)
    thin = perp_proj < max(1.0, np.percentile(perp_proj, 30))

    result = np.zeros(mask.shape, dtype=bool)
    thin_rc = coords[thin].astype(int)
    for pt in thin_rc:
        c, r = pt[0], pt[1]
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
            result[r, c] = True

    return result


def _compute_ridge_angle(is_ridge, planes, mask, ndom):
    """Compute ridge direction in degrees (0-180)."""
    rr, rc = np.where(is_ridge)
    if len(rr) >= 5:
        coords = np.column_stack([rc.astype(float), rr.astype(float)])
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        if cov.ndim == 2:
            eigvals, eigvecs = np.linalg.eigh(cov)
            vec = eigvecs[:, np.argmax(eigvals)]
            return float(np.degrees(np.arctan2(vec[1], vec[0]))) % 180

    if len(planes) >= 2:
        p1, p2 = planes[0], planes[1]
        a1, b1, _ = p1['coefficients']
        a2, b2, _ = p2['coefficients']
        da, db = a1 - a2, b1 - b2
        if abs(da) > 1e-8 or abs(db) > 1e-8:
            direction = np.array([-db, da])
            return float(np.degrees(np.arctan2(direction[1], direction[0]))) % 180

    vals = ndom[mask]
    if vals.size < 10:
        return 0.0
    thresh = np.percentile(vals, 88)
    rpm = mask & (ndom >= thresh)
    rs, cs = np.where(rpm)
    if len(rs) < 4:
        return 0.0
    pts = np.column_stack([cs.astype(float) - cs.mean(), rs.astype(float) - rs.mean()])
    cov = np.cov(pts.T)
    if cov.ndim < 2:
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, np.argmax(eigvals)]
    return float(np.degrees(np.arctan2(vec[1], vec[0]))) % 180


# ══════════════════════════════════════════════════════════════════════════════
# RESIDUALS & DORMER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_residuals(mask, ndom, planes, gsd=NDOM_RES):
    """Height residuals above/below fitted planes. Returns (H,W) float array."""
    residual = np.full(ndom.shape, np.nan, dtype=float)
    for p in planes:
        pm = p['mask']
        rows, cols = np.where(pm)
        if len(rows) == 0:
            continue
        a, b, c = p['coefficients']
        z_pred = a * cols.astype(float) * gsd + b * rows.astype(float) * gsd + c
        z_actual = ndom[rows, cols].astype(float)
        res = z_actual - z_pred

        clean = res < 0.4
        if clean.sum() >= 4:
            A2 = np.column_stack([
                cols[clean].astype(float) * gsd,
                rows[clean].astype(float) * gsd,
                np.ones(clean.sum()),
            ])
            coeff2, *_ = np.linalg.lstsq(A2, z_actual[clean], rcond=None)
            a2, b2, c2 = float(coeff2[0]), float(coeff2[1]), float(coeff2[2])
            z_pred = a2 * cols.astype(float) * gsd + b2 * rows.astype(float) * gsd + c2
            res = z_actual - z_pred
            p['coefficients'] = (a2, b2, c2)

        residual[rows, cols] = res

    unassigned = mask & np.isnan(residual)
    residual[unassigned] = 0.0
    residual[~mask] = np.nan
    return residual


def detect_dormers(mask, ndom, residual_map, dop_arr=None, gsd=NDOM_RES):
    """Detect dormers from nDOM residuals with optional DOP edge refinement."""
    if residual_map is None:
        return []
    res = np.where(np.isnan(residual_map), 0.0, residual_map)
    protrusion = mask & (res > 0.2)
    protrusion = binary_opening(protrusion, iterations=1)
    protrusion = binary_closing(protrusion, iterations=1)

    labeled, n = ndi_label(protrusion)
    if n == 0:
        return []

    min_px = max(1, int(0.5 / gsd ** 2))
    max_px = int(40.0 / gsd ** 2)

    dormers = []
    for bid in range(1, n + 1):
        dm = labeled == bid
        npx = int(dm.sum())
        if not (min_px <= npx <= max_px):
            continue

        if dop_arr is not None and dop_arr.shape[0] >= 3:
            dm = _refine_dormer_cv(dm, dop_arr, mask, gsd, npx, max_px)

        pr, pc = np.where(dm)
        if len(pr) == 0:
            continue
        footprint = float(len(pr)) * gsd ** 2
        mean_h = float(res[dm & (res > 0)].mean()) if (dm & (res > 0)).any() else 0.0
        max_h = float(res[dm & (res > 0)].max()) if (dm & (res > 0)).any() else 0.0
        r_span = float(pr.max() - pr.min() + 1) * gsd
        c_span = float(pc.max() - pc.min() + 1) * gsd
        width = round(min(r_span, c_span), 1)
        depth = round(max(r_span, c_span), 1)
        orient = 'N-S' if r_span >= c_span else 'E-W'

        dormers.append({
            'mask': dm,
            'centroid_r': float(pr.mean()),
            'centroid_c': float(pc.mean()),
            'footprint_m2': round(footprint, 1),
            'height_above_plane': round(mean_h, 2),
            'max_height_m': round(max_h, 2),
            'width_m': width,
            'depth_m': depth,
            'orientation': orient,
        })

    dormers.sort(key=lambda d: d['footprint_m2'], reverse=True)
    return dormers


def _refine_dormer_cv(dm_ndom, dop_arr, mask, gsd, orig_npx, max_px):
    """Refine dormer boundary using Canny edge detection on DOP."""
    from scipy.ndimage import zoom as ndzoom
    h_nd, w_nd = mask.shape
    h_dop, w_dop = dop_arr.shape[1], dop_arr.shape[2]
    scale = max(1, round(h_dop / h_nd))

    pr, pc = np.where(dm_ndom)
    pad = 4
    r0 = max(0, pr.min() - pad)
    r1 = min(h_nd, pr.max() + pad + 1)
    c0 = max(0, pc.min() - pad)
    c1 = min(w_nd, pc.max() + pad + 1)

    r0d, r1d = r0 * scale, min(h_dop, r1 * scale)
    c0d, c1d = c0 * scale, min(w_dop, c1 * scale)
    if r1d - r0d < 5 or c1d - c0d < 5:
        return dm_ndom

    rgb_patch = np.transpose(dop_arr[:3, r0d:r1d, c0d:c1d], (1, 2, 0))
    gray = cv2.cvtColor(rgb_patch.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return dm_ndom

    dm_dop = ndzoom(dm_ndom[r0:r1, c0:c1].astype(float), scale, order=0) > 0.5
    dm_dop = dm_dop[:r1d - r0d, :c1d - c0d]

    best_score = 0.2
    best_cnt = None
    for cnt in contours:
        cnt_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        cnt_bool = cnt_mask > 0
        area_m2 = float(cnt_bool.sum()) * DOP_RES ** 2
        if area_m2 < 0.3 or area_m2 > 50:
            continue
        overlap = float((cnt_bool & dm_dop).sum()) / max(1.0, float(dm_dop.sum()))
        if overlap > best_score:
            best_score = overlap
            best_cnt = cnt_bool

    if best_cnt is None:
        return dm_ndom

    refined_full = np.zeros((h_dop, w_dop), dtype=bool)
    refined_full[r0d:r1d, c0d:c1d] = best_cnt
    refined_nd = ndzoom(refined_full.astype(float), 1.0 / scale, order=1)[:h_nd, :w_nd] > 0.3
    refined_nd = refined_nd & mask

    if refined_nd.sum() < 2 or refined_nd.sum() > max_px * 1.5:
        return dm_ndom
    return refined_nd


# ══════════════════════════════════════════════════════════════════════════════
# SKYLIGHT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_skylights(mask, ndom, dop_arr, gsd=NDOM_RES, dormer_mask=None):
    """Detect flat roof windows via NDVI + color contrast on the DOP."""
    if dop_arr is None or dop_arr.shape[0] < 3:
        return []

    from scipy.ndimage import zoom as ndzoom
    h_nd, w_nd = mask.shape
    h_dop, w_dop = dop_arr.shape[1], dop_arr.shape[2]
    scale = int(round(gsd / DOP_RES))

    mask_dop = ndzoom(mask.astype(float), scale, order=0) > 0.5
    mask_dop = mask_dop[:h_dop, :w_dop]
    inner = binary_erosion(mask_dop, iterations=3) if mask_dop.any() else mask_dop

    if dormer_mask is not None and dormer_mask.any():
        dorm_dop = ndzoom(dormer_mask.astype(float), scale, order=0) > 0.5
        dorm_dop = dorm_dop[:h_dop, :w_dop]
        dorm_dop = binary_dilation(dorm_dop, iterations=scale + 2)
    else:
        dorm_dop = np.zeros((h_dop, w_dop), bool)

    search = inner & ~dorm_dop
    if not search.any():
        return []

    candidate = np.zeros((h_dop, w_dop), dtype=bool)

    if dop_arr.shape[0] >= 4:
        R = dop_arr[0].astype(float)
        NIR = dop_arr[3].astype(float)
        ndvi = (NIR - R) / (NIR + R + 1e-6)
        candidate |= search & (ndvi[:h_dop, :w_dop] < -0.02)

    rgb = dop_arr[:3].astype(float)
    brightness = rgb.mean(axis=0)[:h_dop, :w_dop]
    local_mean = uniform_filter(brightness, size=int(2.0 / DOP_RES))
    local_range = uniform_filter(np.abs(brightness - local_mean), size=int(1.5 / DOP_RES))
    bright_spot = search & (brightness > local_mean + 2.0 * local_range + 8)
    candidate |= bright_spot

    gray_roi = brightness.copy()
    gray_roi[~search] = 0
    roof_med = float(np.median(brightness[search]))
    dark_spot = search & (brightness < roof_med * 0.65)
    dark_clean = binary_opening(dark_spot, iterations=1)
    candidate |= dark_clean

    labeled, n = ndi_label(candidate)
    min_px = max(1, int(0.15 / DOP_RES ** 2))
    max_px = int(6.0 / DOP_RES ** 2)

    skylights = []
    for bid in range(1, n + 1):
        m = labeled == bid
        npx = int(m.sum())
        if not (min_px <= npx <= max_px):
            continue
        area_m2 = npx * DOP_RES ** 2
        rs, cs = np.where(m)

        r_span = (rs.max() - rs.min() + 1) * DOP_RES
        c_span = (cs.max() - cs.min() + 1) * DOP_RES
        short_side = min(r_span, c_span)
        long_side = max(r_span, c_span)
        if long_side > 0 and short_side / long_side < 0.25:
            continue
        bbox_area = r_span * c_span
        if bbox_area > 0 and area_m2 / bbox_area < 0.4:
            continue

        skylights.append({
            'row': float(rs.mean() / scale),
            'col': float(cs.mean() / scale),
            'area_m2': round(area_m2, 2),
            'width_m': round(short_side, 2),
            'height_m': round(long_side, 2),
        })

    return skylights[:12]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (backward-compatible with old feature_extractor.py)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(mask, dsm, gsd=NDOM_RES):
    """
    8-feature dict for the cost model. Backward-compatible interface.
    Uses RANSAC internally for plane count and pitch computation.
    """
    n_pixels = int(mask.sum())
    zero = {k: 0.0 for k in [
        'footprint_area_m2', 'dsm_area_m2', 'mean_pitch_deg',
        'pitch_std_deg', 'n_planes', 'complexity_index',
        'ridge_density', 'ndsm_range_m',
    ]}
    if n_pixels == 0:
        return zero

    planes, _ = segment_planes(mask, dsm, gsd)
    if not planes:
        return zero

    footprint = sum(p['footprint_m2'] for p in planes)
    dsm_area = sum(p['area_3d_m2'] for p in planes)

    total_fp = sum(p['footprint_m2'] for p in planes) or 1.0
    mean_pitch = sum(p['pitch_deg'] * p['footprint_m2'] for p in planes) / total_fp
    if len(planes) > 1:
        pitch_std = float(np.std([p['pitch_deg'] for p in planes]))
    else:
        pitch_std = 0.0

    n_planes = len(planes)
    dsm_values = dsm[mask]
    ndsm_range = float(dsm_values.max() - dsm_values.min()) if len(dsm_values) else 0.0

    complexity = float(np.clip(
        0.4 * (pitch_std / 20.0) + 0.4 * (n_planes / 6.0) + 0.2 * (ndsm_range / 5.0),
        0, 1,
    ))

    smoothed = gaussian_filter(np.where(mask, dsm, 0).astype(float), sigma=2)
    local_max = (smoothed == maximum_filter(smoothed, size=5)) & mask
    ridge_density = float(local_max.sum()) / float(mask.sum()) if mask.sum() > 0 else 0.0

    return {
        'footprint_area_m2': round(footprint, 2),
        'dsm_area_m2': round(dsm_area, 2),
        'mean_pitch_deg': round(mean_pitch, 2),
        'pitch_std_deg': round(pitch_std, 2),
        'n_planes': n_planes,
        'complexity_index': round(complexity, 4),
        'ridge_density': round(ridge_density, 4),
        'ndsm_range_m': round(ndsm_range, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_roof(mask, ndom, dop_arr=None, gsd=NDOM_RES, **kwargs):
    """
    Full roof geometry analysis.

    Optional kwargs:
        rmn_lines : (N, 2, 2) — RoofMapNet lines in 128×128 feature coords.
        rmn_scores : (N,) — RoofMapNet confidence scores.
        alkis_rect : list of 4 (row, col) corners.

    When rmn_lines/rmn_scores are provided, tries ridge-guided segmentation
    first (uses RoofMapNet ridges as cutting boundaries). Falls back to blind
    RANSAC if guided segmentation produces fewer planes.
    """
    h, w = mask.shape
    empty_lines = {k: np.zeros((h, w), dtype=bool)
                   for k in ('ridge', 'valley', 'grat', 'traufe', 'ortgang')}
    empty_lengths = {k: 0.0 for k in ('first_m', 'kehle_m', 'grat_m', 'traufe_m', 'ortgang_m')}
    empty_segs = {k: [] for k in ('ridge', 'valley', 'grat', 'traufe', 'ortgang')}
    empty = {
        'planes': [], 'dormers': [], 'windows': [], 'skylights': [],
        'lines': empty_lines, 'lengths': empty_lengths,
        'line_segments': empty_segs, 'plane_rects': [], 'roof_rect': [],
        'residual_map': None,
        'plane_labels': np.zeros((h, w), dtype=int),
        'n_planes': 0, 'total_3d_m2': 0.0, 'total_fp_m2': 0.0,
        'roof_type': ('Unbekannt', 'Unknown'), 'ridge_angle': 0.0,
    }
    if mask.sum() == 0:
        return empty

    rmn_lines = kwargs.get('rmn_lines')
    rmn_scores = kwargs.get('rmn_scores')

    planes_blind, labels_blind = segment_planes(mask, ndom, gsd)

    planes, plane_labels = planes_blind, labels_blind
    used_guided = False

    if rmn_lines is not None and rmn_scores is not None and len(rmn_lines) > 0:
        from wireframe_fusion import find_all_ridges
        ridge_lines = find_all_ridges(rmn_lines, rmn_scores, ndom, mask, gsd)
        if ridge_lines:
            planes_g, labels_g = segment_planes_guided(
                mask, ndom, ridge_lines, gsd=gsd)
            if len(planes_g) >= len(planes_blind) and len(planes_g) >= 2:
                planes, plane_labels = planes_g, labels_g
                used_guided = True

    if not planes:
        return empty

    # Restrict mask to main roof only (exclude filtered annexes/terraces)
    mask = plane_labels > 0

    residual_map = compute_residuals(mask, ndom, planes, gsd)
    dormers = detect_dormers(mask, ndom, residual_map, dop_arr, gsd)
    dormers = [d for d in dormers if d['footprint_m2'] <= 10.0]

    dormer_mask = np.zeros(mask.shape, bool)
    for d in dormers:
        dormer_mask |= d['mask']

    skylights = detect_skylights(mask, ndom, dop_arr, gsd, dormer_mask)
    lines, lengths, ridge_angle = detect_structural_lines(
        planes, plane_labels, mask, ndom, gsd)

    total_fp = sum(p['footprint_m2'] for p in planes)
    roof_type = classify_roof_type(planes, lengths.get('first_m', 0), total_fp)

    # Geometric line segments: prefer ALKIS rect over mask-derived rect
    alkis_rect = kwargs.get('alkis_rect')
    roof_rect = alkis_rect if alkis_rect else _rect_corners(mask, dop_arr, gsd)
    edge_segs = _classify_rect_edges(roof_rect, ridge_angle)
    line_segments = {
        'ridge': _pixels_to_segments(lines['ridge']),
        'valley': _pixels_to_segments(lines['valley']),
        'grat': _pixels_to_segments(lines['grat']),
        'traufe': edge_segs['traufe'],
        'ortgang': edge_segs['ortgang'],
    }

    def _seg_len(segs):
        return round(sum(np.hypot(r2 - r1, c2 - c1) * gsd
                         for r1, c1, r2, c2 in segs), 2)
    lengths['traufe_m'] = _seg_len(edge_segs['traufe'])
    lengths['ortgang_m'] = _seg_len(edge_segs['ortgang'])

    plane_rects = [_rect_corners(plane_labels == p['label'], dop_arr, gsd)
                   for p in planes]

    return {
        'planes': planes,
        'plane_labels': plane_labels,
        'dormers': dormers,
        'windows': skylights,
        'skylights': skylights,
        'lines': lines,
        'lengths': lengths,
        'line_segments': line_segments,
        'plane_rects': plane_rects,
        'roof_rect': roof_rect,
        'residual_map': residual_map,
        'n_planes': len(planes),
        'total_3d_m2': round(sum(p['area_3d_m2'] for p in planes), 1),
        'total_fp_m2': round(total_fp, 1),
        'roof_type': roof_type,
        'ridge_angle': ridge_angle,
        'used_guided': used_guided,
    }
