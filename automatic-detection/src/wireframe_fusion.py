"""
Wireframe–nDOM fusion: classify RoofMapNet lines using height data.

Takes filtered RoofMapNet wireframe lines and classifies each one
as First/Traufe/Ortgang/Kehle/Grat using the nDOM height profile
and RANSAC plane labels.

Usage:
    from wireframe_fusion import fuse_lines

    result = fuse_lines(
        rmn_lines,          # (N, 2, 2) in 128×128 feature coords
        rmn_scores,         # (N,)
        ndom,               # (H, W) float — height above ground
        mask,               # (H, W) bool — roof mask
        plane_labels,       # (H, W) int — RANSAC plane labels
        planes,             # list of plane dicts from roof_analyzer
        ridge_angle,        # float — ridge direction in degrees
        gsd=0.5,
    )
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

FEAT_SIZE = 128


def _to_ndom_coords(lines_feat, ndom_shape):
    """Convert lines from 128×128 feature space to nDOM pixel space."""
    h, w = ndom_shape
    scale_r = h / FEAT_SIZE
    scale_c = w / FEAT_SIZE
    out = lines_feat.copy().astype(float)
    out[:, :, 0] *= scale_r
    out[:, :, 1] *= scale_c
    return out


def _sample_along_line(r1, c1, r2, c2, n_samples=20):
    """Sample n evenly spaced points along a line segment.
    Returns (rows, cols) arrays."""
    t = np.linspace(0, 1, n_samples)
    rows = r1 + t * (r2 - r1)
    cols = c1 + t * (c2 - c1)
    return rows, cols


def _line_length_m(line_ndom, gsd):
    """Compute physical length of a line in meters."""
    dr = (line_ndom[1, 0] - line_ndom[0, 0]) * gsd
    dc = (line_ndom[1, 1] - line_ndom[0, 1]) * gsd
    return float(np.hypot(dr, dc))


def _line_angle_deg(line_ndom):
    """Angle of line in degrees 0–180."""
    dr = line_ndom[1, 0] - line_ndom[0, 0]
    dc = line_ndom[1, 1] - line_ndom[0, 1]
    return float(np.degrees(np.arctan2(dr, dc))) % 180


def _angular_diff(a1, a2):
    d = abs(a1 - a2) % 180
    return min(d, 180 - d)


def _fraction_on_mask_edge(line_ndom, mask, edge_width=2):
    """Fraction of sampled line points that lie on the mask boundary."""
    h, w = mask.shape
    interior = binary_erosion(mask, iterations=edge_width)
    edge_zone = mask & ~interior

    rows, cols = _sample_along_line(
        line_ndom[0, 0], line_ndom[0, 1],
        line_ndom[1, 0], line_ndom[1, 1],
        n_samples=20,
    )
    ri = np.clip(np.round(rows).astype(int), 0, h - 1)
    ci = np.clip(np.round(cols).astype(int), 0, w - 1)

    on_edge = edge_zone[ri, ci].sum()
    in_mask = mask[ri, ci].sum()
    if in_mask == 0:
        return 0.0
    return float(on_edge) / float(in_mask)


def _height_profile(line_ndom, ndom, mask, n_samples=20):
    """Sample nDOM height along a line. Returns (heights, valid_mask)."""
    h, w = ndom.shape
    rows, cols = _sample_along_line(
        line_ndom[0, 0], line_ndom[0, 1],
        line_ndom[1, 0], line_ndom[1, 1],
        n_samples,
    )
    ri = np.clip(np.round(rows).astype(int), 0, h - 1)
    ci = np.clip(np.round(cols).astype(int), 0, w - 1)
    valid = mask[ri, ci]
    heights = ndom[ri, ci].astype(float)
    return heights, valid


def _perpendicular_height_comparison(line_ndom, ndom, mask, gsd, offset_m=1.5):
    """Compare height along the line vs. height offset_m to each side.

    Returns (mean_diff_left, mean_diff_right):
      positive = line is HIGHER than the offset side (ridge-like)
      negative = line is LOWER (valley-like)
    """
    h, w = ndom.shape
    dr = line_ndom[1, 0] - line_ndom[0, 0]
    dc = line_ndom[1, 1] - line_ndom[0, 1]
    length = np.hypot(dr, dc)
    if length < 1e-6:
        return 0.0, 0.0

    perp_r = -dc / length
    perp_c = dr / length
    offset_px = offset_m / gsd

    rows, cols = _sample_along_line(
        line_ndom[0, 0], line_ndom[0, 1],
        line_ndom[1, 0], line_ndom[1, 1],
        n_samples=20,
    )

    diffs_left = []
    diffs_right = []
    for r, c in zip(rows, cols):
        ri = int(np.clip(round(r), 0, h - 1))
        ci = int(np.clip(round(c), 0, w - 1))
        if not mask[ri, ci]:
            continue
        z_center = float(ndom[ri, ci])

        rl = int(np.clip(round(r + perp_r * offset_px), 0, h - 1))
        cl = int(np.clip(round(c + perp_c * offset_px), 0, w - 1))
        rr = int(np.clip(round(r - perp_r * offset_px), 0, h - 1))
        cr = int(np.clip(round(c - perp_c * offset_px), 0, w - 1))

        if mask[rl, cl]:
            diffs_left.append(z_center - float(ndom[rl, cl]))
        if mask[rr, cr]:
            diffs_right.append(z_center - float(ndom[rr, cr]))

    mean_left = float(np.mean(diffs_left)) if diffs_left else 0.0
    mean_right = float(np.mean(diffs_right)) if diffs_right else 0.0
    return mean_left, mean_right


def _planes_along_line(line_ndom, plane_labels, n_samples=20):
    """Which plane labels appear along the line?
    Returns set of plane labels (excluding 0)."""
    h, w = plane_labels.shape
    rows, cols = _sample_along_line(
        line_ndom[0, 0], line_ndom[0, 1],
        line_ndom[1, 0], line_ndom[1, 1],
        n_samples,
    )
    ri = np.clip(np.round(rows).astype(int), 0, h - 1)
    ci = np.clip(np.round(cols).astype(int), 0, w - 1)
    labels = set(int(x) for x in plane_labels[ri, ci] if x > 0)
    return labels


def _planes_beside_line(line_ndom, plane_labels, mask, gsd, offset_m=1.5):
    """Plane labels on each side of the line.
    Returns (left_labels, right_labels) as sets."""
    h, w = plane_labels.shape
    dr = line_ndom[1, 0] - line_ndom[0, 0]
    dc = line_ndom[1, 1] - line_ndom[0, 1]
    length = np.hypot(dr, dc)
    if length < 1e-6:
        return set(), set()

    perp_r = -dc / length
    perp_c = dr / length
    offset_px = offset_m / gsd

    rows, cols = _sample_along_line(
        line_ndom[0, 0], line_ndom[0, 1],
        line_ndom[1, 0], line_ndom[1, 1],
        n_samples=20,
    )

    left_labels = set()
    right_labels = set()
    for r, c in zip(rows, cols):
        rl = int(np.clip(round(r + perp_r * offset_px), 0, h - 1))
        cl = int(np.clip(round(c + perp_c * offset_px), 0, w - 1))
        rr = int(np.clip(round(r - perp_r * offset_px), 0, h - 1))
        cr = int(np.clip(round(c - perp_c * offset_px), 0, w - 1))

        if mask[rl, cl]:
            lbl = int(plane_labels[rl, cl])
            if lbl > 0:
                left_labels.add(lbl)
        if mask[rr, cr]:
            lbl = int(plane_labels[rr, cr])
            if lbl > 0:
                right_labels.add(lbl)

    return left_labels, right_labels


def classify_line(line_ndom, ndom, mask, plane_labels, planes, ridge_angle, gsd):
    """Classify a single wireframe line using nDOM height context.

    Returns one of: 'ridge', 'traufe', 'ortgang', 'kehle', 'grat', 'unknown'
    """
    edge_frac = _fraction_on_mask_edge(line_ndom, mask)
    line_angle = _line_angle_deg(line_ndom)
    is_external = edge_frac > 0.5

    if is_external:
        angle_diff = _angular_diff(line_angle, ridge_angle)
        if angle_diff < 35:
            return 'traufe'
        else:
            return 'ortgang'

    diff_left, diff_right = _perpendicular_height_comparison(
        line_ndom, ndom, mask, gsd, offset_m=1.5)

    left_labels, right_labels = _planes_beside_line(
        line_ndom, plane_labels, mask, gsd, offset_m=1.5)

    both_positive = diff_left > 0.15 and diff_right > 0.15
    both_negative = diff_left < -0.15 and diff_right < -0.15
    different_planes = left_labels != right_labels and len(left_labels) > 0 and len(right_labels) > 0

    plane_by_label = {p['label']: p for p in planes}

    if different_planes:
        ll = list(left_labels)
        rl = list(right_labels)
        if ll and rl:
            p_left = plane_by_label.get(ll[0])
            p_right = plane_by_label.get(rl[0])
            if p_left and p_right:
                aspect_diff = abs(p_left['aspect_deg'] - p_right['aspect_deg'])
                aspect_diff = min(aspect_diff, 360 - aspect_diff)
                if aspect_diff > 100:
                    if both_positive:
                        return 'ridge'
                    elif both_negative:
                        return 'kehle'

    if both_positive:
        if is_external or edge_frac > 0.3:
            return 'grat'
        return 'ridge'

    if both_negative:
        return 'kehle'

    if different_planes:
        mean_diff = (diff_left + diff_right) / 2
        if mean_diff > 0.1:
            return 'grat'

    return 'unknown'


def fuse_lines(rmn_lines, rmn_scores, ndom, mask, plane_labels, planes,
               ridge_angle, gsd=0.5, min_length_m=1.5, ransac_lines=None,
               ransac_lengths=None, alkis_rect=None):
    """Classify RoofMapNet lines and merge with RANSAC structural lines.

    Parameters
    ----------
    rmn_lines : (N, 2, 2) float — RoofMapNet lines in 128×128 feature coords.
                Each line: [[y1, x1], [y2, x2]].
    rmn_scores : (N,) float — confidence scores.
    ndom : (H, W) float — normalized height model.
    mask : (H, W) bool — roof mask.
    plane_labels : (H, W) int — RANSAC plane label map.
    planes : list of dicts — RANSAC plane info.
    ridge_angle : float — ridge direction 0–180°.
    gsd : float — nDOM ground sampling distance in m/px.
    min_length_m : float — minimum line length to keep.
    ransac_lines : dict — existing RANSAC line_segments dict (optional).
    ransac_lengths : dict — existing RANSAC lengths dict (optional).

    Returns
    -------
    dict with keys:
        line_segments: dict of lists of (r1, c1, r2, c2) in nDOM coords
        lengths: dict of float measurements in meters
        classified_lines: list of dicts with line info + classification
        source: 'fusion'
    """
    if len(rmn_lines) == 0:
        if ransac_lines and ransac_lengths:
            return {
                'line_segments': ransac_lines,
                'lengths': ransac_lengths,
                'classified_lines': [],
                'source': 'ransac_only',
            }
        empty_segs = {k: [] for k in ('ridge', 'traufe', 'ortgang', 'kehle', 'grat')}
        empty_lens = {k: 0.0 for k in ('first_m', 'traufe_m', 'ortgang_m', 'kehle_m', 'grat_m')}
        return {
            'line_segments': empty_segs,
            'lengths': empty_lens,
            'classified_lines': [],
            'source': 'empty',
        }

    lines_ndom = _to_ndom_coords(rmn_lines, ndom.shape)

    if alkis_rect is not None:
        lines_ndom = _snap_to_rect(lines_ndom, alkis_rect, thresh_px=4.0)
        lines_ndom = _merge_close_vertices(lines_ndom, thresh_px=2.0)

    classified = []
    segments = {'ridge': [], 'traufe': [], 'ortgang': [], 'kehle': [], 'grat': []}

    for i in range(len(lines_ndom)):
        ln = lines_ndom[i]
        length_m = _line_length_m(ln, gsd)
        if length_m < min_length_m:
            continue

        ltype = classify_line(ln, ndom, mask, plane_labels, planes,
                              ridge_angle, gsd)
        if ltype == 'unknown':
            continue

        r1, c1 = float(ln[0, 0]), float(ln[0, 1])
        r2, c2 = float(ln[1, 0]), float(ln[1, 1])

        seg = (r1, c1, r2, c2)
        segments[ltype].append(seg)

        classified.append({
            'type': ltype,
            'segment': seg,
            'length_m': round(length_m, 2),
            'score': float(rmn_scores[i]),
            'source': 'roofmapnet',
        })

    if ransac_lines:
        for ltype, segs in ransac_lines.items():
            key = ltype
            if key not in segments:
                continue
            for seg in segs:
                r1, c1, r2, c2 = seg
                dup = False
                for existing in segments[key]:
                    er1, ec1, er2, ec2 = existing
                    mid_dist = np.hypot(
                        (r1 + r2) / 2 - (er1 + er2) / 2,
                        (c1 + c2) / 2 - (ec1 + ec2) / 2,
                    ) * gsd
                    if mid_dist < 2.0:
                        dup = True
                        break
                if not dup:
                    segments[key].append(seg)
                    length_m = float(np.hypot(r2 - r1, c2 - c1)) * gsd
                    classified.append({
                        'type': key,
                        'segment': seg,
                        'length_m': round(length_m, 2),
                        'score': 1.0,
                        'source': 'ransac',
                    })

    lengths = {}
    for ltype, key in [('ridge', 'first_m'), ('traufe', 'traufe_m'),
                       ('ortgang', 'ortgang_m'), ('kehle', 'kehle_m'),
                       ('grat', 'grat_m')]:
        total = sum(np.hypot(r2 - r1, c2 - c1) * gsd
                    for r1, c1, r2, c2 in segments[ltype])
        lengths[key] = round(total, 2)

    return {
        'line_segments': segments,
        'lengths': lengths,
        'classified_lines': classified,
        'source': 'fusion',
    }


def _extend_to_mask(center, direction, mask, step=0.5):
    """Walk from center along +/- direction until leaving the mask.
    Returns (endpoint_neg, endpoint_pos) in nDOM pixel coords.
    If center is outside the mask, searches nearby for a valid start."""
    h, w = mask.shape
    d = direction / (np.linalg.norm(direction) + 1e-9)

    r0, c0 = int(round(center[0])), int(round(center[1]))
    if not (0 <= r0 < h and 0 <= c0 < w and mask[r0, c0]):
        for radius in range(1, 15):
            best_pt = None
            best_d2 = float('inf')
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    tr, tc = r0 + dr, c0 + dc
                    if 0 <= tr < h and 0 <= tc < w and mask[tr, tc]:
                        d2 = dr * dr + dc * dc
                        if d2 < best_d2:
                            best_d2 = d2
                            best_pt = np.array([float(tr), float(tc)])
            if best_pt is not None:
                center = best_pt
                break

    endpoints = []
    for sign in [+1, -1]:
        pt = center.copy()
        last_inside = center.copy()
        for _ in range(500):
            pt = pt + sign * d * step
            r, c = int(round(pt[0])), int(round(pt[1]))
            if r < 0 or r >= h or c < 0 or c >= w or not mask[r, c]:
                break
            last_inside = pt.copy()
        endpoints.append(last_inside)
    return endpoints[1], endpoints[0]


def _fraction_inside_mask(line_ndom, mask, n_samples=20):
    """Fraction of sampled line points that fall inside the roof mask."""
    h, w = mask.shape
    rows, cols = _sample_along_line(
        line_ndom[0, 0], line_ndom[0, 1],
        line_ndom[1, 0], line_ndom[1, 1], n_samples)
    ri = np.clip(np.round(rows).astype(int), 0, h - 1)
    ci = np.clip(np.round(cols).astype(int), 0, w - 1)
    return float(mask[ri, ci].sum()) / n_samples


def _is_duplicate(seg, existing_segs, gsd, dist_thresh_m=2.0):
    """Check if a segment duplicates an existing one (midpoints within threshold)."""
    r1, c1, r2, c2 = seg
    mid_r, mid_c = (r1 + r2) / 2, (c1 + c2) / 2
    for er1, ec1, er2, ec2 in existing_segs:
        emr, emc = (er1 + er2) / 2, (ec1 + ec2) / 2
        if np.hypot(mid_r - emr, mid_c - emc) * gsd < dist_thresh_m:
            return True
    return False


def _mean_height_along_line(line_ndom, ndom, mask):
    """Mean nDOM height along the portion of a line inside the mask."""
    heights, valid = _height_profile(line_ndom, ndom, mask, n_samples=30)
    if valid.sum() == 0:
        return 0.0
    return float(heights[valid].mean())


def _line_center(line_ndom):
    """Midpoint of a line in nDOM coordinates."""
    return (line_ndom[0] + line_ndom[1]) / 2


def _project_onto_axis(point, origin, direction):
    """Signed scalar projection of (point - origin) onto direction vector."""
    d = direction / (np.linalg.norm(direction) + 1e-9)
    return float(np.dot(point - origin, d))


def _near_mask(line_ndom, mask, margin_px=3):
    """True if at least one endpoint is within margin_px of the mask."""
    h, w = mask.shape
    dilated = binary_dilation(mask, iterations=margin_px)
    for pt in [line_ndom[0], line_ndom[1]]:
        r = int(np.clip(round(pt[0]), 0, h - 1))
        c = int(np.clip(round(pt[1]), 0, w - 1))
        if dilated[r, c]:
            return True
    return False


def _best_fragment(fragment_infos, merge_dir, gsd, extend_to_span=False):
    """Pick the longest RoofMapNet fragment — exact original endpoints.

    If *extend_to_span* is True, build a line along *merge_dir* that
    covers the projected span of ALL fragments.  The line is placed at
    the perpendicular centre of the group so its angle matches the
    structural direction (ridge_dir or perp_dir), not whatever angle
    an individual fragment happens to have.
    """
    if not fragment_infos:
        return None, 0.0
    best = max(fragment_infos, key=lambda x: np.hypot(
        x['ln'][1, 0] - x['ln'][0, 0], x['ln'][1, 1] - x['ln'][0, 1]))
    ln = best['ln']
    r1, c1 = float(ln[0, 0]), float(ln[0, 1])
    r2, c2 = float(ln[1, 0]), float(ln[1, 1])

    if extend_to_span and len(fragment_infos) > 1:
        d_hat = merge_dir / np.linalg.norm(merge_dir)
        all_pts = []
        for x in fragment_infos:
            all_pts.extend([x['ln'][0], x['ln'][1]])
        all_pts = np.array(all_pts)
        centre = all_pts.mean(axis=0)
        along = np.dot(all_pts - centre, d_hat)
        p1 = centre + d_hat * float(np.min(along))
        p2 = centre + d_hat * float(np.max(along))
        r1, c1 = float(p1[0]), float(p1[1])
        r2, c2 = float(p2[0]), float(p2[1])
    elif extend_to_span and len(fragment_infos) == 1:
        pass  # single fragment — keep original endpoints

    return (r1, c1, r2, c2), float(np.hypot(r2 - r1, c2 - c1)) * gsd


def _clip_to_mask_extent(seg, mask, gsd, buffer_m=1.0):
    """Clip line endpoints that extend beyond the mask boundary.

    Each endpoint is checked independently against the mask extent in
    a +/-3 m perpendicular band.  Endpoints beyond the mask boundary
    are clipped inward; endpoints within the mask are NOT moved.
    Preserves RoofMapNet's detected lengths and natural asymmetry.
    """
    r1, c1, r2, c2 = seg
    mid = np.array([(r1 + r2) / 2, (c1 + c2) / 2])
    d = np.array([r2 - r1, c2 - c1])
    length = np.linalg.norm(d)
    if length < 1e-6:
        return seg, 0.0
    d_hat = d / length
    perp = np.array([-d_hat[1], d_hat[0]])

    buf_px = buffer_m / gsd
    band_px = 3.0 / gsd

    mr, mc = np.where(mask)
    if len(mr) == 0:
        return seg, length * gsd

    perp_dist = np.abs((mr - mid[0]) * perp[0] + (mc - mid[1]) * perp[1])
    in_band = perp_dist < band_px
    if not np.any(in_band):
        return seg, length * gsd

    along = (mr[in_band] - mid[0]) * d_hat[0] + (mc[in_band] - mid[1]) * d_hat[1]
    t_min = float(np.min(along)) - buf_px
    t_max = float(np.max(along)) + buf_px

    t_p1 = -length / 2
    t_p2 = +length / 2

    t_p1_new = max(t_p1, t_min)
    t_p2_new = min(t_p2, t_max)

    if t_p1_new >= t_p2_new:
        return seg, length * gsd

    new_p1 = mid + d_hat * t_p1_new
    new_p2 = mid + d_hat * t_p2_new
    new_len = (t_p2_new - t_p1_new) * gsd

    return (float(new_p1[0]), float(new_p1[1]),
            float(new_p2[0]), float(new_p2[1])), new_len


def classify_rmn_lines(rmn_lines, rmn_scores, ndom, mask, plane_labels, planes,
                       ridge_angle, gsd=0.5, min_length_m=1.0,
                       min_inside_frac=0.25):
    """Classify RoofMapNet lines for a simple Satteldach and merge fragments.

    Structural knowledge: a Satteldach has exactly 1 First (ridge),
    2 Traufe (eaves), 2 Ortgang (verge).

    Approach: classify ALL lines, group collinear fragments per
    role+position, merge each group into one line.  The final geometry
    comes entirely from RoofMapNet endpoints — no mask extension.
    """
    empty = {
        'line_segments': {k: [] for k in ('ridge', 'traufe', 'ortgang', 'kehle', 'grat')},
        'lengths': {k: 0.0 for k in ('first_m', 'traufe_m', 'ortgang_m', 'kehle_m', 'grat_m')},
        'classified_lines': [],
        'source': 'roofmapnet_classified',
    }
    if len(rmn_lines) == 0:
        return empty

    lines_ndom = _to_ndom_coords(rmn_lines, ndom.shape)

    ridge_rad = np.radians(ridge_angle)
    ridge_dir = np.array([np.sin(ridge_rad), np.cos(ridge_rad)])
    perp_dir = np.array([-ridge_dir[1], ridge_dir[0]])

    mr, mc = np.where(mask)
    if len(mr) == 0:
        return empty
    origin = np.array([mr.mean(), mc.mean()])

    perp_extent = max(1.0, float(np.ptp([
        _project_onto_axis(np.array([r, c]), origin, perp_dir)
        for r, c in zip(mr[::10], mc[::10])])))
    along_extent = max(1.0, float(np.ptp([
        _project_onto_axis(np.array([r, c]), origin, ridge_dir)
        for r, c in zip(mr[::10], mc[::10])])))

    # Grouping radius: fragments within this distance are merged
    merge_dist_px = 2.5 / gsd  # 2.5 m

    # ── 1. Build info for every line ─────────────────────────────────────
    infos = []
    for i in range(len(lines_ndom)):
        ln = lines_ndom[i]
        length_m = _line_length_m(ln, gsd)
        inside_frac = _fraction_inside_mask(ln, mask)
        near = _near_mask(ln, mask, margin_px=4)
        angle = _line_angle_deg(ln)
        angle_diff = _angular_diff(angle, ridge_angle)
        center = _line_center(ln)
        mean_h = _mean_height_along_line(ln, ndom, mask)
        pos_perp = _project_onto_axis(center, origin, perp_dir)
        pos_along = _project_onto_axis(center, origin, ridge_dir)

        r1, c1 = float(ln[0, 0]), float(ln[0, 1])
        r2, c2 = float(ln[1, 0]), float(ln[1, 1])

        infos.append({
            'idx': i, 'seg': (r1, c1, r2, c2), 'ln': ln,
            'length_m': length_m, 'score': float(rmn_scores[i]),
            'inside_frac': inside_frac, 'near_mask': near,
            'angle_diff': angle_diff, 'mean_h': mean_h,
            'pos_perp': pos_perp, 'pos_along': pos_along,
            'is_parallel': angle_diff < 20,
            'is_perpendicular': angle_diff > 70,
        })

    all_classified = []
    assigned = set()

    # ── 2. Ridge: all parallel inside lines near the height peak ────────
    ridge_cands = [x for x in infos
                   if x['is_parallel'] and x['inside_frac'] >= 0.3
                   and x['length_m'] >= min_length_m]
    ridge_perp_pos = 0.0
    if ridge_cands:
        best = max(ridge_cands, key=lambda x: x['mean_h'] - x['angle_diff'] * 0.15)
        ridge_perp_pos = best['pos_perp']
        group = [x for x in ridge_cands
                 if abs(x['pos_perp'] - best['pos_perp']) < merge_dist_px
                 and x['mean_h'] >= best['mean_h'] - 1.5]
        for x in group:
            assigned.add(x['idx'])
        seg, length = _best_fragment(group, ridge_dir, gsd)
        all_classified.append({
            'segment': seg, 'length_m': round(length, 2),
            'score': max(x['score'] for x in group),
            'source': 'roofmapnet', 'type': 'ridge', 'status': 'accepted',
            'n_fragments': len(group),
        })

    # ── 3. Traufe: all parallel edge lines per side, merged ──────────
    side_margin = max(2.0, perp_extent * 0.1)
    traufe_found_sides = set()
    for sign, side_name in [(+1, 'pos'), (-1, 'neg')]:
        cands = [x for x in infos
                 if x['is_parallel'] and x['near_mask']
                 and x['length_m'] >= min_length_m
                 and x['idx'] not in assigned
                 and sign * (x['pos_perp'] - ridge_perp_pos) > side_margin]
        if not cands:
            continue
        best = max(cands, key=lambda x: x['length_m'] * (0.5 + x['score']))
        group = [x for x in cands
                 if abs(x['pos_perp'] - best['pos_perp']) < merge_dist_px]
        for x in group:
            assigned.add(x['idx'])
        seg, length = _best_fragment(group, ridge_dir, gsd)
        all_classified.append({
            'segment': seg, 'length_m': round(length, 2),
            'score': max(x['score'] for x in group),
            'source': 'roofmapnet', 'type': 'traufe', 'status': 'accepted',
            'n_fragments': len(group),
        })
        traufe_found_sides.add(side_name)

    # Fallback: if only 1 traufe found, retry missing side with relaxed criteria
    if len(traufe_found_sides) < 2:
        for sign, side_name in [(+1, 'pos'), (-1, 'neg')]:
            if side_name in traufe_found_sides:
                continue
            cands = [x for x in infos
                     if x['is_parallel']
                     and x['length_m'] >= min_length_m
                     and x['idx'] not in assigned
                     and sign * (x['pos_perp'] - ridge_perp_pos) > 1.0]
            if not cands:
                continue
            best = max(cands, key=lambda x: abs(x['pos_perp'] - ridge_perp_pos))
            group = [x for x in cands
                     if abs(x['pos_perp'] - best['pos_perp']) < merge_dist_px]
            for x in group:
                assigned.add(x['idx'])
            seg, length = _best_fragment(group, ridge_dir, gsd)
            all_classified.append({
                'segment': seg, 'length_m': round(length, 2),
                'score': max(x['score'] for x in group),
                'source': 'roofmapnet', 'type': 'traufe', 'status': 'accepted',
                'n_fragments': len(group),
            })

    # ── 4. Ortgang: per gable end, split by slope only if offset ───────
    #   If RoofMapNet detects fragments on BOTH sides of the ridge that
    #   are genuinely offset (no single fragment spans > 60 % of the
    #   perpendicular extent), accept two half-ortgangs.  Otherwise
    #   treat the end as one line.
    end_margin = max(2.0, along_extent * 0.15)
    ortgang_found_ends = set()
    for sign, end_name in [(+1, 'right'), (-1, 'left')]:
        all_end_cands = [x for x in infos
                         if x['is_perpendicular'] and x['near_mask']
                         and x['inside_frac'] > 0
                         and x['length_m'] >= min_length_m
                         and x['idx'] not in assigned
                         and sign * x['pos_along'] > end_margin]
        if not all_end_cands:
            continue
        pos_cands = [x for x in all_end_cands
                     if x['pos_perp'] >= ridge_perp_pos]
        neg_cands = [x for x in all_end_cands
                     if x['pos_perp'] < ridge_perp_pos]
        longest_m = max(x['length_m'] for x in all_end_cands)
        split = (pos_cands and neg_cands
                 and longest_m < perp_extent * gsd * 0.6)
        if split:
            for slope_cands in [pos_cands, neg_cands]:
                for x in slope_cands:
                    assigned.add(x['idx'])
                seg, length = _best_fragment(slope_cands, perp_dir, gsd,
                                             extend_to_span=True)
                all_classified.append({
                    'segment': seg, 'length_m': round(length, 2),
                    'score': max(x['score'] for x in slope_cands),
                    'source': 'roofmapnet', 'type': 'ortgang',
                    'status': 'accepted',
                    'n_fragments': len(slope_cands),
                })
        else:
            for x in all_end_cands:
                assigned.add(x['idx'])
            seg, length = _best_fragment(all_end_cands, perp_dir, gsd,
                                         extend_to_span=True)
            all_classified.append({
                'segment': seg, 'length_m': round(length, 2),
                'score': max(x['score'] for x in all_end_cands),
                'source': 'roofmapnet', 'type': 'ortgang',
                'status': 'accepted',
                'n_fragments': len(all_end_cands),
            })
        ortgang_found_ends.add(end_name)

    # Fallback: if only 1 end found, retry missing end with relaxed criteria
    if len(ortgang_found_ends) < 2:
        for sign, end_name in [(+1, 'right'), (-1, 'left')]:
            if end_name in ortgang_found_ends:
                continue
            cands = [x for x in infos
                     if x['is_perpendicular']
                     and x['length_m'] >= min_length_m
                     and x['idx'] not in assigned
                     and sign * x['pos_along'] > 1.0]
            if not cands:
                continue
            for x in cands:
                assigned.add(x['idx'])
            seg, length = _best_fragment(cands, perp_dir, gsd,
                                         extend_to_span=True)
            all_classified.append({
                'segment': seg, 'length_m': round(length, 2),
                'score': max(x['score'] for x in cands),
                'source': 'roofmapnet', 'type': 'ortgang',
                'status': 'accepted',
                'n_fragments': len(cands),
            })

    # ── 5. All other lines → rejected with reason ────────────────────────
    for info in infos:
        if info['idx'] in assigned:
            continue
        if info['length_m'] < min_length_m:
            reason = 'too_short'
        elif not info['near_mask']:
            reason = 'outside'
        else:
            kind = 'parallel' if info['is_parallel'] else 'perpendicular'
            reason = f'unassigned ({kind})'
        all_classified.append({
            'segment': info['seg'], 'length_m': round(info['length_m'], 2),
            'score': info['score'], 'source': 'roofmapnet',
            'type': reason, 'status': 'rejected',
        })

    # ── 6. Adjust line lengths to mask boundary (V7) ────────────────────
    for cl in all_classified:
        if cl['status'] != 'accepted' or cl.get('segment') is None:
            continue
        new_seg, new_len = _clip_to_mask_extent(cl['segment'], mask, gsd)
        cl['segment'] = new_seg
        cl['length_m'] = round(new_len, 2)

    # ── 7. Build result ──────────────────────────────────────────────────
    segments = {'ridge': [], 'traufe': [], 'ortgang': [], 'kehle': [], 'grat': []}
    for cl in all_classified:
        if cl['status'] == 'accepted' and cl['type'] in segments:
            segments[cl['type']].append(cl['segment'])

    lengths = {}
    for ltype, key in [('ridge', 'first_m'), ('traufe', 'traufe_m'),
                       ('ortgang', 'ortgang_m'), ('kehle', 'kehle_m'),
                       ('grat', 'grat_m')]:
        total = sum(np.hypot(r2 - r1, c2 - c1) * gsd
                    for r1, c1, r2, c2 in segments[ltype])
        lengths[key] = round(total, 2)

    return {
        'line_segments': segments,
        'lengths': lengths,
        'classified_lines': all_classified,
        'source': 'roofmapnet_classified',
    }


# ── V10: Ortgang from First+Traufe endpoints ───────────────────────────

def _line_endpoint(seg, ridge_dir, which):
    """Return the endpoint of seg that is further in +ridge_dir ('right')
    or -ridge_dir ('left')."""
    r1, c1, r2, c2 = seg
    t1 = r1 * ridge_dir[0] + c1 * ridge_dir[1]
    t2 = r2 * ridge_dir[0] + c2 * ridge_dir[1]
    if which == 'right':
        return (r1, c1) if t1 > t2 else (r2, c2)
    else:
        return (r1, c1) if t1 < t2 else (r2, c2)


def classify_rmn_lines_v10(rmn_lines, rmn_scores, ndom, mask, plane_labels,
                           planes, ridge_angle, gsd=0.5, min_length_m=1.0,
                           min_inside_frac=0.25):
    """V10: Detect ridge + traufe from RMN, derive ortgang from endpoints.

    For simple Satteldach only.  Ridge and traufe are classified from
    RoofMapNet fragments (same logic as V7).  Ortgang is NOT detected
    from RMN fragments — instead, each ortgang connects a ridge endpoint
    to the nearest traufe endpoint on the same gable side.
    """
    empty = {
        'line_segments': {k: [] for k in ('ridge', 'traufe', 'ortgang', 'kehle', 'grat')},
        'lengths': {k: 0.0 for k in ('first_m', 'traufe_m', 'ortgang_m', 'kehle_m', 'grat_m')},
        'classified_lines': [],
        'source': 'roofmapnet_v10',
    }
    if len(rmn_lines) == 0:
        return empty

    lines_ndom = _to_ndom_coords(rmn_lines, ndom.shape)

    ridge_rad = np.radians(ridge_angle)
    ridge_dir = np.array([np.sin(ridge_rad), np.cos(ridge_rad)])
    perp_dir = np.array([-ridge_dir[1], ridge_dir[0]])

    mr, mc = np.where(mask)
    if len(mr) == 0:
        return empty
    origin = np.array([mr.mean(), mc.mean()])

    perp_extent = max(1.0, float(np.ptp([
        _project_onto_axis(np.array([r, c]), origin, perp_dir)
        for r, c in zip(mr[::10], mc[::10])])))

    merge_dist_px = 2.5 / gsd

    # ── 1. Build info for every line ─────────────────────────────────────
    infos = []
    for i in range(len(lines_ndom)):
        ln = lines_ndom[i]
        length_m = _line_length_m(ln, gsd)
        inside_frac = _fraction_inside_mask(ln, mask)
        near = _near_mask(ln, mask, margin_px=4)
        angle = _line_angle_deg(ln)
        angle_diff = _angular_diff(angle, ridge_angle)
        center = _line_center(ln)
        mean_h = _mean_height_along_line(ln, ndom, mask)
        pos_perp = _project_onto_axis(center, origin, perp_dir)
        pos_along = _project_onto_axis(center, origin, ridge_dir)
        r1, c1 = float(ln[0, 0]), float(ln[0, 1])
        r2, c2 = float(ln[1, 0]), float(ln[1, 1])
        infos.append({
            'idx': i, 'seg': (r1, c1, r2, c2), 'ln': ln,
            'length_m': length_m, 'score': float(rmn_scores[i]),
            'inside_frac': inside_frac, 'near_mask': near,
            'angle_diff': angle_diff, 'mean_h': mean_h,
            'pos_perp': pos_perp, 'pos_along': pos_along,
            'is_parallel': angle_diff < 20,
            'is_perpendicular': angle_diff > 70,
        })

    all_classified = []
    assigned = set()

    # ── 2. Ridge (same as V7) ────────────────────────────────────────────
    ridge_cands = [x for x in infos
                   if x['is_parallel'] and x['inside_frac'] >= 0.3
                   and x['length_m'] >= min_length_m]
    ridge_seg = None
    ridge_perp_pos = 0.0
    if ridge_cands:
        best = max(ridge_cands, key=lambda x: x['mean_h'] - x['angle_diff'] * 0.15)
        ridge_perp_pos = best['pos_perp']
        group = [x for x in ridge_cands
                 if abs(x['pos_perp'] - best['pos_perp']) < merge_dist_px
                 and x['mean_h'] >= best['mean_h'] - 1.5]
        for x in group:
            assigned.add(x['idx'])
        ridge_seg, length = _best_fragment(group, ridge_dir, gsd)
        all_classified.append({
            'segment': ridge_seg, 'length_m': round(length, 2),
            'score': max(x['score'] for x in group),
            'source': 'roofmapnet', 'type': 'ridge', 'status': 'accepted',
            'n_fragments': len(group),
        })

    # ── 3. Traufe (same as V7) ──────────────────────────────────────────
    side_margin = max(2.0, perp_extent * 0.1)
    traufe_segs = []
    for sign, side_name in [(+1, 'pos'), (-1, 'neg')]:
        cands = [x for x in infos
                 if x['is_parallel'] and x['near_mask']
                 and x['length_m'] >= min_length_m
                 and x['idx'] not in assigned
                 and sign * (x['pos_perp'] - ridge_perp_pos) > side_margin]
        if not cands:
            continue
        best = max(cands, key=lambda x: x['length_m'] * (0.5 + x['score']))
        group = [x for x in cands
                 if abs(x['pos_perp'] - best['pos_perp']) < merge_dist_px]
        for x in group:
            assigned.add(x['idx'])
        seg, length = _best_fragment(group, ridge_dir, gsd)
        traufe_segs.append(seg)
        all_classified.append({
            'segment': seg, 'length_m': round(length, 2),
            'score': max(x['score'] for x in group),
            'source': 'roofmapnet', 'type': 'traufe', 'status': 'accepted',
            'n_fragments': len(group),
        })

    # Traufe fallback (relaxed)
    if len(traufe_segs) < 2:
        for sign, side_name in [(+1, 'pos'), (-1, 'neg')]:
            if any(sign * (_project_onto_axis(
                    np.array([(s[0]+s[2])/2, (s[1]+s[3])/2]), origin, perp_dir)
                    - ridge_perp_pos) > side_margin for s in traufe_segs):
                continue
            cands = [x for x in infos
                     if x['is_parallel'] and x['length_m'] >= min_length_m
                     and x['idx'] not in assigned
                     and sign * (x['pos_perp'] - ridge_perp_pos) > 1.0]
            if not cands:
                continue
            best = max(cands, key=lambda x: abs(x['pos_perp'] - ridge_perp_pos))
            group = [x for x in cands
                     if abs(x['pos_perp'] - best['pos_perp']) < merge_dist_px]
            for x in group:
                assigned.add(x['idx'])
            seg, length = _best_fragment(group, ridge_dir, gsd)
            traufe_segs.append(seg)
            all_classified.append({
                'segment': seg, 'length_m': round(length, 2),
                'score': max(x['score'] for x in group),
                'source': 'roofmapnet', 'type': 'traufe', 'status': 'accepted',
                'n_fragments': len(group),
            })

    # ── 4. Clip ridge + traufe to mask BEFORE deriving ortgang ──────────
    for cl in all_classified:
        if cl['status'] != 'accepted' or cl.get('segment') is None:
            continue
        new_seg, new_len = _clip_to_mask_extent(cl['segment'], mask, gsd)
        cl['segment'] = new_seg
        cl['length_m'] = round(new_len, 2)

    # ── 4b. Clip traufe to ridge extent along ridge_dir ──────────────
    #   On a Satteldach the traufe should not extend beyond the ridge
    #   endpoints (+ small overhang tolerance).
    if ridge_seg is not None:
        r_end_l = _line_endpoint(ridge_seg, ridge_dir, 'left')
        r_end_r = _line_endpoint(ridge_seg, ridge_dir, 'right')
        t_ridge_l = float(np.dot(np.array(r_end_l), ridge_dir))
        t_ridge_r = float(np.dot(np.array(r_end_r), ridge_dir))
        t_min_ridge = min(t_ridge_l, t_ridge_r) - 0.5 / gsd  # 0.5m overhang
        t_max_ridge = max(t_ridge_l, t_ridge_r) + 0.5 / gsd

        for cl in all_classified:
            if cl['type'] != 'traufe' or cl['status'] != 'accepted':
                continue
            r1, c1, r2, c2 = cl['segment']
            t1 = r1 * ridge_dir[0] + c1 * ridge_dir[1]
            t2 = r2 * ridge_dir[0] + c2 * ridge_dir[1]
            t1_new = max(min(t1, t2), t_min_ridge)
            t2_new = min(max(t1, t2), t_max_ridge)
            if t1_new >= t2_new:
                continue
            d_hat = ridge_dir / np.linalg.norm(ridge_dir)
            perp_comp = np.array([r1, c1]) - d_hat * t1
            p1 = perp_comp + d_hat * t1_new
            p2 = perp_comp + d_hat * t2_new
            cl['segment'] = (float(p1[0]), float(p1[1]),
                             float(p2[0]), float(p2[1]))
            cl['length_m'] = round(
                np.hypot(p2[0] - p1[0], p2[1] - p1[1]) * gsd, 2)

    # Update traufe_segs with clipped versions
    traufe_segs = [cl['segment'] for cl in all_classified
                   if cl['status'] == 'accepted' and cl['type'] == 'traufe']

    # ── 5. Ortgang: derived from clipped traufe endpoints ────────────────
    #   For each gable side, connect the two traufe endpoints on that side.
    #   This gives one straight ortgang per gable (traufe1 → traufe2).
    if len(traufe_segs) >= 2:
        for which in ['left', 'right']:
            ends = [_line_endpoint(tseg, ridge_dir, which)
                    for tseg in traufe_segs]
            seg = (ends[0][0], ends[0][1], ends[1][0], ends[1][1])
            length = np.hypot(seg[2] - seg[0], seg[3] - seg[1]) * gsd
            all_classified.append({
                'segment': seg, 'length_m': round(length, 2),
                'score': 1.0, 'source': 'derived',
                'type': 'ortgang', 'status': 'accepted',
                'n_fragments': 0,
            })

    # ── 6. Remaining lines → rejected ────────────────────────────────────
    for info in infos:
        if info['idx'] in assigned:
            continue
        if info['length_m'] < min_length_m:
            reason = 'too_short'
        elif not info['near_mask']:
            reason = 'outside'
        else:
            kind = 'parallel' if info['is_parallel'] else 'perpendicular'
            reason = f'unassigned ({kind})'
        all_classified.append({
            'segment': info['seg'], 'length_m': round(info['length_m'], 2),
            'score': info['score'], 'source': 'roofmapnet',
            'type': reason, 'status': 'rejected',
        })

    # ── 7. Build result ──────────────────────────────────────────────────
    segments = {'ridge': [], 'traufe': [], 'ortgang': [], 'kehle': [], 'grat': []}
    for cl in all_classified:
        if cl['status'] == 'accepted' and cl['type'] in segments:
            segments[cl['type']].append(cl['segment'])

    lengths = {}
    for ltype, key in [('ridge', 'first_m'), ('traufe', 'traufe_m'),
                       ('ortgang', 'ortgang_m'), ('kehle', 'kehle_m'),
                       ('grat', 'grat_m')]:
        total = sum(np.hypot(r2 - r1, c2 - c1) * gsd
                    for r1, c1, r2, c2 in segments[ltype])
        lengths[key] = round(total, 2)

    return {
        'line_segments': segments,
        'lengths': lengths,
        'classified_lines': all_classified,
        'source': 'roofmapnet_v10',
    }


def _find_ridge_col(ndom, mask, ridge_angle, gsd):
    """Find the ridge position from nDOM height maxima.

    For a near-vertical ridge (angle ~90°), returns the column of peak height.
    For a near-horizontal ridge (angle ~0°), returns the row of peak height.
    """
    h, w = ndom.shape
    is_vertical = abs(ridge_angle - 90) < 45

    if is_vertical:
        rows_m, cols_m = np.where(mask)
        if len(cols_m) == 0:
            return w / 2
        col_min, col_max = cols_m.min(), cols_m.max()
        best_col, best_h = col_min, -np.inf
        for c in range(col_min, col_max + 1):
            col_vals = ndom[mask[:, c], c]
            if len(col_vals) > 0:
                mean_h = float(col_vals.mean())
                if mean_h > best_h:
                    best_h, best_col = mean_h, c
        return best_col
    else:
        rows_m, cols_m = np.where(mask)
        if len(rows_m) == 0:
            return h / 2
        row_min, row_max = rows_m.min(), rows_m.max()
        best_row, best_h = row_min, -np.inf
        for r in range(row_min, row_max + 1):
            row_vals = ndom[r, mask[r, :]]
            if len(row_vals) > 0:
                mean_h = float(row_vals.mean())
                if mean_h > best_h:
                    best_h, best_row = mean_h, r
        return best_row


def _proj_to_segment(pt, a, b):
    """Project point onto line segment a→b. Returns (projected_point, distance)."""
    d = b - a
    len_sq = float(np.dot(d, d))
    if len_sq < 1e-10:
        return a.copy(), float(np.linalg.norm(pt - a))
    t = float(np.clip(np.dot(pt - a, d) / len_sq, 0, 1))
    proj = a + t * d
    return proj, float(np.linalg.norm(pt - proj))


def _snap_to_rect(lines_ndom, rect, thresh_px=4.0):
    """Snap line endpoints within thresh_px of an ALKIS rect edge onto that edge."""
    corners = np.array(rect)
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    out = lines_ndom.copy()
    for i in range(len(out)):
        for ep in range(2):
            best_d = thresh_px
            best_p = None
            for a, b in edges:
                p, d = _proj_to_segment(out[i, ep], a, b)
                if d < best_d:
                    best_d = d
                    best_p = p
            if best_p is not None:
                out[i, ep] = best_p
    return out


def _merge_close_vertices(lines_ndom, thresh_px=2.0):
    """Merge endpoints closer than thresh_px into shared vertices."""
    if len(lines_ndom) == 0:
        return lines_ndom
    out = lines_ndom.copy()
    all_ep = [(i, ep) for i in range(len(out)) for ep in range(2)]
    visited = set()
    for idx in range(len(all_ep)):
        if idx in visited:
            continue
        i1, ep1 = all_ep[idx]
        cluster = [(i1, ep1)]
        visited.add(idx)
        for idx2 in range(idx + 1, len(all_ep)):
            if idx2 in visited:
                continue
            i2, ep2 = all_ep[idx2]
            if np.linalg.norm(out[i1, ep1] - out[i2, ep2]) < thresh_px:
                cluster.append((i2, ep2))
                visited.add(idx2)
        if len(cluster) > 1:
            centroid = np.mean([out[i, ep] for i, ep in cluster], axis=0)
            for i, ep in cluster:
                out[i, ep] = centroid
    return out


def _mask_rect(mask):
    """Compute minAreaRect from mask contour, return 4 corners as (row, col)."""
    import cv2
    u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return [(float(pt[1]), float(pt[0])) for pt in box]


def _find_best_ridge(rmn_lines, rmn_scores, ndom, mask, ridge_angle, gsd):
    """Find the best ridge candidate from RoofMapNet lines (in feature coords).

    Accepts raw (unfiltered) lines — applies its own score + mask + height checks.
    Key constraint: ridge must be higher than both perpendicular sides.
    """
    if len(rmn_lines) == 0:
        return None
    lines_ndom = _to_ndom_coords(rmn_lines, ndom.shape)
    h, w = mask.shape
    best_score = -1
    best_line = None
    for i in range(len(lines_ndom)):
        if float(rmn_scores[i]) < 0.3:
            continue
        ln = lines_ndom[i]
        mid_r = int(np.clip(round((ln[0, 0] + ln[1, 0]) / 2), 0, h - 1))
        mid_c = int(np.clip(round((ln[0, 1] + ln[1, 1]) / 2), 0, w - 1))
        if not mask[mid_r, mid_c]:
            continue
        if _angular_diff(_line_angle_deg(ln), ridge_angle) > 20:
            continue
        length_m = _line_length_m(ln, gsd)
        if length_m < 2.0:
            continue
        if _fraction_on_mask_edge(ln, mask) > 0.3:
            continue
        diff_left, diff_right = _perpendicular_height_comparison(
            ln, ndom, mask, gsd, offset_m=1.5)
        if not (diff_left > 0.15 and diff_right > 0.15):
            continue
        heights, valid = _height_profile(ln, ndom, mask)
        if valid.sum() < 5:
            continue
        h_valid = heights[valid]
        mean_h = float(h_valid.mean())
        if float(h_valid.std()) > 1.0:
            continue
        score = float(rmn_scores[i]) * max(mean_h, 0.1)
        if score > best_score:
            best_score = score
            best_line = ln
    return best_line


def find_all_ridges(rmn_lines, rmn_scores, ndom, mask, gsd,
                    min_score=0.3, min_length_m=2.0, min_height_diff=0.10,
                    dedup_dist_m=2.0):
    """Find ALL ridge candidates from RoofMapNet lines (in feature coords).

    Unlike _find_best_ridge, this does NOT filter by ridge_angle — it accepts
    ridges in any orientation to handle T/L/complex roof forms.

    Returns list of (N, 2, 2) lines in nDOM pixel coords, sorted by mean height.
    """
    if len(rmn_lines) == 0:
        return []
    lines_ndom = _to_ndom_coords(rmn_lines, ndom.shape)
    h, w = mask.shape

    mask_buf = binary_dilation(mask, iterations=2)

    candidates = []
    for i in range(len(lines_ndom)):
        if float(rmn_scores[i]) < min_score:
            continue
        ln = lines_ndom[i]
        # Both endpoints must be within or near the mask
        r1 = int(np.clip(round(ln[0, 0]), 0, h - 1))
        c1 = int(np.clip(round(ln[0, 1]), 0, w - 1))
        r2 = int(np.clip(round(ln[1, 0]), 0, h - 1))
        c2 = int(np.clip(round(ln[1, 1]), 0, w - 1))
        if not (mask_buf[r1, c1] and mask_buf[r2, c2]):
            continue
        mid_r = int(np.clip(round((ln[0, 0] + ln[1, 0]) / 2), 0, h - 1))
        mid_c = int(np.clip(round((ln[0, 1] + ln[1, 1]) / 2), 0, w - 1))
        if not mask[mid_r, mid_c]:
            continue
        length_m = _line_length_m(ln, gsd)
        if length_m < min_length_m:
            continue
        if _fraction_on_mask_edge(ln, mask) > 0.3:
            continue
        diff_left, diff_right = _perpendicular_height_comparison(
            ln, ndom, mask, gsd, offset_m=1.5)
        if not (diff_left > min_height_diff and diff_right > min_height_diff):
            continue
        heights, valid = _height_profile(ln, ndom, mask)
        if valid.sum() < 3:
            continue
        h_valid = heights[valid]
        mean_h = float(h_valid.mean())
        if float(h_valid.std()) > 1.5:
            continue
        candidates.append({
            'line': ln.copy(),
            'mean_h': mean_h,
            'length_m': length_m,
            'score': float(rmn_scores[i]),
        })

    if not candidates:
        return []

    candidates.sort(key=lambda c: c['mean_h'], reverse=True)

    # Deduplicate: remove lines whose midpoint is within dedup_dist of a kept line
    dedup_px = dedup_dist_m / gsd
    kept = []
    for c in candidates:
        mid = (c['line'][0] + c['line'][1]) / 2
        too_close = False
        for k in kept:
            k_mid = (k['line'][0] + k['line'][1]) / 2
            if np.linalg.norm(mid - k_mid) < dedup_px:
                ang_c = _line_angle_deg(c['line'])
                ang_k = _line_angle_deg(k['line'])
                if _angular_diff(ang_c, ang_k) < 25:
                    too_close = True
                    break
        if not too_close:
            kept.append(c)

    return [c['line'] for c in kept]


def _intersect_line_with_segment(line_pt, line_dir, seg_a, seg_b):
    """Intersect infinite line (through line_pt, direction line_dir) with segment a→b.
    Returns intersection point (clamped to segment) or None if parallel."""
    ab = seg_b - seg_a
    det = line_dir[0] * ab[1] - line_dir[1] * ab[0]
    if abs(det) < 1e-10:
        return None
    dp = line_pt - seg_a
    s = (line_dir[0] * dp[1] - line_dir[1] * dp[0]) / det
    s = float(np.clip(s, 0.02, 0.98))
    return seg_a + s * ab


def generate_satteldach_wireframe(alkis_rect, ridge_angle, ndom, mask, gsd=0.5,
                                  alkis_mask=None, rmn_lines=None, rmn_scores=None):
    """Generate connected Satteldach wireframe: RoofMapNet ridge + ALKIS rect.

    Ridge position from the best RoofMapNet internal line parallel to ridge_angle.
    Falls back to nDOM height maxima if no RoofMapNet ridge found.
    Boundary from ALKIS rect (or mask rect when coverage ≥70%).

    Returns dict with 7 segments: 1 ridge + 2 traufe + 4 ortgang.
    """
    if not alkis_rect or len(alkis_rect) < 4:
        return None

    alkis_area = int(alkis_mask.sum()) if alkis_mask is not None else 0
    mask_area = int(mask.sum())
    coverage = mask_area / max(alkis_area, 1) if alkis_area > 0 else 1.0
    rect = alkis_rect
    if coverage >= 0.70:
        mr = _mask_rect(mask)
        if mr is not None:
            rect = mr

    corners = np.array(rect)

    traufe_edges = []
    ortgang_edges = []
    for i in range(4):
        c1, c2 = corners[i], corners[(i + 1) % 4]
        edge_angle = float(np.degrees(np.arctan2(c2[0]-c1[0], c2[1]-c1[1]))) % 180
        if _angular_diff(edge_angle, ridge_angle) < 45:
            traufe_edges.append((i, (i + 1) % 4))
        else:
            ortgang_edges.append((i, (i + 1) % 4))

    if len(traufe_edges) != 2 or len(ortgang_edges) != 2:
        return None

    ridge_line = None
    if rmn_lines is not None:
        ridge_line = _find_best_ridge(rmn_lines, rmn_scores, ndom, mask,
                                      ridge_angle, gsd)

    if ridge_line is not None:
        ridge_dir = ridge_line[1] - ridge_line[0]
        ridge_mid = (ridge_line[0] + ridge_line[1]) / 2
        mid_points = []
        for oi0, oi1 in ortgang_edges:
            a, b = corners[oi0], corners[oi1]
            pt = _intersect_line_with_segment(ridge_mid, ridge_dir, a, b)
            if pt is None:
                proj, _ = _proj_to_segment(ridge_mid, a, b)
                pt = proj
            mid_points.append(pt)
        mid1, mid2 = mid_points
    else:
        ridge_pos = _find_ridge_col(ndom, mask, ridge_angle, gsd)
        is_vertical = abs(ridge_angle - 90) < 45
        mid_points = []
        for oi0, oi1 in ortgang_edges:
            start, end = corners[oi0], corners[oi1]
            if is_vertical:
                frac = np.clip((ridge_pos - start[1]) / (end[1] - start[1] + 1e-9),
                               0.05, 0.95)
            else:
                frac = np.clip((ridge_pos - start[0]) / (end[0] - start[0] + 1e-9),
                               0.05, 0.95)
            mid_points.append(start + frac * (end - start))
        mid1, mid2 = mid_points

    ti0, ti1 = traufe_edges[0]
    tj0, tj1 = traufe_edges[1]
    oi0, oi1 = ortgang_edges[0]
    oj0, oj1 = ortgang_edges[1]

    segments = {'ridge': [], 'traufe': [], 'ortgang': [], 'kehle': [], 'grat': []}
    classified = []

    def _add(ltype, p1, p2):
        seg = (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
        length_m = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) * gsd
        segments[ltype].append(seg)
        classified.append({
            'type': ltype, 'segment': seg,
            'length_m': round(length_m, 2), 'score': 1.0,
            'source': 'wireframe',
        })

    _add('ridge', mid1, mid2)
    _add('traufe', corners[ti0], corners[ti1])
    _add('traufe', corners[tj0], corners[tj1])
    _add('ortgang', corners[oi0], mid1)
    _add('ortgang', mid1, corners[oi1])
    _add('ortgang', corners[oj0], mid2)
    _add('ortgang', mid2, corners[oj1])

    lengths = {}
    for ltype, key in [('ridge', 'first_m'), ('traufe', 'traufe_m'),
                       ('ortgang', 'ortgang_m'), ('kehle', 'kehle_m'),
                       ('grat', 'grat_m')]:
        total = sum(np.hypot(r2 - r1, c2 - c1) * gsd
                    for r1, c1, r2, c2 in segments[ltype])
        lengths[key] = round(total, 2)

    return {
        'line_segments': segments,
        'lengths': lengths,
        'classified_lines': classified,
        'source': 'wireframe',
    }


def _simplify_polygon(poly, min_edge_m=1.0, gsd=1.0):
    """Remove short edges from polygon by merging close vertices."""
    if len(poly) < 3:
        return poly
    kept = [poly[0]]
    for i in range(1, len(poly)):
        d = np.hypot(poly[i][0] - kept[-1][0], poly[i][1] - kept[-1][1]) * gsd
        if d >= min_edge_m:
            kept.append(poly[i])
    if (len(kept) > 1 and
        np.hypot(kept[-1][0] - kept[0][0], kept[-1][1] - kept[0][1]) * gsd < min_edge_m):
        kept.pop()
    return np.array(kept)


def _edge_angle_deg(p1, p2):
    """Angle of edge p1->p2 in degrees [0, 360), using atan2(Δcol, Δrow)."""
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))) % 360


def _find_dominant_directions(poly):
    """Find the two dominant perpendicular edge directions of a polygon.

    Returns (traufe_dir, ortgang_dir) in [0, 180). traufe_dir is the direction
    with the most total edge length (long sides = traufe for gabled roofs).
    """
    n = len(poly)
    edges = []
    for i in range(n):
        p1, p2 = poly[i], poly[(i + 1) % n]
        angle = _edge_angle_deg(p1, p2) % 180
        length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        edges.append((angle, length))

    best_a, best_len = 0, 0
    for angle, length in edges:
        total = sum(l for a, l in edges if _angular_diff(a, angle) < 20)
        if total > best_len:
            best_len = total
            best_a = angle

    dir_a = best_a
    dir_b = (dir_a + 90) % 180
    total_b = sum(l for a, l in edges if _angular_diff(a, dir_b) < 20)
    if total_b > best_len:
        return dir_b, dir_a
    return dir_a, dir_b


def _polygon_signed_area(poly):
    """Signed area via shoelace. Positive = CW in image coords (row↓, col→)."""
    n = len(poly)
    return 0.5 * sum(
        poly[i][1] * poly[(i + 1) % n][0] - poly[(i + 1) % n][1] * poly[i][0]
        for i in range(n)
    )


def _find_concave_vertices(poly, min_cross=10.0):
    """Find concave vertices, accounting for polygon winding direction."""
    n = len(poly)
    signed_area = _polygon_signed_area(poly)
    concave = []
    for idx in range(n):
        p0 = poly[(idx - 1) % n]
        p1 = poly[idx]
        p2 = poly[(idx + 1) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if signed_area * cross > 0 and abs(cross) > min_cross:
            concave.append(idx)
    return concave


def _find_t_junction_vertices(poly, gsd=1.0):
    """Find T-junction vertices: concave vertex + the vertex connected by
    the shortest edge in the ortgang direction near the junction.

    Returns list of (concave_idx, partner_idx) pairs, or empty if rectangular.
    """
    concave = _find_concave_vertices(poly)
    if not concave:
        return []

    n = len(poly)
    junctions = []
    for ci in concave:
        prev_i = (ci - 1) % n
        next_i = (ci + 1) % n
        len_prev = np.hypot(poly[ci][0] - poly[prev_i][0],
                            poly[ci][1] - poly[prev_i][1]) * gsd
        len_next = np.hypot(poly[next_i][0] - poly[ci][0],
                            poly[next_i][1] - poly[ci][1]) * gsd
        partner = prev_i if len_prev < len_next else next_i
        junctions.append((ci, partner))

    return junctions


def _find_ridge_for_wing(poly_verts, ndom, mask, gsd, traufe_dir):
    """Find the ridge line for a wing by scanning nDOM heights perpendicular to traufe.

    Returns (ridge_p1, ridge_p2) in nDOM pixel coords.
    """
    h, w = ndom.shape
    verts = np.array(poly_verts)
    center = verts.mean(axis=0)

    traufe_rad = np.radians(traufe_dir)
    ortgang_rad = traufe_rad + np.pi / 2

    traufe_vec = np.array([np.cos(traufe_rad), np.sin(traufe_rad)])
    ortgang_vec = np.array([np.cos(ortgang_rad), np.sin(ortgang_rad)])

    ortgang_projs = [(v - center).dot(ortgang_vec) for v in verts]
    ort_min, ort_max = min(ortgang_projs), max(ortgang_projs)

    best_h = -1
    best_ort = (ort_min + ort_max) / 2

    n_steps = max(10, int((ort_max - ort_min) / gsd))
    for step in range(n_steps + 1):
        frac = step / max(n_steps, 1)
        ort_pos = ort_min + frac * (ort_max - ort_min)
        sample_pt = center + ort_pos * ortgang_vec

        r, c = int(round(sample_pt[0])), int(round(sample_pt[1]))
        if 0 <= r < h and 0 <= c < w and mask[r, c]:
            height = float(ndom[r, c])
            if height > best_h:
                best_h = height
                best_ort = ort_pos

    ridge_center = center + best_ort * ortgang_vec

    traufe_projs = [(v - center).dot(traufe_vec) for v in verts]
    tr_min, tr_max = min(traufe_projs), max(traufe_projs)
    margin = 0.3
    ridge_p1 = ridge_center + (tr_min + margin) * traufe_vec
    ridge_p2 = ridge_center + (tr_max - margin) * traufe_vec

    return ridge_p1, ridge_p2


def _closest_point_on_segment(p, a, b):
    """Project point p onto segment a-b, return closest point."""
    ab = b - a
    ab_len2 = ab.dot(ab)
    if ab_len2 < 1e-12:
        return a.copy()
    t = np.clip((p - a).dot(ab) / ab_len2, 0, 1)
    return a + t * ab


def generate_polygon_wireframe(alkis_poly, ndom, mask, gsd=1.0):
    """Generate a wireframe from the actual ALKIS polygon shape.

    Handles T-shaped and L-shaped buildings by:
    1. Classifying all polygon edges as traufe/ortgang by angle
    2. Finding ridge lines for each wing using nDOM heights
    3. Computing kehle lines at T-junctions (from junction to ridge)

    Parameters
    ----------
    alkis_poly : (N, 2) array of (row, col) vertices in nDOM pixel coords.
    ndom : (H, W) float — normalized height model.
    mask : (H, W) bool — roof mask.
    gsd : float — ground sampling distance in m/px.

    Returns
    -------
    dict with classified_lines, line_segments, lengths, source='polygon'.
    """
    poly = _simplify_polygon(alkis_poly, min_edge_m=1.0, gsd=gsd)
    n = len(poly)
    if n < 4:
        return None

    traufe_dir, ortgang_dir = _find_dominant_directions(poly)

    segments = {'ridge': [], 'traufe': [], 'ortgang': [], 'kehle': [], 'grat': []}
    classified = []

    def _add(ltype, p1, p2):
        seg = (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
        length_m = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) * gsd
        if length_m < 0.5:
            return
        segments[ltype].append(seg)
        classified.append({
            'type': ltype, 'segment': seg,
            'length_m': round(length_m, 2), 'score': 1.0,
            'source': 'polygon',
        })

    for i in range(n):
        p1, p2 = poly[i], poly[(i + 1) % n]
        edge_a = _edge_angle_deg(p1, p2) % 180
        if _angular_diff(edge_a, traufe_dir) < 30:
            _add('traufe', p1, p2)
        else:
            _add('ortgang', p1, p2)

    junctions = _find_t_junction_vertices(poly, gsd=gsd)
    is_t_shape = len(junctions) > 0

    if is_t_shape:
        junction_verts = set()
        for ci, pi in junctions:
            junction_verts.add(ci)
            junction_verts.add(pi)

        junction_pts = np.array([poly[i] for i in junction_verts])
        junction_center = junction_pts.mean(axis=0)

        traufe_vec = np.array([np.cos(np.radians(traufe_dir)),
                               np.sin(np.radians(traufe_dir))])

        all_projs = [(poly[i] - junction_center).dot(traufe_vec) for i in range(n)]
        split_val = 0.0

        wing1_verts = [poly[i] for i in range(n)
                       if (poly[i] - junction_center).dot(traufe_vec) <= split_val + 2]
        wing2_verts = [poly[i] for i in range(n)
                       if (poly[i] - junction_center).dot(traufe_vec) >= split_val - 2]

        if len(wing1_verts) >= 3:
            ridge1_p1, ridge1_p2 = _find_ridge_for_wing(
                wing1_verts, ndom, mask, gsd, traufe_dir)
            _add('ridge', ridge1_p1, ridge1_p2)
        if len(wing2_verts) >= 3:
            ridge2_p1, ridge2_p2 = _find_ridge_for_wing(
                wing2_verts, ndom, mask, gsd, traufe_dir)
            _add('ridge', ridge2_p1, ridge2_p2)

        for ci, pi in junctions:
            jpt = poly[ci]
            best_dist = float('inf')
            best_pt = jpt
            for seg in segments['ridge']:
                r1, c1, r2, c2 = seg
                closest = _closest_point_on_segment(
                    jpt, np.array([r1, c1]), np.array([r2, c2]))
                d = np.hypot(closest[0] - jpt[0], closest[1] - jpt[1])
                if d < best_dist:
                    best_dist = d
                    best_pt = closest
            if best_dist * gsd > 1.0:
                _add('kehle', jpt, best_pt)
    else:
        ridge_p1, ridge_p2 = _find_ridge_for_wing(poly, ndom, mask, gsd, traufe_dir)
        _add('ridge', ridge_p1, ridge_p2)

    lengths = {}
    for ltype, key in [('ridge', 'first_m'), ('traufe', 'traufe_m'),
                       ('ortgang', 'ortgang_m'), ('kehle', 'kehle_m'),
                       ('grat', 'grat_m')]:
        total = sum(np.hypot(r2 - r1, c2 - c1) * gsd
                    for r1, c1, r2, c2 in segments[ltype])
        lengths[key] = round(total, 2)

    return {
        'line_segments': segments,
        'lengths': lengths,
        'classified_lines': classified,
        'source': 'polygon',
        'is_t_shape': is_t_shape,
    }


def auto_align_drone(alkis_poly, drone_img, gsd=1.0):
    """Auto-compute nDOM→drone transform by IoU optimization.

    Uses color segmentation to detect the roof in the drone image,
    then grid-searches for the (scale, offset) that maximizes overlap
    with the ALKIS polygon.

    Returns (scale, off_r, off_c) or None if alignment fails.
    """
    import cv2

    h_d, w_d = drone_img.shape[:2]
    hsv = cv2.cvtColor(drone_img, cv2.COLOR_BGR2HSV)
    mask_r1 = (hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 60)
    mask_r2 = (hsv[:, :, 0] > 160) & (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 60)
    roof_mask = (mask_r1 | mask_r2).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_CLOSE, kernel)
    roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_OPEN, kernel)
    roof_bool = roof_mask > 0

    poly = alkis_poly
    centroid_r = poly[:, 0].mean()
    centroid_c = poly[:, 1].mean()

    best_iou = 0
    best_params = None

    for scale in range(80, 220, 8):
        for cx in range(w_d // 4, 3 * w_d // 4, 60):
            for cy in range(h_d // 4, 3 * h_d // 4, 60):
                off_r = cy - centroid_r * scale
                off_c = cx - centroid_c * scale

                pts_img = []
                for p in poly:
                    px = int(p[1] * scale + off_c)
                    py = int(p[0] * scale + off_r)
                    pts_img.append([px, py])
                pts_cv = np.array(pts_img, dtype=np.int32)

                poly_mask = np.zeros((h_d, w_d), dtype=np.uint8)
                cv2.fillPoly(poly_mask, [pts_cv], 1)

                intersection = ((poly_mask > 0) & roof_bool).sum()
                union = ((poly_mask > 0) | roof_bool).sum()
                iou = intersection / max(union, 1)

                if iou > best_iou:
                    best_iou = iou
                    best_params = (scale, cx, cy)

    if best_params is None:
        return None

    scale, cx, cy = best_params
    for scale2 in range(scale - 8, scale + 9, 2):
        for cx2 in range(cx - 60, cx + 61, 15):
            for cy2 in range(cy - 60, cy + 61, 15):
                off_r = cy2 - centroid_r * scale2
                off_c = cx2 - centroid_c * scale2

                pts_img = []
                for p in poly:
                    px = int(p[1] * scale2 + off_c)
                    py = int(p[0] * scale2 + off_r)
                    pts_img.append([px, py])
                pts_cv = np.array(pts_img, dtype=np.int32)

                poly_mask = np.zeros((h_d, w_d), dtype=np.uint8)
                cv2.fillPoly(poly_mask, [pts_cv], 1)

                intersection = ((poly_mask > 0) & roof_bool).sum()
                union = ((poly_mask > 0) | roof_bool).sum()
                iou = intersection / max(union, 1)

                if iou > best_iou:
                    best_iou = iou
                    best_params = (scale2, cx2, cy2)

    scale, cx, cy = best_params
    off_r = cy - centroid_r * scale
    off_c = cx - centroid_c * scale

    return (scale, off_r, off_c)
