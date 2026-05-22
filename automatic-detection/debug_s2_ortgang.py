"""Debug: show which RMN fragments are selected as ortgang for S2.
Monkey-patches classify_rmn_lines to add debug output."""
import os, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'src'))

import wireframe_fusion as wf

_orig_classify = wf.classify_rmn_lines

def _debug_classify(rmn_lines, rmn_scores, ndom, mask, plane_labels, planes,
                    ridge_angle, gsd=0.5, min_length_m=1.0, min_inside_frac=0.25):
    lines_ndom = wf._to_ndom_coords(rmn_lines, ndom.shape)
    ridge_rad = np.radians(ridge_angle)
    ridge_dir = np.array([np.sin(ridge_rad), np.cos(ridge_rad)])
    perp_dir = np.array([-ridge_dir[1], ridge_dir[0]])

    mr, mc = np.where(mask)
    origin = np.array([mr.mean(), mc.mean()])

    perp_extent = max(1.0, float(np.ptp([
        wf._project_onto_axis(np.array([r, c]), origin, perp_dir)
        for r, c in zip(mr[::10], mc[::10])])))
    along_extent = max(1.0, float(np.ptp([
        wf._project_onto_axis(np.array([r, c]), origin, ridge_dir)
        for r, c in zip(mr[::10], mc[::10])])))

    print(f"\n  DEBUG: ridge_angle={ridge_angle:.1f}°")
    print(f"  DEBUG: ridge_dir={ridge_dir}, perp_dir={perp_dir}")
    print(f"  DEBUG: perp_extent={perp_extent:.1f}px ({perp_extent*gsd:.1f}m)")
    print(f"  DEBUG: along_extent={along_extent:.1f}px ({along_extent*gsd:.1f}m)")

    infos = []
    for i in range(len(lines_ndom)):
        ln = lines_ndom[i]
        length_m = wf._line_length_m(ln, gsd)
        inside_frac = wf._fraction_inside_mask(ln, mask)
        near = wf._near_mask(ln, mask, margin_px=4)
        angle = wf._line_angle_deg(ln)
        angle_diff = wf._angular_diff(angle, ridge_angle)
        center = wf._line_center(ln)
        mean_h = wf._mean_height_along_line(ln, ndom, mask)
        pos_perp = wf._project_onto_axis(center, origin, perp_dir)
        pos_along = wf._project_onto_axis(center, origin, ridge_dir)
        r1, c1 = float(ln[0,0]), float(ln[0,1])
        r2, c2 = float(ln[1,0]), float(ln[1,1])
        infos.append({
            'idx': i, 'seg': (r1, c1, r2, c2), 'ln': ln,
            'length_m': length_m, 'score': float(rmn_scores[i]),
            'inside_frac': inside_frac, 'near_mask': near,
            'angle_diff': angle_diff, 'mean_h': mean_h,
            'pos_perp': pos_perp, 'pos_along': pos_along,
            'is_parallel': angle_diff < 20,
            'is_perpendicular': angle_diff > 70,
        })

    # Find ridge
    ridge_cands = [x for x in infos if x['is_parallel'] and x['inside_frac'] >= 0.3 and x['length_m'] >= 1.0]
    ridge_perp_pos = 0.0
    assigned = set()
    if ridge_cands:
        best_ridge = max(ridge_cands, key=lambda x: x['mean_h'] - x['angle_diff'] * 0.15)
        ridge_perp_pos = best_ridge['pos_perp']
        group = [x for x in ridge_cands if abs(x['pos_perp'] - best_ridge['pos_perp']) < 2.5/gsd
                 and x['mean_h'] >= best_ridge['mean_h'] - 1.5]
        for x in group:
            assigned.add(x['idx'])
        print(f"  DEBUG: ridge at perp_pos={ridge_perp_pos:.1f}, assigned={[x['idx'] for x in group]}")

    # Traufe
    side_margin = max(2.0, perp_extent * 0.1)
    for sign in [+1, -1]:
        cands = [x for x in infos if x['is_parallel'] and x['near_mask']
                 and x['length_m'] >= 1.0 and x['idx'] not in assigned
                 and sign * (x['pos_perp'] - ridge_perp_pos) > side_margin]
        if cands:
            best = max(cands, key=lambda x: x['length_m'] * (0.5 + x['score']))
            group = [x for x in cands if abs(x['pos_perp'] - best['pos_perp']) < 2.5/gsd]
            for x in group:
                assigned.add(x['idx'])

    print(f"  DEBUG: assigned after ridge+traufe: {sorted(assigned)}")

    # ORTGANG DEBUG
    end_margin = max(2.0, along_extent * 0.15)
    print(f"\n  DEBUG ORTGANG: end_margin={end_margin:.1f}px, 60% perp={perp_extent*gsd*0.6:.1f}m")

    perp_lines = [x for x in infos if x['is_perpendicular']]
    print(f"\n  ALL PERPENDICULAR lines ({len(perp_lines)}):")
    print(f"  {'idx':>4} {'len_m':>6} {'score':>5} {'ang_d':>5} {'inside':>6} {'near':>4} "
          f"{'along':>7} {'perp':>7} {'assigned':>8}")
    for x in sorted(perp_lines, key=lambda x: x['pos_along']):
        r1, c1, r2, c2 = x['seg']
        print(f"  {x['idx']:4d} {x['length_m']:6.1f} {x['score']:5.2f} {x['angle_diff']:5.1f} "
              f"{x['inside_frac']:6.2f} {'Y' if x['near_mask'] else 'N':>4} "
              f"{x['pos_along']:7.1f} {x['pos_perp']:7.1f} "
              f"{'YES' if x['idx'] in assigned else '':>8}")

    for sign, end_name in [(+1, 'right'), (-1, 'left')]:
        all_end_cands = [x for x in infos
                         if x['is_perpendicular'] and x['near_mask']
                         and x['inside_frac'] > 0
                         and x['length_m'] >= 1.0
                         and x['idx'] not in assigned
                         and sign * x['pos_along'] > end_margin]
        if not all_end_cands:
            print(f"\n  {end_name.upper()}: no candidates (sign={sign})")
            continue

        pos_cands = [x for x in all_end_cands if x['pos_perp'] >= ridge_perp_pos]
        neg_cands = [x for x in all_end_cands if x['pos_perp'] < ridge_perp_pos]
        longest_m = max(x['length_m'] for x in all_end_cands)
        split = (pos_cands and neg_cands and longest_m < perp_extent * gsd * 0.6)

        print(f"\n  {end_name.upper()}: {len(all_end_cands)} candidates")
        print(f"    pos_cands({len(pos_cands)}): {[x['idx'] for x in pos_cands]}")
        print(f"    neg_cands({len(neg_cands)}): {[x['idx'] for x in neg_cands]}")
        print(f"    longest={longest_m:.1f}m, threshold={perp_extent*gsd*0.6:.1f}m → SPLIT={split}")
        for x in all_end_cands:
            side = "POS" if x['pos_perp'] >= ridge_perp_pos else "NEG"
            r1, c1, r2, c2 = x['seg']
            print(f"      i={x['idx']:3d} {side} len={x['length_m']:.1f}m s={x['score']:.2f} "
                  f"in={x['inside_frac']:.2f} along={x['pos_along']:.1f} perp={x['pos_perp']:.1f} "
                  f"({r1:.1f},{c1:.1f})-({r2:.1f},{c2:.1f})")

        if split:
            for label, slope_cands in [("POS", pos_cands), ("NEG", neg_cands)]:
                seg, length = wf._best_fragment(slope_cands, perp_dir, gsd, extend_to_span=True)
                seg_clip, len_clip = wf._clip_to_mask_extent(seg, mask, gsd)
                print(f"    → {label}: raw seg=({seg[0]:.1f},{seg[1]:.1f})-({seg[2]:.1f},{seg[3]:.1f}) len={length:.1f}m")
                print(f"           clipped=({seg_clip[0]:.1f},{seg_clip[1]:.1f})-({seg_clip[2]:.1f},{seg_clip[3]:.1f}) len={len_clip:.1f}m")
        else:
            seg, length = wf._best_fragment(all_end_cands, perp_dir, gsd, extend_to_span=True)
            seg_clip, len_clip = wf._clip_to_mask_extent(seg, mask, gsd)
            print(f"    → merged: raw seg=({seg[0]:.1f},{seg[1]:.1f})-({seg[2]:.1f},{seg[3]:.1f}) len={length:.1f}m")
            print(f"       clipped=({seg_clip[0]:.1f},{seg_clip[1]:.1f})-({seg_clip[2]:.1f},{seg_clip[3]:.1f}) len={len_clip:.1f}m")

    # Run actual classify
    result = _orig_classify(rmn_lines, rmn_scores, ndom, mask, plane_labels, planes,
                            ridge_angle, gsd=gsd)
    return result

wf.classify_rmn_lines = _debug_classify

# Now run the pipeline for S2
from test_validation_survey import process_building
import pandas as pd

BUILDINGS = {
    'S2': dict(
        id='S2', address='[Site S2]', city='NRW',
        state='Nordrhein-Westfalen',
        lat=None, lon=None, gsd=0.5,  # from sites.local.json
        merge_alkis=3.0, max_units=2,
    ),
}

gt_df = pd.read_csv(os.path.join(BASE, 'data', 'validation_buildings.csv'))
gt_row = gt_df[gt_df['id'] == 'S2']
if len(gt_row) > 0:
    gt_row = gt_row.iloc[0]
else:
    gt_row = None

process_building('S2', BUILDINGS['S2'], gt_row)
