# Wireframe Line Classification — Version Changes

Changes to `src/wireframe_fusion.py` function `classify_rmn_lines()` and overlay generation.

This log covers V1–V6. The code has since moved on (V7 mask-boundary adjustment,
V10 `classify_rmn_lines_v10()`, which derives the ortgang from ridge and eaves
endpoints). The results in the top-level README come from the current version,
so the V5/V6 numbers below differ from them. The overlay images named in the
headings are not part of the repository.

---

## V1 — Mean-Projection Merge (`fusion_overlay_v1_cornerclip.jpg`)

**Approach:** Group collinear RoofMapNet fragments, merge via mean-projection onto the group's dominant direction. Extend merged line to cover the full span of all fragments in the group.

**Issues:**
- Mean-projection creates synthetic lines that deviate from actual RoofMapNet detections
- Extension along group direction amplifies errors from offset fragments

---

## V2 — Argmin/Argmax Endpoints (`fusion_overlay_v2_exactlines.jpg`)

**Approach:** Instead of mean-projection, pick the extreme endpoints (argmin/argmax along merge direction) from all fragments in the group.

**Issues:**
- Picking endpoints from DIFFERENT fragments creates diagonal lines (e.g., start of fragment A to end of fragment B)
- S2 eaves (traufe) showed clearly wrong diagonal lines

---

## V3 — Longest Fragment + Ridge-Direction Extension (`fusion_overlay_v3_bestfrag.jpg`)

**Approach:** Select the longest fragment per group, then extend its endpoints along the `ridge_dir`/`perp_dir` axis to cover the full group span.

**Issues:**
- Extension along `ridge_dir`/`perp_dir` deviates from the fragment's actual RoofMapNet angle
- Lines appear slightly rotated compared to the original detections
- Line assignment produced slanted lines that deviate noticeably from the RoofMapNet detections

---

## V4 — Longest Fragment + Own-Direction Extension (`fusion_overlay_v4_owndir.jpg`)

**Approach:** Select the longest fragment, extend along its OWN direction (preserving the RoofMapNet angle) to cover the group span.

**Changes from V3:**
- Extension direction changed from `ridge_dir`/`perp_dir` to the fragment's own unit vector `d`

**Issues:**
- Works well for S1 but problematic for S2/S3
- `is_parallel` threshold too loose (35°) — lines at 31° from ridge accepted as parallel
- No explicit `is_perpendicular` check — ortgang accepted anything not parallel (>35°)
- Ridge selection by height only — diagonal fragment with 0.1m higher height won over aligned fragment
- Groups of 10–23 fragments cause excessive extension

---

## V5 — Pure Longest Fragment, No Extension (`fusion_overlay_v5_exactfrag.jpg`)

**Approach:** Select the longest fragment per group, use its EXACT original endpoints (no extension at all).

**Changes from V4:**
- `_best_fragment()` returns exact RoofMapNet endpoints, no extension
- `is_parallel` threshold tightened: `angle_diff < 20` (was 35)
- New `is_perpendicular` field: `angle_diff > 70`
- Ridge selection uses angle penalty: `mean_h - angle_diff * 0.15`
- Ortgang selection requires `is_perpendicular` instead of `not is_parallel`

**Results:**
- S1: ridge 31.2m, traufe 31.0+31.4m, ortgang 15.5+15.7m
- S2: ridge 14.2m, traufe 15.7+14.9m, ortgang 7.4+10.5m
- S3: ridge 16.5m, traufe 14.6+15.3m, ortgang 10.2+5.7m

**Issues:**
- S2: bottom ortgang 10.5m extends far from building (fragment too far south)
- S3: top ortgang only 5.7m (fragment too short, doesn't span full traufe range)
- No corner fitting — lines don't meet at building corners

---

## V6 — Corner Snap + Mask Constraint (`fusion_overlay_v6_cornersnap.jpg`)

**Approach:** After fragment selection, snap line endpoints so ortgang/traufe meet at corners, then constrain all lines to the mask boundary.

**Changes from V5:**

### 1. Ortgang distance limit
- Added `max_ortgang_along = along_extent / 2 + 3.0 / gsd`
- Ortgang candidates filtered: `sign * pos_along < max_ortgang_along`
- Prevents selecting fragments beyond the building footprint

### 2. Corner snap (extend AND clip)
- New `_snap_endpoints()` function snaps both endpoints of a line along its own direction to match a target projection range
- Ortgang snapped to traufe perpendicular range → ortgang spans from traufe to traufe
- Traufe snapped to ortgang along range → traufe spans from ortgang to ortgang
- Snap ranges capped to mask extent + 1m buffer

### 3. Mask boundary constraint
- After snapping, all accepted lines constrained to mask extent + 0.5m buffer
- Endpoint clipping along the line's own direction (when dot product ≥ 0.3)
- Center translation along the mask axis (when dot product < 0.3, i.e., line perpendicular to clipping direction)

**Results:**
- S1: ridge 31.2m, traufe 31.4+31.4m, ortgang 15.2+15.2m
- S2: ridge 14.2m, traufe 14.4+14.4m, ortgang 10.5+10.5m
- S3: ridge 15.6m, traufe 15.6+15.6m, ortgang 11.0+11.0m

**Improvements vs V5:**
- S2 bottom ortgang now stays within the building footprint; both ortgang lines symmetric at 10.5 m
- S3 short ortgang extended from 5.7m to 11.0m (both now symmetric)
- All lines now symmetric per role (both traufe equal, both ortgang equal)
- Lines constrained to building footprint
