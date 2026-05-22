"""
Download real aerial imagery and height data from Geobasis NRW.

Sources (free, no registration, open data licence dl-de/by-2-0):
  DOP  – Digitales Orthophoto 10 cm/px, 4-band RGBI
         https://www.wcs.nrw.de/geobasis/wcs_nw_dop
  nDOM – normiertes Digitales Oberflächenmodell 0.5 m/px  (buildings above ground)
         https://www.wcs.nrw.de/geobasis/wcs_nw_ndom

All coordinates: EPSG 25832 (UTM Zone 32N)

Study areas: 8 residential tiles across NRW chosen to represent
  typical German detached/semi-detached housing stock
  (pitch range ~20°–55°, various roof complexity levels)
"""

import os, time, requests
import numpy as np
import rasterio
from scipy.ndimage import label as ndi_label, zoom

# ── NRW WCS endpoints ────────────────────────────────────────────────────────
DOP_URL  = "https://www.wcs.nrw.de/geobasis/wcs_nw_dop"
NDOM_URL = "https://www.wcs.nrw.de/geobasis/wcs_nw_ndom"

# ── Study areas: (name, E_min, N_min, E_max, N_max) in EPSG:25832 ───────────
# Each tile is 400×400 m → gives ~100-300 buildings per tile
STUDY_AREAS = [
    ("bonn_endenich",       367800, 5621000, 368200, 5621400),
    ("bonn_kessenich",      369000, 5619500, 369400, 5619900),
    ("koeln_suelz",         355200, 5643000, 355600, 5643400),
    ("koeln_nippes",        356500, 5648000, 356900, 5648400),
    ("duesseldorf_benrath", 354800, 5667500, 355200, 5667900),
    ("muenster_centrum",    399500, 5756000, 399900, 5756400),
    ("dortmund_hombruch",   411500, 5703000, 411900, 5703400),
    ("aachen_burtscheid",   295800, 5624500, 296200, 5624900),
]

NDOM_RES = 0.5   # metres per pixel (native nDOM resolution)
DOP_RES  = 0.10  # metres per pixel (native DOP resolution)


def _wcs_get(url, coverage_id, x_min, x_max, y_min, y_max, fmt="image/tiff"):
    params = {
        "SERVICE":    "WCS",
        "VERSION":    "2.0.1",
        "REQUEST":    "GetCoverage",
        "COVERAGEID": coverage_id,
        "SUBSET":     [f"x({x_min},{x_max})", f"y({y_min},{y_max})"],
        "FORMAT":     fmt,
    }
    # requests doesn't handle duplicate keys → build URL manually
    base = f"{url}?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
    base += f"&COVERAGEID={coverage_id}"
    base += f"&SUBSET=x({x_min},{x_max})&SUBSET=y({y_min},{y_max})"
    base += f"&FORMAT={fmt}"
    r = requests.get(base, timeout=60)
    r.raise_for_status()
    if b"ExceptionReport" in r.content[:500]:
        raise RuntimeError(f"WCS error: {r.content[:300]}")
    return r.content


def download_tile(name, e_min, n_min, e_max, n_max, out_dir, skip_existing=True):
    """Download DOP and nDOM for one study area tile."""
    os.makedirs(out_dir, exist_ok=True)
    dop_path  = os.path.join(out_dir, f"{name}_dop.tif")
    ndom_path = os.path.join(out_dir, f"{name}_ndom.tif")

    if skip_existing and os.path.exists(dop_path) and os.path.exists(ndom_path):
        print(f"  [{name}] already cached – skipping download")
        return dop_path, ndom_path

    print(f"  [{name}] downloading DOP  ({e_max-e_min}×{n_max-n_min} m) …", end=" ", flush=True)
    t0 = time.time()
    dop_bytes = _wcs_get(DOP_URL, "nw_dop", e_min, e_max, n_min, n_max)
    with open(dop_path, "wb") as f:
        f.write(dop_bytes)
    print(f"{len(dop_bytes)/1e6:.1f} MB  ({time.time()-t0:.1f}s)")

    print(f"  [{name}] downloading nDOM …", end=" ", flush=True)
    t0 = time.time()
    ndom_bytes = _wcs_get(NDOM_URL, "nw_ndom", e_min, e_max, n_min, n_max)
    with open(ndom_path, "wb") as f:
        f.write(ndom_bytes)
    print(f"{len(ndom_bytes)/1e6:.1f} MB  ({time.time()-t0:.1f}s)")

    return dop_path, ndom_path


def extract_buildings(dop_path, ndom_path,
                      min_height_m=3.0,
                      min_area_m2=40.0,
                      max_area_m2=400.0,   # single/semi-detached: <400 m² footprint
                      chip_pad_m=3.0):
    """
    Segment buildings from nDOM threshold + connected components.
    Resample DOP to nDOM resolution for aligned chips.

    Returns list of (mask, ndom_chip, dop_chip) at NDOM_RES m/px.
    """
    with rasterio.open(ndom_path) as src:
        ndom = src.read(1).astype(np.float32)
        res  = src.res[0]   # should be 0.5 m

    with rasterio.open(dop_path) as src:
        dop_full = src.read()   # (4, H_dop, W_dop), uint8

    # resample DOP to nDOM resolution via zoom
    scale = res / DOP_RES   # e.g. 0.5/0.1 = 5 → downsample by 5
    dop_small = np.stack([
        zoom(dop_full[b], 1.0 / scale, order=1)
        for b in range(dop_full.shape[0])
    ], axis=0)   # (4, H_ndom, W_ndom)

    # ensure matching shape (rounding artefacts)
    h, w = ndom.shape
    dop_small = dop_small[:, :h, :w]

    # building mask: nDOM > threshold
    bld_mask = ndom > min_height_m

    # connected components
    labeled, n_comp = ndi_label(bld_mask)

    chips = []
    pad_px = int(chip_pad_m / res)
    min_px = int(min_area_m2 / res ** 2)
    max_px = int(max_area_m2 / res ** 2)

    for bid in range(1, n_comp + 1):
        bmask = labeled == bid
        n_px  = bmask.sum()
        if not (min_px <= n_px <= max_px):
            continue

        rows = np.where(bmask.any(axis=1))[0]
        cols = np.where(bmask.any(axis=0))[0]
        r0 = max(0, rows[0]  - pad_px)
        r1 = min(h, rows[-1] + pad_px)
        c0 = max(0, cols[0]  - pad_px)
        c1 = min(w, cols[-1] + pad_px)

        chip_mask = bmask[r0:r1, c0:c1]
        chip_ndom = ndom[r0:r1, c0:c1]
        chip_dop  = dop_small[:, r0:r1, c0:c1].transpose(1, 2, 0)  # (H,W,4)

        chips.append((chip_mask, chip_ndom, chip_dop))

    return chips, res


def download_all(data_dir, areas=None):
    """Download all study areas, return list of (mask, ndom, dop) chips."""
    if areas is None:
        areas = STUDY_AREAS
    raw_dir = os.path.join(data_dir, "nrw_raw")
    os.makedirs(raw_dir, exist_ok=True)

    all_chips = []
    for (name, e_min, n_min, e_max, n_max) in areas:
        try:
            dop_path, ndom_path = download_tile(name, e_min, n_min, e_max, n_max, raw_dir)
            chips, res = extract_buildings(dop_path, ndom_path)
            print(f"  [{name}] extracted {len(chips)} buildings")
            all_chips.extend(chips)
        except Exception as ex:
            print(f"  [{name}] FAILED: {ex}")

    print(f"\n  Total buildings from real NRW data: {len(all_chips)}")
    return all_chips
