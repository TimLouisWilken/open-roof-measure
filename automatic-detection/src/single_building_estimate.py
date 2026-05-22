"""
Single-building cost estimator from address or GPS coordinates.

Usage
-----
  python single_building_estimate.py --address "Musterstraße 1, 50667 Köln"
  python single_building_estimate.py --lat 50.938 --lon 6.960
  python single_building_estimate.py --address "..." --screenshot path/to/image.png

What it does
------------
1. Geocode address → UTM-32N coordinates (via Nominatim, free, no key needed)
2. Download 80×80 m nDOM tile from NRW Geobasis (0.5 m/px)
3. Download 80×80 m DOP tile (aerial photo, 10 cm/px)
4. Detect building footprint from nDOM threshold
5. Extract roof features (area, pitch, complexity)
6. Compute insulation cost estimate + energy savings

Works for single-family and semi-detached houses in NRW, Germany.
For other regions: same approach works with equivalent open-data DSM sources.
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from scipy.ndimage import binary_erosion, label as ndi_label, zoom

sys.path.insert(0, os.path.dirname(__file__))
try:
    from feature_extractor import extract_features
    from cost_model import unit_cost, energy_savings_kwh, energy_savings_eur, payback_years
except ImportError:
    pass
from building_detector import detect_roof

NDOM_URL = "https://www.wcs.nrw.de/geobasis/wcs_nw_ndom"
DOP_URL  = "https://www.wcs.nrw.de/geobasis/wcs_nw_dop"
NDOM_RES = 0.5   # m/px
DOP_RES  = 0.10  # m/px
TILE_M   = 80    # download 80×80 m around the building


def geocode(address):
    """Return (lat, lon) via Nominatim (OpenStreetMap, free, no key needed)."""
    url = "https://nominatim.openstreetmap.org/search"
    r = requests.get(url, params={"q": address, "format": "json", "limit": 1},
                     headers={"User-Agent": "roof-estimator/1.0"}, timeout=10)
    r.raise_for_status()
    results = r.json()
    if not results:
        raise ValueError(f"Address not found: {address}")
    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    return lat, lon


def latlon_to_utm32(lat, lon):
    """Convert WGS84 lat/lon to EPSG:25832 (UTM Zone 32N) easting/northing.
    Uses a pure-Python approximation to avoid pyproj PROJ DB issues.
    Accuracy: <1 m for Central Europe.
    """
    import math
    # WGS84 ellipsoid parameters
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 1 - (b / a) ** 2

    lon0 = math.radians(9.0)   # UTM Zone 32 central meridian
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)

    N = a / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
    T = math.tan(lat_r) ** 2
    C = e2 / (1 - e2) * math.cos(lat_r) ** 2
    A = math.cos(lat_r) * (lon_r - lon0)

    e4 = e2 ** 2; e6 = e2 ** 3
    M = a * ((1 - e2/4 - 3*e4/64 - 5*e6/256) * lat_r
             - (3*e2/8 + 3*e4/32 + 45*e6/1024) * math.sin(2*lat_r)
             + (15*e4/256 + 45*e6/1024) * math.sin(4*lat_r)
             - (35*e6/3072) * math.sin(6*lat_r))

    k0 = 0.9996
    easting  = k0 * N * (A + (1-T+C)*A**3/6
                         + (5-18*T+T**2+72*C-58*e2/(1-e2))*A**5/120) + 500000
    northing = k0 * (M + N*math.tan(lat_r)*(A**2/2 + (5-T+9*C+4*C**2)*A**4/24
                                              + (61-58*T+T**2+600*C-330*e2/(1-e2))*A**6/720))
    return easting, northing


def download_chip(url, coverage_id, e_center, n_center, half_m):
    e_min = int(e_center - half_m)
    e_max = int(e_center + half_m)
    n_min = int(n_center - half_m)
    n_max = int(n_center + half_m)
    endpoint = (f"{url}?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
                f"&COVERAGEID={coverage_id}"
                f"&SUBSET=x({e_min},{e_max})&SUBSET=y({n_min},{n_max})"
                f"&FORMAT=image/tiff")
    r = requests.get(endpoint, timeout=60)
    r.raise_for_status()
    if not r.ok or b"ExceptionReport" in r.content[:500]:
        raise RuntimeError(
            f"WCS returned error for {coverage_id}.\n"
            "  → Is the address in NRW (Nordrhein-Westfalen)?\n"
            "  → The free nDOM is only available for NRW. Other states: see README.\n"
            f"  Raw: {r.content[:200]}"
        )
    import tempfile, rasterio
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name
    with rasterio.open(tmp_path) as src:
        data = src.read()
    os.unlink(tmp_path)
    return data




def estimate(address=None, lat=None, lon=None, screenshot_path=None,
             output_dir="results", show_plot=True):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Geocode
    if address:
        print(f"  Geocoding: {address}")
        lat, lon = geocode(address)
        print(f"  → lat={lat:.5f}, lon={lon:.5f}")
    elif lat is None or lon is None:
        raise ValueError("Provide --address or --lat/--lon")

    # 2. UTM conversion
    e, n = latlon_to_utm32(lat, lon)
    print(f"  UTM32N: E={e:.0f}, N={n:.0f}")

    # 3. Download nDOM + DOP
    half = TILE_M / 2
    print(f"  Downloading {TILE_M}×{TILE_M} m nDOM chip …")
    ndom_arr = download_chip(NDOM_URL, "nw_ndom", e, n, half)[0]   # (H, W)

    print(f"  Downloading {TILE_M}×{TILE_M} m DOP chip …")
    dop_arr = download_chip(DOP_URL, "nw_dop", e, n, half)          # (4, H, W)

    # 4. Detect roof (ALKIS primary → NDVI fallback)
    e_min = e - half
    n_min = n - half
    mask, found, method = detect_roof(ndom_arr, dop_arr, e_min, n_min)
    print(f"  Building detector: {method}")
    if not found:
        print("  WARNING: No building found at this location.")
        print("  Check coordinates or try a slightly different address.")
        return None

    # 5. Extract features
    feats = extract_features(mask, ndom_arr, gsd=NDOM_RES)

    # 6. Cost estimate
    uc       = unit_cost(feats["mean_pitch_deg"], feats["complexity_index"])
    cost_est = uc * feats["dsm_area_m2"]
    e_kwh    = energy_savings_kwh(feats["dsm_area_m2"])
    e_eur    = energy_savings_eur(feats["dsm_area_m2"])
    pb       = payback_years(cost_est, feats["dsm_area_m2"])

    # 7. Print result
    print("\n" + "="*55)
    print("  ROOF INSULATION COST ESTIMATE")
    if address:
        print(f"  {address}")
    print("="*55)
    print(f"  Roof footprint (satellite view) : {feats['footprint_area_m2']:.0f} m²")
    print(f"  3-D roof area  (DSM-corrected)  : {feats['dsm_area_m2']:.0f} m²")
    print(f"  Mean pitch                      : {feats['mean_pitch_deg']:.1f}°")
    print(f"  Complexity index                : {feats['complexity_index']:.2f}")
    print(f"  No. of roof planes              : {feats['n_planes']}")
    print("-"*55)
    print(f"  Unit cost                       : {uc:.0f} €/m²")
    print(f"  ESTIMATED RENOVATION COST       : {cost_est:,.0f} €")
    print(f"  (±20% confidence interval:       {cost_est*0.80:,.0f} – {cost_est*1.20:,.0f} €)")
    print("-"*55)
    print(f"  Annual energy savings           : {e_kwh:.0f} kWh/yr")
    print(f"  Annual cost savings             : {e_eur:.0f} €/yr")
    print(f"  Simple payback period           : {pb:.1f} years")
    print("="*55)

    # 8. Save figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Panel A: aerial photo (RGB from RGBI)
    dop_small = np.stack([zoom(dop_arr[b], NDOM_RES / DOP_RES / 1, order=1)
                          for b in range(3)], axis=0)
    h, w = ndom_arr.shape
    dop_small = np.clip(dop_small[:, :h, :w].transpose(1, 2, 0), 0, 255).astype(np.uint8)

    axes[0].imshow(dop_small)
    axes[0].set_title("(a) Aerial image (DOP 10 cm)")
    axes[0].axis("off")
    if address:
        axes[0].set_xlabel(address[:60], fontsize=7)

    # Panel B: nDOM with detected roof
    im = axes[1].imshow(ndom_arr, cmap="terrain")
    axes[1].contour(mask, levels=[0.5], colors="red", linewidths=1.5)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Height above ground (m)")
    axes[1].set_title(f"(b) nDOM + detected roof\n"
                      f"Footprint: {feats['footprint_area_m2']:.0f} m²  |  "
                      f"3D: {feats['dsm_area_m2']:.0f} m²  |  "
                      f"Pitch: {feats['mean_pitch_deg']:.1f}°")
    axes[1].axis("off")

    # Panel C: cost summary bar
    ax = axes[2]
    categories = ["Naive\n(footprint only)", "This tool\n(DSM-corrected)"]
    naive_cost = 85 * feats["footprint_area_m2"]
    costs = [naive_cost, cost_est]
    bars = ax.bar(categories, [c / 1000 for c in costs],
                  color=["#d62728", "#2ca02c"], edgecolor="white", width=0.4)
    for bar, val in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"€{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Estimated renovation cost (k€)")
    ax.set_title(f"(c) Cost estimate\nPayback: {pb:.1f} yr  |  Savings: {e_eur:.0f} €/yr")
    ax.set_ylim(0, max(costs) / 1000 * 1.3)

    fig.suptitle("Satellite-Based Roof Insulation Cost Estimator", fontsize=13)
    fig.tight_layout()

    slug = (address or f"lat{lat:.3f}_lon{lon:.3f}").replace(" ", "_").replace(",", "")[:50]
    out_path = os.path.join(output_dir, f"estimate_{slug}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")

    return {
        "address": address,
        "lat": lat, "lon": lon,
        "cost_eur": round(cost_est, 0),
        "cost_low": round(cost_est * 0.80, 0),
        "cost_high": round(cost_est * 1.20, 0),
        "pitch_deg": feats["mean_pitch_deg"],
        "area_3d_m2": feats["dsm_area_m2"],
        "payback_yr": round(pb, 1),
        "energy_kwh_yr": round(e_kwh, 0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roof insulation cost estimator")
    parser.add_argument("--address",    type=str, help="Street address (in NRW, Germany)")
    parser.add_argument("--lat",        type=float)
    parser.add_argument("--lon",        type=float)
    parser.add_argument("--screenshot", type=str, help="Path to satellite screenshot (optional)")
    parser.add_argument("--output",     type=str, default="results")
    args = parser.parse_args()

    estimate(
        address=args.address,
        lat=args.lat,
        lon=args.lon,
        screenshot_path=args.screenshot,
        output_dir=args.output,
    )
