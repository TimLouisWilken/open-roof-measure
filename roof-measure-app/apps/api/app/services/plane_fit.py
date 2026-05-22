"""Plane-Fitting pro Dachflächen-Sektion.

Pro Sektion:
1. nDOM-Raster für die Polygon-Bbox einmal holen (WCS).
2. Innerhalb jeder Sektion ein 1×1 m Grid samplen.
3. Least-Squares-Plane z = ax + by + c.
4. Aus Plane: Pitch (Neigung gegen Horizontale), Aspect (Himmelsrichtung der
   Falllinie), Schräge-Fläche = 2D-Fläche / cos(pitch).
"""

from __future__ import annotations

import io
import math

import httpx
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from shapely.geometry import Point as ShPoint
from shapely.geometry import Polygon as ShPolygon

from app.services.sections import _latlon_to_utm

NDOM_WCS = "https://www.wcs.nrw.de/geobasis/wcs_nw_ndom"
NDOM_COVERAGE = "nw_ndom"
GRID_STEP_M = 0.5  # Abtastauflösung innerhalb jeder Sektion
MIN_SAMPLES = 5
INNER_BUFFER_M = 0.5  # Sektion-Inset: vermeidet Rand-Punkte am Dach↔Wand-Übergang
MIN_ROOF_HEIGHT_M = 1.5  # Punkte darunter sind sicher kein Dach
RANSAC_ITER = 80
RANSAC_INLIER_M = 0.25  # Z-Residuum bis das ein Punkt als Inlier zählt


async def _fetch_ndom_bbox(
    client: httpx.AsyncClient,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> bytes:
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": NDOM_COVERAGE,
        "subset": [f"x({xmin},{xmax})", f"y({ymin},{ymax})"],
        "format": "image/tiff",
    }
    resp = await client.get(NDOM_WCS, params=params, timeout=30.0)
    resp.raise_for_status()
    if not resp.headers.get("content-type", "").startswith("image/"):
        raise RuntimeError(f"nDOM WCS non-image: {resp.text[:200]}")
    return resp.content


def _fit_plane_lse(samples: list[tuple[float, float, float]]):
    """LSE: z = ax + by + c. Returns (a, b, c)."""
    if len(samples) < 3:
        return None
    A = np.array([[s[0], s[1], 1.0] for s in samples], dtype=np.float64)
    z = np.array([s[2] for s in samples], dtype=np.float64)
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(coeff[0]), float(coeff[1]), float(coeff[2])


def _fit_plane_ransac(samples: list[tuple[float, float, float]]):
    """RANSAC-Plane-Fit, robust gegen Ausreißer (Schornstein, Dachfenster, Rand)."""
    n = len(samples)
    if n < 5:
        return _fit_plane_lse(samples)

    rng = np.random.default_rng(42)
    best_inliers: list[tuple[float, float, float]] = []
    best_plane: tuple[float, float, float] | None = None

    for _ in range(RANSAC_ITER):
        idx = rng.choice(n, 3, replace=False)
        subset = [samples[i] for i in idx]
        plane = _fit_plane_lse(subset)
        if plane is None:
            continue
        a, b, c = plane
        inliers = [
            s for s in samples if abs(a * s[0] + b * s[1] + c - s[2]) < RANSAC_INLIER_M
        ]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = plane

    if len(best_inliers) < MIN_SAMPLES:
        return _fit_plane_lse(samples)
    # Final LSE auf den Inliern → robust + optimal innerhalb der Konsensmenge
    return _fit_plane_lse(best_inliers)


def _plane_to_pitch_aspect(a: float, b: float) -> tuple[float, float]:
    """Pitch (°) und Aspect (° von Nord, CW) aus den Plane-Koeffizienten."""
    # Normal-Vektor der Ebene z = ax + by + c: (-a, -b, 1) (zeigt nach oben)
    norm = math.sqrt(a * a + b * b + 1.0)
    cos_to_up = 1.0 / norm
    pitch = math.degrees(math.acos(cos_to_up))

    # Falllinien-Richtung (steepest descent) in horizontaler Projektion: (-a, -b)
    # UTM: +x=East, +y=North → bearing = atan2(east, north) = atan2(-a, -b)
    if abs(a) < 1e-9 and abs(b) < 1e-9:
        aspect = 0.0
    else:
        aspect = math.degrees(math.atan2(-a, -b)) % 360.0
    return pitch, aspect


def _aspect_label(deg: float) -> str:
    dirs = ["N", "NO", "O", "SO", "S", "SW", "W", "NW", "N"]
    idx = int(round(deg / 45.0))
    return dirs[idx]


async def analyze_sections(
    polygon_latlon: list[tuple[float, float]],
    sections_latlon: list[list[tuple[float, float]]],
) -> list[dict]:
    """Berechnet pitch/aspect/slope-area für jede Sektion.

    Args:
        polygon_latlon: Umriss-Polygon (für Bbox des nDOM-Abrufs).
        sections_latlon: Liste der Sub-Polygone in (lat, lon).

    Returns:
        Liste von Dicts pro Sektion mit pitch_deg/aspect_deg/area_m2_slope/...
    """
    if not sections_latlon:
        return []

    # Bbox in UTM für den nDOM-Abruf (ganzes Polygon einmalig)
    poly_utm = [_latlon_to_utm(lat, lon) for lat, lon in polygon_latlon]
    xs = [p[0] for p in poly_utm]
    ys = [p[1] for p in poly_utm]
    xmin, xmax = min(xs) - 2.0, max(xs) + 2.0
    ymin, ymax = min(ys) - 2.0, max(ys) + 2.0

    async with httpx.AsyncClient() as client:
        tiff_bytes = await _fetch_ndom_bbox(client, xmin, ymin, xmax, ymax)

    results: list[dict] = []
    with MemoryFile(io.BytesIO(tiff_bytes)) as memfile:
        with memfile.open() as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)

            for sec_latlon in sections_latlon:
                sec_utm = [_latlon_to_utm(lat, lon) for lat, lon in sec_latlon]
                if len(sec_utm) < 3:
                    results.append(_empty_result())
                    continue
                sec_poly = ShPolygon(sec_utm)
                if not sec_poly.is_valid:
                    sec_poly = sec_poly.buffer(0)
                area_2d = float(sec_poly.area)

                # Inner-Buffer: vermeidet Rand-Punkte (Wand-Übergang). Wenn die
                # Sektion zu klein wird, ohne Buffer arbeiten.
                sample_poly = sec_poly.buffer(-INNER_BUFFER_M)
                if sample_poly.is_empty or sample_poly.area < 0.5:
                    sample_poly = sec_poly

                samples: list[tuple[float, float, float]] = []
                sxmin, symin, sxmax, symax = sample_poly.bounds
                y = symin + GRID_STEP_M / 2
                while y < symax:
                    x = sxmin + GRID_STEP_M / 2
                    while x < sxmax:
                        if sample_poly.contains(ShPoint(x, y)):
                            try:
                                row, col = src.index(x, y)
                                if 0 <= row < src.height and 0 <= col < src.width:
                                    v = arr[row, col]
                                    if not np.isnan(v) and v >= MIN_ROOF_HEIGHT_M:
                                        samples.append((x, y, float(v)))
                            except Exception:
                                pass
                        x += GRID_STEP_M
                    y += GRID_STEP_M

                if len(samples) < MIN_SAMPLES:
                    results.append(_empty_result())
                    continue

                fit = _fit_plane_ransac(samples)
                if fit is None:
                    results.append(_empty_result())
                    continue
                a, b, _c = fit
                pitch, aspect = _plane_to_pitch_aspect(a, b)
                slope_area = area_2d / max(math.cos(math.radians(pitch)), 1e-6)
                heights = [s[2] for s in samples]
                results.append(
                    {
                        "pitch_deg": pitch,
                        "aspect_deg": aspect,
                        "aspect_label": _aspect_label(aspect),
                        "area_m2_slope": float(slope_area),
                        "mean_height_m": float(np.mean(heights)),
                        "min_height_m": float(np.min(heights)),
                        "max_height_m": float(np.max(heights)),
                        "n_samples": len(samples),
                    }
                )

    return results


def _empty_result() -> dict:
    return {
        "pitch_deg": None,
        "aspect_deg": None,
        "aspect_label": None,
        "area_m2_slope": None,
        "mean_height_m": None,
        "min_height_m": None,
        "max_height_m": None,
        "n_samples": 0,
    }
