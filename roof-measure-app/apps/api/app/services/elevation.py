"""nDOM-Sampling über NRW WCS.

NRW liefert das normalisierte Höhenmodell (Höhe über Boden) als GeoTIFF in
EPSG:25832 (ETRS89 / UTM-32N) mit 0.5 m Pixelauflösung. Wir holen ein einziges
TIFF für die BBox aller angefragten Punkte und samplen dann lokal.
"""

from __future__ import annotations

import io
from typing import Sequence

import httpx
import numpy as np
import rasterio
from pyproj import Transformer

NDOM_WCS = "https://www.wcs.nrw.de/geobasis/wcs_nw_ndom"
NDOM_COVERAGE = "nw_ndom"

# Wgs84 <-> UTM-32N. Cache pro process.
_transformer = Transformer.from_crs(4326, 25832, always_xy=True)


def _bbox_utm(lonlat_points: Sequence[tuple[float, float]], buffer_m: float = 5.0):
    xs: list[float] = []
    ys: list[float] = []
    for lon, lat in lonlat_points:
        x, y = _transformer.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    return (
        min(xs) - buffer_m,
        min(ys) - buffer_m,
        max(xs) + buffer_m,
        max(ys) + buffer_m,
    ), list(zip(xs, ys))


async def sample_ndom(points_latlon: Sequence[tuple[float, float]]) -> list[float | None]:
    """Sample nDOM heights (m) for each (lat, lon) point. None if out of bounds."""
    if not points_latlon:
        return []

    lonlat = [(lon, lat) for lat, lon in points_latlon]
    (xmin, ymin, xmax, ymax), utm_points = _bbox_utm(lonlat)

    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": NDOM_COVERAGE,
        "subset": [f"x({xmin},{xmax})", f"y({ymin},{ymax})"],
        "format": "image/tiff",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(NDOM_WCS, params=params)
        resp.raise_for_status()
        if not resp.headers.get("content-type", "").startswith("image/"):
            raise RuntimeError(f"NRW nDOM WCS returned non-image: {resp.text[:200]}")
        tiff_bytes = resp.content

    out: list[float | None] = []
    with rasterio.io.MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            for x, y in utm_points:
                # Bilinear sample: convert UTM to fractional pixel (row, col)
                row_f, col_f = src.index(x, y, op=lambda v: v)
                if (
                    row_f < 0
                    or row_f > src.height - 1
                    or col_f < 0
                    or col_f > src.width - 1
                ):
                    out.append(None)
                    continue
                r0, c0 = int(np.floor(row_f)), int(np.floor(col_f))
                r1, c1 = min(r0 + 1, src.height - 1), min(c0 + 1, src.width - 1)
                fr, fc = row_f - r0, col_f - c0
                v00, v01 = arr[r0, c0], arr[r0, c1]
                v10, v11 = arr[r1, c0], arr[r1, c1]
                v0 = v00 * (1 - fc) + v01 * fc
                v1 = v10 * (1 - fc) + v11 * fc
                v = v0 * (1 - fr) + v1 * fr
                out.append(None if np.isnan(v) else float(v))
    return out
