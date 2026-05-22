"""DOP-Tile-Proxy für NRW. Bedient XYZ-Tile-Requests aus zwei NRW-Endpunkten:

* z ≤ 16: WMTS — schnell, vom Provider gecached, aber nur bis ~1.5 m/px
* z > 16: WMS  — on-demand-render aus den nativen 10 cm/px-Daten

Tiles werden lokal als Datei gecached. Reicht für MVP; in Production durch
Cloudflare R2 / Redis ersetzbar.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Response

router = APIRouter(prefix="/tiles", tags=["tiles"])

CACHE_DIR = Path(os.getenv("TILE_CACHE_DIR", "/tmp/roof-measure-tiles"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Web-Mercator Extent ±20037508.342789244 m
EXTENT_3857 = 20037508.342789244

# NRW WMTS verschiebt die TileMatrix-IDs um 5 gegenüber XYZ
NRW_WMTS_OFFSET = 5
NRW_WMTS_BASE = (
    "https://www.wmts.nrw.de/geobasis/wmts_nw_dop/tiles/nw_dop/EPSG_3857_16"
)
NRW_WMS_BASE = "https://www.wms.nrw.de/geobasis/wms_nw_dop"


def _tile_bbox_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    tile_size = (EXTENT_3857 * 2) / (2**z)
    xmin = -EXTENT_3857 + x * tile_size
    xmax = -EXTENT_3857 + (x + 1) * tile_size
    ymax = EXTENT_3857 - y * tile_size
    ymin = EXTENT_3857 - (y + 1) * tile_size
    return xmin, ymin, xmax, ymax


async def _fetch_wmts(client: httpx.AsyncClient, z: int, x: int, y: int) -> bytes:
    matrix = f"{max(0, z - NRW_WMTS_OFFSET):02d}"
    url = f"{NRW_WMTS_BASE}/{matrix}/{x}/{y}"
    resp = await client.get(url, timeout=10.0)
    resp.raise_for_status()
    return resp.content


async def _fetch_wms(client: httpx.AsyncClient, z: int, x: int, y: int) -> bytes:
    xmin, ymin, xmax, ymax = _tile_bbox_3857(z, x, y)
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": "nw_dop_rgb",
        "styles": "",
        "format": "image/jpeg",
        "crs": "EPSG:3857",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "width": "256",
        "height": "256",
    }
    resp = await client.get(NRW_WMS_BASE, params=params, timeout=15.0)
    resp.raise_for_status()
    if resp.headers.get("content-type", "").startswith("text/xml"):
        raise HTTPException(status_code=502, detail=f"NRW WMS: {resp.text[:200]}")
    return resp.content


_locks: dict[str, asyncio.Lock] = {}


def _lock_for(key: str) -> asyncio.Lock:
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock


@router.get("/dop/{z}/{x}/{y}")
async def dop_tile(z: int, x: int, y: int) -> Response:
    if z < 8 or z > 21:
        raise HTTPException(status_code=404, detail="zoom out of supported range (8–21)")
    max_xy = 2**z
    if x < 0 or x >= max_xy or y < 0 or y >= max_xy:
        raise HTTPException(status_code=400, detail="tile coordinates out of range")

    use_wms = z > 16
    ext = "jpg" if use_wms else "png"
    cache_path = CACHE_DIR / "dop" / str(z) / str(x) / f"{y}.{ext}"
    media_type = "image/jpeg" if use_wms else "image/png"

    if cache_path.exists():
        return Response(
            content=cache_path.read_bytes(),
            media_type=media_type,
            headers={"X-Cache": "HIT", "Cache-Control": "public, max-age=604800"},
        )

    async with _lock_for(f"{z}/{x}/{y}"):
        if cache_path.exists():
            return Response(
                content=cache_path.read_bytes(),
                media_type=media_type,
                headers={"X-Cache": "HIT-AFTER-WAIT", "Cache-Control": "public, max-age=604800"},
            )

        try:
            async with httpx.AsyncClient() as client:
                data = await (
                    _fetch_wms(client, z, x, y) if use_wms else _fetch_wmts(client, z, x, y)
                )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_bytes(data)
        tmp_path.rename(cache_path)

    return Response(
        content=data,
        media_type=media_type,
        headers={"X-Cache": "MISS", "Cache-Control": "public, max-age=604800"},
    )
