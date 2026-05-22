"""Adress-Geocoding über Nominatim (OpenStreetMap)."""

import httpx
from pydantic import BaseModel

from app.config import get_settings


class GeocodeResult(BaseModel):
    query: str
    display_name: str
    lat: float
    lon: float
    state: str | None = None
    postcode: str | None = None
    country_code: str | None = None


async def geocode_address(query: str) -> GeocodeResult | None:
    settings = get_settings()
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": "1",
        "countrycodes": "de",
        "limit": "1",
    }
    headers = {"User-Agent": settings.nominatim_user_agent}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{settings.nominatim_base}/search", params=params, headers=headers)
        resp.raise_for_status()
        items = resp.json()

    if not items:
        return None

    item = items[0]
    address = item.get("address", {})
    return GeocodeResult(
        query=query,
        display_name=item["display_name"],
        lat=float(item["lat"]),
        lon=float(item["lon"]),
        state=address.get("state"),
        postcode=address.get("postcode"),
        country_code=address.get("country_code"),
    )
