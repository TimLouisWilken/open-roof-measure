from fastapi import APIRouter, HTTPException, Query

from app.services.geocode import GeocodeResult, geocode_address

router = APIRouter(prefix="/geocode", tags=["geocoding"])


@router.get("", response_model=GeocodeResult)
async def geocode(q: str = Query(..., min_length=3, description="Adresse")) -> GeocodeResult:
    result = await geocode_address(q)
    if result is None:
        raise HTTPException(status_code=404, detail="Adresse nicht gefunden")
    return result
