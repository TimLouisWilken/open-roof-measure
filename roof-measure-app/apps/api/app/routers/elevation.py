from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.elevation import sample_ndom

router = APIRouter(prefix="/elevation", tags=["elevation"])


class SampleRequest(BaseModel):
    # Liste von [lat, lon] Paaren.
    points: list[tuple[float, float]] = Field(..., min_length=1, max_length=500)


class SampleResponse(BaseModel):
    # Höhen in Metern über Boden (nDOM). None wenn außerhalb der NRW-Abdeckung.
    heights: list[float | None]


@router.post("/sample", response_model=SampleResponse)
async def sample(req: SampleRequest) -> SampleResponse:
    try:
        heights = await sample_ndom(req.points)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"nDOM error: {exc}") from exc
    return SampleResponse(heights=heights)
