from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.plane_fit import analyze_sections
from app.services.sections import compute_sections

router = APIRouter(prefix="/roof", tags=["roof"])


class SectionsRequest(BaseModel):
    polygon: list[tuple[float, float]] = Field(..., min_length=3)
    ridges: list[list[tuple[float, float]]] = []
    valleys: list[list[tuple[float, float]]] = []


class Section(BaseModel):
    points: list[tuple[float, float]]
    area_m2_2d: float
    # Phase 3: Plane-Fit aus nDOM
    pitch_deg: float | None = None
    aspect_deg: float | None = None
    aspect_label: str | None = None
    area_m2_slope: float | None = None
    mean_height_m: float | None = None
    min_height_m: float | None = None
    max_height_m: float | None = None
    n_samples: int = 0


class SectionsResponse(BaseModel):
    sections: list[Section]


@router.post("/sections", response_model=SectionsResponse)
async def sections(req: SectionsRequest) -> SectionsResponse:
    try:
        geo = compute_sections(req.polygon, req.ridges, req.valleys)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"section split failed: {exc}") from exc

    # Plane-Fit pro Sektion aus dem nDOM
    try:
        analyses = await analyze_sections(req.polygon, [s["points"] for s in geo])
    except Exception as exc:
        # Fehler im plane-fit blockt die Sektionen nicht — geo wird ohne fit ausgeliefert.
        analyses = [{} for _ in geo]
        # Log: in einer prod env loggt man das. Hier: nicht hart fehlschlagen.

    merged: list[Section] = []
    for g, a in zip(geo, analyses):
        merged.append(Section(**{**g, **a}))
    return SectionsResponse(sections=merged)
