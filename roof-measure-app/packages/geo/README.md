# packages/geo

Planned, not implemented. This folder contains no code yet; the geo logic
currently lives in `apps/api/app/services/` and in `automatic-detection/src/`.

Intended contents (reusable geo logic extracted from `automatic-detection/src/`):

- `pitch.py`: roof pitch from an nDOM polygon (RANSAC plane fit), derived from `roof_analyzer.py`
- `area.py`: slope area from polygon and pitch
- `projection.py`: UTM 32N ↔ WGS84 conversion
- `providers/`: WCS/WFS adapters per federal state
  - `nrw.py`: DOP/nDOM/ALKIS for NRW (refactor of `nrw_downloader.py`)

The backend (`apps/api`) would then import it as a local dependency:

```toml
# apps/api/pyproject.toml
dependencies = [
    "roof-measure-geo @ file:../../packages/geo",
]
```
