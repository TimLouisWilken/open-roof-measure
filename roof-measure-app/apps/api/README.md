# roof-measure-api

FastAPI backend for the roof measurement tool.

## Setup

```bash
# Python 3.12+ required
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env

uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The server runs on http://127.0.0.1:8000, OpenAPI docs at `/docs`.

Postgres and Redis (`cd ../../infra && docker compose up -d`) are only needed
for the planned persistence; the current endpoints do not use them.

## Endpoints

- `GET /health`: liveness check
- `GET /geocode?q=<address>`: Nominatim geocoding of a German address
- `GET /tiles/dop/{z}/{x}/{y}`: NRW DOP tile proxy (WMTS up to z16, WMS above)
- `POST /elevation/sample`: sample nDOM heights at given points
- `POST /roof/sections`: split the drawn polygon by ridges/valleys and fit a plane per section

## Planned endpoints

- `GET /buildings/{lat}/{lon}`: fetch ALKIS polygon
- `GET /tiles/ndom/{z}/{x}/{y}`: nDOM tile (proxy + cache)
- `POST /annotations`: store user annotation
- `POST /measure`: compute measurement from annotation

## Tests

```bash
pytest
```
