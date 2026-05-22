# roof-measure-app

Open-data web tool for measuring residential roofs in NRW. The user enters an
address, sees the building on the aerial image, draws the roof outline, ridges
and valleys, and gets per roof section:

1. Slope area (m²)
2. Pitch (°, plane fit on the nDOM elevation model)
3. Aspect
4. Ridge and valley lengths (m)

There is no automatic detection in this tool; the user draws the lines. A rough
renovation cost estimate and storing annotations (with consent, as training data
for later models) are planned but not implemented. Annotations are currently not
stored.

Data: Geobasis NRW (DOP, nDOM). Other federal states with open data could be
added later.

## Layout

```
apps/
├── api/         FastAPI backend (Python 3.12), incl. Dockerfile and railway.json
└── web/         Next.js frontend (TypeScript, Leaflet)

packages/
└── geo/         planned shared geo package (not implemented yet)

infra/           docker-compose.yml (Postgres/PostGIS + Redis, for planned persistence)
docs/            ARCHITECTURE.md, DEPLOYMENT.md (target concept)
```

## Tech stack

| Layer    | Choice                                    | Notes                                         |
|----------|-------------------------------------------|-----------------------------------------------|
| Backend  | FastAPI + uvicorn                         | Reuses the existing Python geo code           |
| Frontend | Next.js 15 (App Router) + TypeScript      |                                               |
| Map      | Leaflet                                   | NRW tiles via the backend tile proxy          |
| Planned  | PostgreSQL + PostGIS, Redis               | Storing polygons, caching tiles               |
| Planned  | Railway (backend + DB), Vercel (frontend) | See `docs/DEPLOYMENT.md`                      |

## Local setup

```bash
# Backend
cd apps/api
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Frontend (new terminal)
cd apps/web
npm install
cp .env.example .env.local
npm run dev
```

The database and cache (`docker compose -f infra/docker-compose.yml up -d`) are
only needed for the planned persistence; the current API does not use them.

## Status

Implemented: backend (geocoding, NRW tile proxy, nDOM sampling, section split
and plane fit), frontend with the drawing tool.
Not implemented: persistence, accounts, cost estimate, deployment.

See `docs/ARCHITECTURE.md` for the target architecture.

## Licence

MIT, see [LICENSE](LICENSE).
