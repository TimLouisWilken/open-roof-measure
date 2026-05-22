# Deployment on Railway (concept)

Planned deployment, not set up yet. Parts of it refer to components that do not
exist in the repository (Alembic migrations, `app.jobs.anonymize`); the current
API needs neither Postgres nor Redis. `apps/api/Dockerfile` and
`apps/api/railway.json` exist.

## Services

Railway project with three services:

1. **api**: Python backend (`apps/api/Dockerfile`)
2. **postgres**: Railway Postgres add-on (with PostGIS extension)
3. **redis**: Railway Redis add-on

The frontend (`apps/web`) would be deployed **separately on Vercel** (best Next.js integration; Railway also works but builds more slowly).

## First deploy

```bash
# Backend
cd apps/api
railway link              # interactive: select project
railway up                # builds the Dockerfile and deploys
# planned, no Alembic setup in the repo yet:
# railway run alembic upgrade head

# Frontend
cd apps/web
vercel link
vercel --prod
```

## Environment variables

### Backend (Railway)

Names as in `apps/api/.env.example` and `app/config.py`:

```
DATABASE_URL=postgresql+asyncpg://...   # from the Postgres add-on (planned)
REDIS_URL=redis://...                   # from the Redis add-on (planned)
NRW_WCS_DOP=https://www.wcs.nrw.de/geobasis/wcs_nw_dop
NRW_WCS_NDOM=https://www.wcs.nrw.de/geobasis/wcs_nw_ndom
NRW_WFS_ALKIS=https://www.wfs.nrw.de/geobasis/wfs_nw_alkis_vereinfacht
NOMINATIM_BASE=https://nominatim.openstreetmap.org
NOMINATIM_USER_AGENT=roof-measure-app/0.1 (kontakt@example.com)
CORS_ORIGINS=https://roof-measure-app.vercel.app
LOG_LEVEL=info
```

### Frontend (Vercel)

```
NEXT_PUBLIC_API_URL=https://api.up.railway.app
```

## GDPR setup

- Database region: choose an EU region in the Railway settings (check Railway's current region names)
- Vercel region: **Frankfurt (fra1)** as primary region
- Privacy policy and legal notice (Impressum) as static pages in the frontend
- Anonymisation cron (planned): GitHub Action running `railway run python -m app.jobs.anonymize`

## Cost estimate (as of 2026)

| Service     | Plan        | Cost               |
|-------------|-------------|--------------------|
| Railway     | Hobby       | ~€5/month (incl. 500 h compute, Postgres, Redis) |
| Vercel      | Hobby       | €0                 |
| Cloudflare  | R2 (tiles)  | €0 (under 10 GB)   |
| Cloudflare  | DNS + CDN   | €0                 |
| Domain      | .de via INWX| ~€6/year           |
| **Total**   |             | **~€6/month**      |
