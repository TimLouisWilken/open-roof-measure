# Architecture (target concept)

This document describes the planned target architecture. Most of it is not
implemented: there is no database, no accounts, no `/buildings`, `/measure`,
`/annotations` or `/snap` endpoint, no PDF report and no anonymisation job. The
current implementation is described in the top-level README and in
`apps/api/README.md` / `apps/web/README.md`.

## Overview

```
┌──────────────────────────┐    ┌──────────────────────────────┐
│   Browser (Next.js)      │    │   FastAPI Backend            │
│                          │    │                              │
│  ┌────────────────────┐  │    │  ┌────────────────────────┐  │
│  │ Leaflet map        │  │    │  │ /geocode               │  │
│  │ + drawing layer    │◄─┼────┼──► /buildings/{lat}/{lon} │  │
│  └────────────────────┘  │    │  │ /tiles/dop/{z}/{x}/{y} │  │
│                          │    │  │ /tiles/ndom/{z}/{x}/{y}│  │
│  ┌────────────────────┐  │    │  │ /measure               │  │
│  │ step-by-step flow  │  │    │  │ /annotations           │  │
│  │ ridge → eaves …    │  │    │  └────────────────────────┘  │
│  └────────────────────┘  │    │            │                 │
└──────────────────────────┘    │            ▼                 │
                                │  ┌────────────────────────┐  │
                                │  │ packages/geo           │  │
                                │  │ - Pitch (RANSAC)       │  │
                                │  │ - area calculation     │  │
                                │  │ - projections UTM↔WGS  │  │
                                │  └────────────────────────┘  │
                                │            │                 │
                                │   ┌────────┴────────┐        │
                                │   ▼                 ▼        │
                                │ ┌──────────┐  ┌──────────┐   │
                                │ │ Postgres │  │ Redis    │   │
                                │ │ +PostGIS │  │ (Tiles)  │   │
                                │ └──────────┘  └──────────┘   │
                                │       │                      │
                                │       ▼                      │
                                │  ┌────────────────────────┐  │
                                │  │ NRW Geobasis WCS/WFS   │  │
                                │  │ (DOP, nDOM, ALKIS)     │  │
                                │  └────────────────────────┘  │
                                └──────────────────────────────┘
```

## Data model (Postgres / PostGIS)

```
users
  id            uuid PK
  email         text unique
  created_at    timestamptz

addresses
  id            uuid PK
  query         text              -- original user input
  lat, lon      double precision
  utm_x, utm_y  double precision
  alkis_id      text              -- ALKIS polygon id, if found
  alkis_geom    geometry(Polygon, 25832)

buildings
  id            uuid PK
  address_id    uuid FK
  footprint     geometry(Polygon, 25832)
  state         text              -- nrw, ni, by, ...

annotations
  id            uuid PK
  user_id       uuid FK
  building_id   uuid FK
  step          text              -- 'first' | 'traufe' | 'ortgang' | 'aufbauten'
  geom          geometry(Geometry, 25832)
  created_at    timestamptz

measurements
  id            uuid PK
  building_id   uuid FK
  pitch_deg     double precision
  area_m2       double precision
  ridge_m       double precision
  eaves_m       double precision
  verge_m       double precision
  computed_at   timestamptz

cost_estimates
  id              uuid PK
  measurement_id  uuid FK
  scenario        text             -- 'dämmung_aufdach', 'eindeckung_ton', ...
  cost_low_eur    integer
  cost_high_eur   integer
  computed_at     timestamptz
```

## Drawing workflow (frontend, planned)

1. **Address input** → backend returns ALKIS polygon and tile URLs for DOP + nDOM
2. **Map centred on the ALKIS polygon** (semi-transparent overlay as "your house")
3. **Step 1: ridge** — user clicks two endpoints, snapping to nDOM height steps
4. **Step 2: eaves** — user clicks four corners (two lines)
5. **Step 3: verges** — backend proposes the verges from ridge and eaves endpoints, user can adjust
6. **Step 4: superstructures (optional)** — user clicks PV, roof windows, dormers, chimneys
7. **Compute** → backend returns pitch, area, lengths, cost estimate
8. **PDF report** → download

The implemented flow differs: roof type → outline polygon → ridges → valleys →
per-section plane fit (see top-level README).

## Snapping (planned)

- **nDOM snap**: on mouse hover the frontend calls `/snap?lat={}&lon={}&radius=2m`, which returns the nearest height-step edge
- **Endpoint snap**: existing endpoints attract the cursor (within ~5 px)
- **Parallel snap**: while drawing the eaves, parallelism to the ridge is hinted (no constraint)

## Anonymisation (planned)

- Right after a measurement: address, lat/lon and alkis_id stay in the user account
- 30 days after the user's last login: a cron job removes `addresses.query` and `addresses.lat/lon` and keeps only `alkis_geom` and the annotations
- After anonymisation the annotations are no longer personal data and can be released for training

## Scaling beyond NRW

NRW is the first region. Any federal state with an open WCS/WFS can be added as an adapter:

```python
packages/geo/providers/
├── base.py              GeoProvider-Interface
├── nrw.py               WCS/WFS NRW (default)
├── niedersachsen.py     WCS Niedersachsen
└── bayern.py            WCS Bayern (später)
```

Providers would be selected by the `state` field in `addresses` (derived from the geocoding response).
