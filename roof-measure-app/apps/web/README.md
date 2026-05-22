# roof-measure-web

Next.js frontend for the roof measurement tool.

## Setup

```bash
npm install
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000. The backend must be running at `NEXT_PUBLIC_API_URL`
(`.env.example`: `http://127.0.0.1:8000`).

## Structure

```
app/
├── layout.tsx               Root layout (Tailwind, metadata)
├── page.tsx                 Landing page with address input
└── messen/page.tsx          Map view after geocoding

components/
├── drawing-flow.tsx         Client-only wrapper
└── drawing-flow-impl.tsx    Drawing flow with Leaflet

lib/
└── api.ts                   Backend client
```

## Drawing flow

1. Select roof type (gable / hip / shed / flat)
2. Click the roof outline (polygon)
3. Click ridge lines (several for L- and T-shapes)
4. Click valleys (optional)
5. The backend (`POST /roof/sections`) splits the polygon into sections and
   returns pitch, aspect and slope area per section
