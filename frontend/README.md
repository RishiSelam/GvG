# GvG Defense Frontend

This frontend is the current command-center UI for the GvG IDS project. It is no longer just a static concept demo: the app is wired to the real FastAPI backend and renders live artifacts whenever the API is available.

## Current App Scope

The shipped experience is focused around five routes:

- `Dashboard`
- `Architecture`
- `Training Lab`
- `Analytics`
- `Live Demo`

These pages map directly to the current router in `src/app/routes.tsx`.

## What It Shows Today

When the backend is online, the frontend reads:

- training manifest metadata
- evaluation metrics
- baseline and robust loss histories
- generator state and feedback
- EDA summaries and plot references
- confusion matrices
- live prediction and evasion-simulation responses

When the backend is offline, the UI keeps loading safely and surfaces health state instead of crashing.

## Technology Stack

- React 18
- TypeScript
- React Router 7
- Vite
- Tailwind CSS v4
- Motion
- Recharts
- Lucide React
- Radix UI primitives
- Sonner

## Local Development

Install dependencies and start the app:

```bash
npm install
npm run dev
```

To connect to the local Python API:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

Build for production:

```bash
npm run build
```

## Project Structure

```text
src/
├── app/
│   ├── components/
│   ├── layouts/
│   ├── lib/
│   │   ├── api.ts
│   │   ├── mockData.ts
│   │   └── utils.ts
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── Architecture.tsx
│   │   ├── TrainingLab.tsx
│   │   ├── Analytics.tsx
│   │   ├── LiveDemo.tsx
│   │   └── NotFound.tsx
│   ├── layouts/RootLayout.tsx
│   └── routes.tsx
└── styles/
```

## Current API Contract

The frontend currently uses these backend calls from `src/app/lib/api.ts`:

- `GET /`
- `POST /predict`
- `POST /simulate_evasion`
- `GET /artifacts/manifest`
- `GET /artifacts/metrics`
- `GET /artifacts/training-history`
- `GET /artifacts/generator`
- `GET /artifacts/eda`
- `GET /artifacts/confusion-matrices`
- `GET /artifacts/eda/plots/{filename}`

If you change backend routes, update `src/app/lib/api.ts` and the documentation in `API_INTEGRATION.md` together.
