# Frontend Development Guide

This guide reflects the current frontend, not the earlier expanded mock sitemap.

## Current Architecture

The app is a React + Vite single-page frontend with a compact route structure and one shared shell:

- `RootLayout` handles navigation, backend health polling, and the global visual frame
- `Dashboard` summarizes artifact and health data
- `Architecture` explains the GAN-vs-GAN concept
- `TrainingLab` visualizes training history and generator feedback
- `Analytics` compares baseline and robust metrics and renders confusion matrices
- `LiveDemo` uploads CSVs and calls the scoring endpoints

## Setup

```bash
npm install
npm run dev
```

For local backend integration:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Directory Notes

```text
src/app/
├── components/      reusable display and utility components
├── layouts/         shared app frame
├── lib/             API client, helpers, mock data
├── pages/           route-level screens
├── App.tsx          root app
└── routes.tsx       router definition
```

## Working With Data

### Real API first

Most of the important pages already use the real API client in `src/app/lib/api.ts`. Prefer extending that service layer instead of scattering raw `fetch` calls through page components.

### Graceful failure

The current UI intentionally tolerates missing backend data. When extending pages:

- keep loading states explicit
- catch failed API calls locally
- render partial UI when only some endpoints succeed

### CSV scoring flow

`LiveDemo` currently:

1. reads the uploaded CSV in-browser
2. parses numeric rows client-side
3. sends the rows to `/predict`
4. optionally sends an individual row to `/simulate_evasion`
5. exports summarized prediction results back to CSV

## When Updating Routes

If you add or rename routes, update all of the following together:

- `src/app/routes.tsx`
- navigation items in `src/app/layouts/RootLayout.tsx`
- `frontend/README.md`

## When Updating The Backend Contract

If backend responses change, update all of the following together:

- `src/app/lib/api.ts`
- any page-level response mapping logic
- `frontend/API_INTEGRATION.md`
- root `README-API-integration.md`

## Current Gaps Worth Knowing

- Several older page files still exist in `src/app/pages/`, but the active router uses only a smaller set.
- Mock data still exists for some presentation elements, but core analytics and live demo flows already use backend data.
- There is no dedicated frontend test setup documented in this repo yet.
