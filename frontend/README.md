# ResearchMind UI

Dark-theme web app for the ResearchMind API: start a run, follow it live, read
the report, explore the evidence graph, and check sources and usage.

React 18 + TypeScript, built with Vite; the evidence graph uses React Flow
(`@xyflow/react`). No router or state library: hash routes (`#/`,
`#/runs`, `#/runs/<id>[/report|/graph|/sources]`) and a small typed client
(`src/api.ts`).

## Run it

```bash
# 1. The API, from backend/ (it also runs a research worker)
uvicorn app.main:app --port 8000 --timeout-graceful-shutdown 5

# 2. The UI, from frontend/
npm install
npm run dev          # http://localhost:5173
```

The dev server proxies `/api`, `/health`, `/ready`, and `/docs` to the API,
so there is no CORS setup. Point it at another API with `API_TARGET`, e.g.
`API_TARGET=http://localhost:8080 npm run dev` for the Docker stack.

If the API has an `API_KEY`, paste it under Settings (top right); it is kept
in this browser's local storage and sent as `X-API-Key`. Live progress uses
the events stream through `fetch`, since `EventSource` cannot send that
header, and reconnects with `Last-Event-ID` after a drop.

## Check and build

```bash
npm run typecheck
npm run build        # static files in dist/
```

## Layout

| File | What it holds |
|---|---|
| `src/api.ts` | API client, error envelope, SSE over fetch |
| `src/types.ts` | Response shapes (mirrors `backend/app/schemas`) |
| `src/hooks.ts` | Hash router, polling loader, live event stream |
| `src/lib.ts` | Labels, colours, formatting, event descriptions |
| `src/ui.tsx`, `src/TopBar.tsx` | Shared pieces, top bar and settings |
| `src/pages/` | New research and runs, run progress, report, graph, sources and usage |
| `src/styles.css` | Theme tokens and components |
