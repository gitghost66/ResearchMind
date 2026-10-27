# ResearchMind — Agent Working Rules

## Rules

1. Read the PRD (`ResearchMind_PRD.md`) before starting any task; work only on the current milestone. (`prompt.txt` is an identical copy kept for reference — treat the `.md` as canonical.)
2. Backend contains no prompt text or agent logic; agents package contains no HTTP or DB-session code (it talks to storage through interfaces passed in).
3. Every function that calls an LLM returns a validated Pydantic object.
4. Never hard-code model names, API keys, or thresholds; use `Settings`.
5. Write tests alongside code; a milestone is done only when its acceptance criteria pass.
6. Before finishing a task, run ruff, mypy, and pytest and report results.
7. Use mocks for LLM/search in tests; real API calls only in clearly marked optional integration tests.
8. Ask before adding a new dependency; prefer the stack listed below.
9. Commit per milestone with a descriptive message.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI + Pydantic v2 |
| Orchestration | LangGraph |
| LLM (primary) | Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) via `google-genai` |
| LLM (fallback) | Any OpenAI-compatible endpoint (OpenRouter / Groq / local) over `httpx` — satisfies the PRD's two-provider requirement with no extra dependency |
| Embeddings | Gemini `gemini-embedding-001` at 768d (`output_dimensionality`) via `google-genai` — `text-embedding-004` was shut down 2026-01-14 |
| Database | Supabase Cloud (PostgreSQL + pgvector) |
| Realtime | SSE streamed from `run_events` by `seq` (replaces Redis pub/sub; in-process wake-ups, Postgres LISTEN/NOTIFY in M8) |
| Background jobs | In-process asyncio tasks (`app/workers/job_runner.py`; no Redis/ARQ) — run the API as one worker |
| Search | DuckDuckGo via `ddgs` (successor of `duckduckgo-search`, which stopped returning results), behind the `SearchProvider` interface |
| Page extraction | httpx + BeautifulSoup |
| Testing | pytest, pytest-asyncio, httpx |
| Quality | ruff, mypy |

## Repository Layout

```
researchmind/
├── AGENTS.md
├── .env.example
├── README.md
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── core/          # config, logging, security, deps
│   │   ├── api/v1/        # routers
│   │   ├── models/        # SQLAlchemy models
│   │   ├── schemas/       # Pydantic request/response
│   │   ├── services/      # business logic
│   │   ├── workers/       # background task runners
│   │   └── db/            # session, migrations (Alembic)
│   └── tests/
├── agents/
│   ├── pyproject.toml
│   ├── researchmind_agents/
│   │   ├── config.py      # AgentConfig — built from backend Settings
│   │   ├── state.py       # shared LangGraph state
│   │   ├── graph.py       # graph assembly
│   │   ├── nodes/         # agent nodes
│   │   ├── tools/         # search, page_extract, vector_search
│   │   ├── prompts/       # one file per agent
│   │   ├── llm/           # provider abstraction
│   │   └── evals/         # evaluation harness
│   └── tests/
└── frontend/              # placeholder, built later
```

## Package boundary

The agents package must not import `app.*`, SQLAlchemy, or FastAPI. The two
sides meet in exactly two places:

- **Config** — `backend/app/services/agent_runtime.py` turns `Settings` into an
  `AgentConfig` (a plain frozen dataclass) via `AgentConfig.from_mapping`.
- **Storage** — the backend injects three things, all built in
  `agent_runtime.py` and bundled into an `AgentRuntime`
  (`researchmind_agents/runtime.py`):
  - a `UsageSink` that writes `llm_usage` rows (the runtime's `BudgetTracker`
    wraps it, so every call also counts against the run budget);
  - a `VectorSearchFn` adapted from `PassageRepository`;
  - a `ResearchStore` (`researchmind_agents/storage.py`) —
    `SqlResearchStore` in `backend/app/services/research_store.py` — through
    which nodes save the plan, sources, passages, events, and the report.
    `InMemoryResearchStore` implements the same protocol for tests and DB-less
    runs.

All are keyword-only contracts. When adding a setting, add it to `Settings`,
`.env.example`, and the matching `AgentConfig` sub-config together. Graph state
holds only JSON-compatible values (dumped Pydantic models), so any LangGraph
checkpointer can serialise it.

Test doubles for the graph (scripted LLM, fake web, deterministic embeddings)
live in `researchmind_agents/testing.py` so both test suites can use them.
