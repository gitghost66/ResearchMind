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
| Embeddings | Gemini `text-embedding-004` (768d) via `google-genai` |
| Database | Supabase Cloud (PostgreSQL + pgvector) |
| Realtime | Supabase Realtime (replaces Redis pub/sub) |
| Background jobs | In-process asyncio tasks (no Redis/ARQ) |
| Search | DuckDuckGo (`duckduckgo-search`) |
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
- **Storage** — the backend injects callables: a `UsageSink` that writes
  `llm_usage` rows, and a `VectorSearchFn` adapted from `PassageRepository`.

Both are keyword-only contracts. When adding a setting, add it to `Settings`,
`.env.example`, and the matching `AgentConfig` sub-config together.
