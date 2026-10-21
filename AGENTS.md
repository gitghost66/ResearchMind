# ResearchMind — Agent Working Rules

## Rules

1. Read the PRD (`prompt.txt`) before starting any task; work only on the current milestone.
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
| LLM | Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) via `google-genai` |
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
