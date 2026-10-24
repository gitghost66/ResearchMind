# ResearchMind

A multi-agent AI research system that decomposes complex questions, researches them in parallel with specialized agents, extracts and verifies claims, detects contradictions, and produces citation-backed reports with confidence scores.

## Quick Start

### Prerequisites
- Python 3.11+
- A [Supabase](https://supabase.com) project (free tier works)
- A [Google AI Studio](https://aistudio.google.com) API key

### Setup

1. **Clone and install dependencies:**
   ```bash
   # Install backend
   cd backend
   pip install -e ".[dev]"

   # Install agents package
   cd ../agents
   pip install -e ".[dev]"
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Supabase and Gemini API credentials
   ```

3. **Run the API:**
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```

4. **Verify:**
   ```bash
   curl http://localhost:8000/health
   # => {"status": "healthy"}
   ```

### Run Tests
```bash
# Both packages from the repo root
pytest

# Or per package
cd backend && pytest
cd ../agents && pytest
```

Tests mock every LLM, search, and HTTP call, so the suite runs with no API keys
and no database. Tests needing live services are marked `integration` and skip
unless their environment variable is set:

```bash
# Verifies the Alembic migration against a real PostgreSQL + pgvector
DATABASE_URL=postgresql+asyncpg://... pytest -m integration
```

### Run Quality Checks
```bash
ruff check .
mypy backend/app agents/researchmind_agents
```

## LLM providers

The LLM layer is provider-agnostic; the choice is config, not code.

| `LLM_PROVIDER` | Transport | Keys |
|---|---|---|
| `gemini` (default) | `google-genai` | `GEMINI_API_KEY` |
| `openai_compatible` / `openrouter` / `groq` | OpenAI chat-completions over httpx | `LLM_BASE_URL`, `LLM_API_KEY` |

Every call goes through the wrapper, which handles retries, Pydantic-validated
structured output (re-prompting with the validation error on failure), and
writes a row to `llm_usage` with tokens, latency, and USD cost. Cost is
computed from `LLM_PRICE_PER_1M_*` or a built-in table, and is `0.0` on free
tiers rather than estimated.

Embeddings always use `google-genai`, independently of `LLM_PROVIDER`, so an
OpenAI-compatible LLM can be paired with Gemini embeddings.

## Architecture

```
Client ──REST/SSE──► FastAPI ──async task──► LangGraph Workflow
                        │                        │
                        ▼                        ▼
                   Supabase (PostgreSQL + pgvector)
                                                 │
                                    ┌─────────────┼──────────────┐
                                    ▼             ▼              ▼
                              Gemini API    DuckDuckGo    Supabase Realtime
```

## Project Structure

- `backend/` — FastAPI REST API, database models, services
- `agents/` — LangGraph agent workflow, LLM wrapper, tools, prompts
- `frontend/` — Placeholder (built in a later phase)

## Milestones

- [x] **M0** — Foundation (scaffold, config, health endpoint)
- [x] **M1** — Data layer (SQLAlchemy models, Alembic migrations, pgvector)
- [x] **M2** — LLM and tool layer (provider-agnostic wrapper with usage/cost
      recording, embeddings, search with TTL cache, SSRF-guarded page extraction,
      vector search)
- [ ] **M3** — MVP graph (Planner → Web Research → Synthesizer)
- [ ] **M4** — API and background jobs (full REST API, SSE streaming)
- [ ] **M5** — Specialized agents and verification
- [ ] **M6** — Contradictions, loop, evidence graph
- [ ] **M7** — Evaluation framework
- [ ] **M8** — Hardening and deployment

## License

MIT
