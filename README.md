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
   # backend/.env is read wherever you run from (a repo-root .env also works;
   # backend/.env wins). Only GEMINI_API_KEY is required; see .env.example
   # for every other setting.
   cp .env.example backend/.env
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

### Run a research question (M3)

From the repository root, with `GEMINI_API_KEY` set in `backend/.env`:

```bash
# One question: prints the cited markdown report, usage, and a grounding check
python backend/scripts/run_mvp.py "Is RAG better than fine-tuning for enterprise QA?"

# Seed eval: runs agents/researchmind_agents/evals/seed.jsonl and writes
# results to agents/researchmind_agents/evals/results/
python backend/scripts/run_mvp.py --seed
```

On Gemini's free tier a run takes about 1.5–2 minutes: embeddings are capped at
30k tokens per minute, so the run waits out the rate limit (up to
`LLM_MAX_RETRY_DELAY`) instead of dropping passages. When Gemini answers
503 "high demand", calls retry after `LLM_OVERLOAD_RETRY_DELAY` (10 s, doubling).
If the model still cannot write the report, the run stores an extractive report
that quotes passages; the seed eval counts that as a failure, since it does not
test the synthesizer.

`--store memory` runs with no database. `--store db` persists the run, sources,
passages, events, usage, and report to `DATABASE_URL`; apply migrations first
with `cd backend && alembic upgrade head`. The default, `auto`, uses the
database when `DATABASE_URL` is set.

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
OpenAI-compatible LLM can be paired with Gemini embeddings. The default model is
`gemini-embedding-001`, asked for 768-dimensional vectors to match the
`vector(768)` column (`text-embedding-004` was shut down on 2026-01-14).

## Research graph (quick mode)

```
START → planner ─(one branch per sub-question, in parallel)→ web_researcher → synthesizer → END
              └────────────(budget spent)────────────────────────────────────↗
```

- **Planner** — splits the question into 2–5 sub-questions with search queries.
- **Web researcher** — searches, lets the model pick which results to read,
  fetches pages, stores each as a source (de-duplicated across branches), then
  chunks and embeds its text as passages.
- **Synthesizer** — retrieves the passages most similar to each sub-question
  and writes the report. The model sees sources as `S1`, `S2`, … and must cite
  them on every statement; unknown or missing citations get one retry, then the
  statement is dropped. Markdown is rendered from the validated JSON, so every
  citation points to a stored source.
- **Budget** — `MAX_TOKENS_PER_RUN` / `MAX_COST_PER_RUN_USD` are enforced per
  run. Research stops early to leave `BUDGET_SYNTHESIS_RESERVE` for the report;
  if the cap is fully spent, the report quotes passages without a model call.
  A run always ends with a stored report.

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
