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
cd backend && pytest
cd ../agents && pytest
```

### Run Quality Checks
```bash
ruff check .
mypy backend/app agents/researchmind_agents
```

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
- [ ] **M1** — Data layer (SQLAlchemy models, Alembic migrations, pgvector)
- [ ] **M2** — LLM and tool layer (Gemini wrapper, search, page extraction)
- [ ] **M3** — MVP graph (Planner → Web Research → Synthesizer)
- [ ] **M4** — API and background jobs (full REST API, SSE streaming)
- [ ] **M5** — Specialized agents and verification
- [ ] **M6** — Contradictions, loop, evidence graph
- [ ] **M7** — Evaluation framework
- [ ] **M8** — Hardening and deployment

## License

MIT
