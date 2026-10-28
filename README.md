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

3. **Apply migrations** (once, and after pulling new ones):
   ```bash
   cd backend
   alembic upgrade head
   ```

4. **Run the API** — one worker: research runs execute inside the API process.
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000 --timeout-graceful-shutdown 5
   ```
   Keep `--timeout-graceful-shutdown`: an open event stream lasts until its run
   ends, and uvicorn waits for open connections before the app's shutdown runs,
   so without it stopping the server waits for every streamed run to finish.
   Without `DATABASE_URL` the API still starts, but the research endpoints
   answer 503.

5. **Verify:**
   ```bash
   curl http://localhost:8000/health
   # => {"status": "healthy"}
   curl http://localhost:8000/ready
   # => {"status": "ready", "checks": {"database": "ok", ...}, "runs": {...}}
   ```
   Interactive API docs are at http://localhost:8000/docs.

### Run a research question

From the repository root, with `GEMINI_API_KEY` set in `backend/.env`:

```bash
# One question: prints the cited markdown report, usage, and a grounding check
python backend/scripts/run_mvp.py "Is RAG better than fine-tuning for enterprise QA?"

# Deep, academic, or decision mode: specialist researchers, fact-checked claims,
# contradiction detection, gap-filling rounds, and a confidence score (more model
# calls, so slower on the free tier); decision mode ends in a recommendation
python backend/scripts/run_mvp.py --mode deep "PostgreSQL or MongoDB for an AI SaaS in 2026?"
python backend/scripts/run_mvp.py --mode decision "PostgreSQL or MongoDB for an AI SaaS in 2026?"

# Seed eval: runs agents/researchmind_agents/evals/seed.jsonl and writes
# results to agents/researchmind_agents/evals/results/
python backend/scripts/run_mvp.py --seed

# Verification eval (M5): the fact checker on 20 hand-labelled claims
# (one small model call each; no search, embeddings, or database)
python backend/scripts/eval_verification.py

# Contradiction eval (M6): the detector on 15 seeded cases (14 contradictions,
# 3 cases with none); one embedding call and a model call or two per case
python backend/scripts/eval_contradictions.py
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

## Research graph

| Mode | Researchers | Verified, with contradictions | Rounds | Report |
|---|---|---|---|---|
| `quick` | web | no | 1 | cited passages |
| `deep` | web, technical, market | yes | up to `MAX_ITERATIONS` (3) | key findings, confidence |
| `academic` | web, technical, market, academic (favoured) | yes | up to 3 | key findings, confidence |
| `decision` | web, technical, market (favoured), academic | yes | up to 3 | plus a required recommendation |

```
quick:     START → planner ─(branch per sub-question)→ web_researcher → synthesizer → END

deep /     START → planner ─(branch per new sub-question, to its domain's researcher)→
academic /         web | technical | market | academic researcher
decision:          → claim_extractor → fact_checker → contradiction_detector
                   → sufficiency_checker ─(gaps, and another round allowed)→ planner
                                         └─(enough, or no round allowed)→ synthesizer → END
```

- **Planner** — splits the question into sub-questions with search queries
  (2–5 in `quick`, 4–8 in the other modes), and tags each with the researcher
  that should handle it (`web`, `technical`, `market`, `academic`, as the mode
  allows). In a gap-filling round it gets the coverage so far and the gaps, and
  proposes at most `PLANNER_MAX_GAP_SUB_QUESTIONS` *new* sub-questions; one that
  fills a sub-question's gap counts toward that sub-question.
- **Web, technical, and market researchers** — one pipeline with different
  guidance: search, let the model pick which results to read (technical prefers
  documentation and benchmarks; market prefers pricing pages and industry
  reports), fetch pages, store each as a source (de-duplicated across branches),
  then chunk and embed its text as passages.
- **Academic researcher** — searches arXiv and Semantic Scholar (over httpx; no
  extra dependency) and stores each chosen paper's title and abstract as a
  source. If the indexes return nothing (Semantic Scholar's keyless pool is
  often rate-limited), it falls back to web search that prefers journals,
  archives, and university sites.
- **Claim extractor** — reads the most relevant passages per sub-question and
  lists atomic claims, each tied to the passages it came from. A claim without
  a valid passage is dropped, so every claim links to stored evidence.
- **Fact checker** — checks each claim against its own passages and the most
  similar passages from *other* sources. The model gives a verdict
  (`supported`, `weak`, `mixed`, `unsupported`) and a stance per passage; code
  makes the verdict agree with the stances and computes the confidence.
- **Contradiction detector** — pairs usable claims that are similar
  (`CONTRADICTION_MIN_SIMILARITY`) or that the fact checker linked (one claim's
  passage contradicts the other), groups them by topic, and asks the model which
  claims cannot both be true and why: `conditions`, `date`, `version`,
  `dataset`, `methodology`, `definition`, or `unexplained`. Code decides from the
  cause whether the contradiction is *resolved* (a difference in scope, so both
  claims can hold — `CONFIDENCE_RESOLVED_CAUSES`); unresolved ones lower the
  report's confidence.
- **Sufficiency checker** — rules first: every sub-question needs
  `SUFFICIENCY_MIN_CLAIMS_PER_SUB_QUESTION` usable claims from
  `SUFFICIENCY_MIN_SITES_PER_SUB_QUESTION` independent sites. When the rules
  pass, the model judges whether the claims answer the question and may name
  missing aspects. Gaps send the run back to the planner, unless the run is at
  `MAX_ITERATIONS`, out of research budget, or past its mode's
  `LOOP_DEADLINE_SECONDS`. Each later round researches, extracts, checks, and
  compares only its new material.
- **Synthesizer** — in `quick` mode, writes from retrieved passages, citing
  sources `S1`, `S2`, …. In the verified modes, writes only from usable verified
  claims, citing claims `C1`, `C2`, …; each statement's sources are those of its
  claims, and the report lists the key findings with verdicts and confidence,
  and the contradictions between them (`X1`, …) with their likely causes. In
  `decision` mode it must also write a recommendation; if none can be grounded,
  the report says so. Unknown or missing citations get one retry, then the
  statement is dropped. Markdown is rendered from the validated JSON, so every
  citation points to a stored source.
- **Budget** — `MAX_TOKENS_PER_RUN` / `MAX_COST_PER_RUN_USD` are enforced per
  run. Research stops early to leave `BUDGET_SYNTHESIS_RESERVE` for the report;
  if the cap is fully spent, the report quotes passages without a model call.
  A run always ends with a stored report.

### Scoring (no number comes from a model)

Formulas and weights are documented in
`agents/researchmind_agents/reliability.py`; every weight is a setting
(`RELIABILITY_*`, `CONFIDENCE_*`).

- **Source reliability (0–1)** — base score by source type (peer-reviewed,
  standards bodies, official docs, government > academic, preprints, news >
  reference, code repositories, company blogs > personal blogs, unknown >
  forums), times a recency factor, plus citation and methodology bonuses for
  papers. Types come from the versioned list in
  `agents/researchmind_agents/data/domains.json` plus TLD, host, and path rules.
- **Claim confidence (0–1)** — `verdict weight × corroboration × agreement`.
  Each site counts once; independent supporting sites combine by noisy-OR (one
  source alone reaches at most 0.8 × its reliability), and contradicting
  sources lower the agreement ratio.
- **Report confidence** — the cited claims' confidence, weighted by how many
  statements cite each, reduced for sub-questions without a verified claim and
  by `CONFIDENCE_CONTRADICTION_PENALTY` per unresolved contradiction (up to the
  cap).

## Research API (M4, M6)

Every `/api/v1/research` route needs the `X-API-Key` header (when `API_KEY` is
set) and counts against a per-key rate limit. Errors always look like
`{"error": {"code": "...", "message": "...", "details": ...}}`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/research` | Start a run: `{question, mode?, user_context?, max_sources?}` → 202 `{run_id, status, links}` |
| GET | `/api/v1/research` | List runs, newest first: `?limit=&offset=&status=&mode=` |
| GET | `/api/v1/research/{id}` | Status, plan, sub-questions, and progress per step |
| GET | `/api/v1/research/{id}/events` | Server-Sent Events: replay, then live, until the run ends |
| GET | `/api/v1/research/{id}/report` | Report JSON + markdown; in the verified modes, key findings with verdicts, contradictory evidence, and an overall confidence, plus a recommendation in `decision` (409 while running, 404 if none) |
| GET | `/api/v1/research/{id}/graph` | Evidence graph for a React Flow view: `{nodes, edges}` (see below); available while the run is in progress |
| GET | `/api/v1/research/{id}/sources` | Stored sources with type, reliability score, publication date, passage counts, and whether the report cites them |
| GET | `/api/v1/research/{id}/usage` | Calls, tokens, cost, and latency — total, per agent, per model |
| POST | `/api/v1/research/{id}/cancel` | Cancel a queued or running run (409 if it already ended) |
| DELETE | `/api/v1/research/{id}` | Cancel if active, then delete the run and all its data (204) |

```bash
KEY="X-API-Key: your-api-key-for-clients"
RUN=$(curl -s -H "$KEY" -H "Content-Type: application/json"   -d '{"question": "Is RAG better than fine-tuning for enterprise QA?"}'   http://localhost:8000/api/v1/research | python -c "import sys, json; print(json.load(sys.stdin)['run_id'])")

curl -N -H "$KEY" http://localhost:8000/api/v1/research/$RUN/events   # live progress
curl -s -H "$KEY" http://localhost:8000/api/v1/research/$RUN/report   # once it completes
curl -s -X POST -H "$KEY" http://localhost:8000/api/v1/research/$RUN/cancel
```

**Evidence graph.** Nodes are `{id, type, label, meta}` with `type` one of
`question`, `sub_question`, `claim`, `passage`, `source` (IDs are
`<type>:<record id>`); claim nodes carry `verdict` and `confidence`. Edges are
`{id, from, to, type, meta}`: `decomposes` (question → sub-question,
sub-question → gap-filling follow-up, sub-question → claim), `supports` /
`contradicts` (passage → claim, from the fact checker), `contradicts` (claim →
claim, a detected contradiction, with `likely_cause` and `resolved` in `meta`),
and `cites` (passage → its source). The schema is `EvidenceGraph` in
`backend/app/schemas/graph.py` and in the OpenAPI docs.

**Events.** Each SSE event's `event:` is its type and its `id:` is the run's
event sequence number (`seq`, 1, 2, 3, ...). Types include `run_status`
(queued → running → completed / failed / cancelled), `agent_started`,
`agent_finished`, `search_completed`, `source_found` (with the source's type and
reliability), `claims_extracted`, `claim_verified` (verdict and confidence),
`contradiction_found` (claims, likely cause, resolved), `sufficiency_checked`
(the round's gaps and whether another round runs), `report_ready`, and `error`
(an agent hit a problem; the run continues). The stream closes after
the terminal `run_status`. To resume after a dropped connection, send
`Last-Event-ID: <seq>` (browsers' `EventSource` does this itself) or
`?after=<seq>`. `EventSource` cannot send `X-API-Key`, so a web client should
use a fetch-based SSE client.

**Limits.** `MAX_QUESTION_CHARS`, `MAX_USER_CONTEXT_BYTES`,
`MAX_REQUEST_BODY_BYTES` (413), `MAX_SOURCES_LIMIT`; `RATE_LIMIT_REQUESTS` and
`RATE_LIMIT_RUNS` per `RATE_LIMIT_WINDOW_SECONDS` (429 with `Retry-After`);
`MAX_CONCURRENT_RUNS` run at once and up to `MAX_PENDING_RUNS` wait (503 beyond).

**Durability.** Runs are asyncio tasks in the API process. On shutdown they
are recorded as failed ("interrupted") — serve with
`--timeout-graceful-shutdown` so open event streams do not hold shutdown until
their runs finish. On startup, any run a crashed process left queued or running
is marked failed the same way, so no run stays stuck. Resuming interrupted runs
from a checkpoint is M8. Until then, run a single API worker: the job registry,
rate limits, and live event wake-ups are per process. For the same reason, do
not start the API while `run_mvp.py` is running against the same database: the
startup sweep would mark the script's run failed.

**Tracing.** Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to send each
run to LangSmith as one trace (named `research:<mode>`, tagged with its
`run_id`) with a span per graph node.

## Architecture

```
Client ──REST──► FastAPI ──asyncio task (JobRunner)──► LangGraph workflow
   ▲               │  ▲                                   │      │
   │ SSE           │  └──── wake-up on each new event ────┤      ├──► Gemini API
   └─ run_events ──┘                                      │      ├──► web search + pages
                                                          │      └──► arXiv + Semantic Scholar
      (seq order)  ▼                                      ▼
            Supabase PostgreSQL + pgvector  ◄── runs, sources, passages, claims, contradictions,
                                                events, usage, reports
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
- [x] **M4** — API and background jobs (REST API, SSE with resume, cancel,
      in-process job runner, API-key auth, rate limits, startup recovery)
- [x] **M5** — Specialized agents and verification (technical, market, and
      academic researchers; claim extraction; fact checking; deterministic
      source reliability and claim/report confidence; `deep` and `academic`
      modes). Live verification eval pending.
- [x] **M6** — Contradictions, loop, evidence graph (contradiction detector with
      likely causes; sufficiency check and gap-filling loop; `decision` mode;
      per-mode planner bounds and loop deadlines; `/graph` endpoint). Live
      contradiction eval and latency checks pending.
- [ ] **M7** — Evaluation framework
- [ ] **M8** — Hardening and deployment

## License

MIT
