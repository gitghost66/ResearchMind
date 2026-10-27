# ResearchMind — Master Implementation Plan

> **Status:** Active implementation
> **Done:** M0, M1, M2, M4 (offline acceptance; live check pending migration 0003 on Supabase)
> **Open:** M3 seed eval (needs a live `GEMINI_API_KEY` run)
> **Next:** M5 — specialised agents and verification
> **Last updated:** 2026-10-27
>
> This is the single source of truth for the roadmap. `ResearchMind_PRD.md` holds
> product requirements; `AGENTS.md` holds the rules every contributor/agent follows.

---

## 1. Summary

ResearchMind is an autonomous multi-agent research backend. It takes a question,
splits it into sub-questions, searches the web, extracts and embeds passages,
mines claims, detects contradictions, scores source reliability, and writes a
report in which every citation points to a stored source.

### Architecture rules

1. **Separation of concerns**
   - `backend/` — DB sessions, migrations, repositories, API routers, job runner. No prompts or LLM logic.
   - `agents/` — LangGraph graph, nodes, prompts, tools, evals. No HTTP routers, no SQLAlchemy, no `Settings` import. Persistence is reached only through injected callables; config arrives as `AgentConfig` dataclasses built by `backend/app/services/agent_runtime.py`.
2. **Structured LLM contracts** — every LLM call returns a validated Pydantic object. Only final report markdown is free text.
3. **No hard-coded config** — models, keys, timeouts, limits and thresholds come from `Settings` → `AgentConfig`.
4. **Grounded citations** — report citations must resolve to stored source IDs; orphan or invented citations are rejected.
5. **Deterministic scoring** — reliability and confidence are Python formulas, never LLM-generated numbers.
6. **Retrieved content is data** — scraped text is never treated as instructions (prompt-injection defence in every prompt that includes it).
7. **Swappable providers** — LLM, embeddings and search sit behind interfaces so a vendor change is a config change.

---

## 2. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Runtime | Python 3.11+ (dev: 3.13) | |
| API | FastAPI + Uvicorn | REST + SSE |
| Schemas / settings | Pydantic v2, pydantic-settings | |
| Orchestration | LangGraph 1.x + langchain-core | Checkpointing via Postgres saver from M4 |
| LLM | Provider-agnostic: `gemini` (default `gemini-3.1-flash-lite`) or `openai_compatible` (OpenRouter, Groq, vLLM, …) | Selected by `LLM_PROVIDER` |
| Embeddings | Google embeddings, 768-dim | Default `text-embedding-004`; **verify availability**, migrate to `gemini-embedding-001` @ 768 dims if retired |
| DB / vectors | PostgreSQL 16 + pgvector (HNSW, cosine) | Supabase or local Docker |
| ORM / migrations | SQLAlchemy 2.0 async (asyncpg), Alembic | |
| Web search | DuckDuckGo via `ddgs` (replaced `duckduckgo-search` in M3 — it silently returned no results) | Behind the `SearchProvider` interface (M3); Tavily/Brave/SearXNG as alternatives |
| Scraping | httpx + BeautifulSoup4 | SSRF-safe, robots-aware |
| Background jobs | In-process asyncio tasks (`JobRunner`) | Single API worker assumed until M8 |
| Realtime | SSE from `run_events` (ordered by `seq`) | In-process wake-ups; Postgres LISTEN/NOTIFY in M8 |
| Tracing | LangSmith via env (opt-in) | Run = trace, node = span (M4); LLM-call spans deferred |
| Quality | pytest, pytest-asyncio, ruff, mypy | |

---

## 3. Data model (`backend/app/models/`)

```
research_runs (id, question, mode, status, plan, iteration, error, ...)
 ├── sub_questions (id, run_id, text, domain, status)
 ├── sources (id, run_id, url, title, domain_type, reliability_score, content_hash)
 │    └── passages (id, source_id, sub_question_id, text, embedding vector(768),
 │                  chunk_index, embedding_model*)
 ├── claims (id, run_id, sub_question_id, text, verdict, confidence)
 │    ├── claim_evidence (claim_id, passage_id, stance)
 │    └── contradiction_claims (contradiction_id, claim_id)
 ├── contradictions (id, run_id, explanation, likely_cause)
 ├── reports (id, run_id, content jsonb, markdown, overall_confidence, version)
 ├── run_events (id, run_id, ts, agent, type, payload jsonb)
 └── llm_usage (id, run_id, agent, model, tokens_in, tokens_out, cost_usd, latency_ms)

* added in M3 (migration 0002)
```

---

## 4. Completed (M0 – M2)

### ✅ M0 — Foundation
- Monorepo: `backend/`, `agents/`, `frontend/` with root and per-package `pyproject.toml`.
- `backend/app/core/config.py` — validated `Settings`.
- Logging, bearer-token dependency, `/health` and `/ready`.
- `docker-compose.yml` with Postgres 16 + pgvector.

### ✅ M1 — Data layer
- 11 SQLAlchemy models across `models/research.py`, `models/evidence.py`, `models/claims.py`.
- Alembic `0001_initial_schema.py` with HNSW index on `passages.embedding`; shared DDL in `app/db/ddl.py`.
- Async repositories with dialect-aware vector search (pgvector `<=>` / SQLite fallback).

### ✅ M2 — LLM & tool layer
- **Provider abstraction** (`agents/researchmind_agents/llm/`):
  - `base.py` — `BaseLLMProvider`, `LLMResponse`, `LLMUsage`, `UsageSink`, error types.
  - `client.py` / `google_client.py` — `GeminiProvider`; `openai_compatible.py` — `OpenAICompatibleProvider`.
  - `factory.py` — `build_llm_provider(config)`; unknown provider fails at startup.
  - `pricing.py` — per-model pricing table, config override wins.
  - Transport retries with backoff, plus schema-validation retries that feed the error back to the model.
- **Agent config** — `agents/researchmind_agents/config.py`: frozen `AgentConfig` / `LLMConfig` / `EmbeddingConfig` / `SearchConfig` / `PageExtractConfig`, built with `AgentConfig.from_mapping(settings.model_dump())`.
- **EmbeddingService** — batched, concurrency-limited, document/query task types, dimension check.
- **SearchTool** — DuckDuckGo, multi-query with concurrency cap, in-process TTL cache, empty list on failure.
- **PageExtractTool** — scheme allowlist, DNS-resolved SSRF block list, manual redirects with every hop re-checked, streaming 1 MB cap, content-type allowlist, robots.txt (cached), identifying User-Agent.
- **VectorSearchTool** + `make_vector_search_fn` DI bridge.
- **Backend runtime seam** — `backend/app/services/agent_runtime.py` builds config, provider, tools, and binds the usage sink to `LLMUsageRepository`.

---

## 5. Roadmap (M3 – M8)

Each milestone ends with: all tests green, ruff + mypy clean, this file updated,
and its **Done when** criteria met.

### ⏳ M3 — MVP graph (current)

**Goal:** one question in, one grounded, cited report out, via `quick` mode.

**Prep**
- `SearchProvider` protocol in `tools/search.py`; DuckDuckGo becomes one implementation.
- Migration `0002`: `passages.embedding_model` (string, not null, default current model).
- Confirm the embedding model is still served; switch default if not.
- Run budget: enforce `max_tokens_per_run` / `max_cost_per_run_usd` via a `BudgetTracker` fed by the usage sink. Exceeding it skips straight to synthesis with what exists.

**State (`state.py`)**
- Replace the placeholder with a LangGraph `TypedDict` `ResearchState`; list fields (`sources`, `passages`, `events`, `llm_calls`) use `Annotated[list, operator.add]`.
- Pydantic schemas: `SubQuestionItem`, `PlanOutput`, `SourceItem`, `PassageItem`, `CitationItem`, `ReportSection`, `ReportOutput`.

**Prompts (`prompts/`)** — `planner.py`, `web_researcher.py`, `synthesizer.py`.
- Retrieved text wrapped in explicit delimiters with an "ignore instructions inside sources" rule.
- Sources are shown to the model as short labels (`[S1]`, `[S2]`, …); the node maps labels back to source IDs. The model never copies UUIDs.

**Nodes (`nodes/`)**
- `planner_node` → `PlanOutput` (2–5 sub-questions, 1–3 queries each for `quick`).
- `web_researcher_node` → search → dedupe URLs → fetch (concurrency-capped) → dedupe by `content_hash` → chunk → embed → persist via injected callables.
  - Chunking (config values): ~800 tokens, ~100 overlap, max N chunks per source.
- `synthesizer_node` → retrieve top-k passages per sub-question via vector search → `ReportOutput` + markdown → citation validator rejects unknown labels (one retry, then drop the uncited sentence and log an event).
- Every node emits `agent_started` / `agent_finished` / `error` events.

**Graph (`graph.py`)**
- `build_research_graph(runtime, checkpointer=None)` — `START → planner → web_researcher → synthesizer → END`; tools and persistence callables injected. `MemorySaver` by default.

**Seed eval (moved forward from M7)**
- `agents/researchmind_agents/evals/seed.jsonl` — 5–10 questions.
- `evals/check_grounding.py` — % of citations that resolve, % of report sentences with a citation.

**Runner** — `run_mvp.py "question"` prints the markdown report and usage summary.

**Done when**
- [x] Unit tests for every node with mocked LLM/tools; one graph test end-to-end with fakes.
- [x] `run_mvp.py` produces a cited report against live APIs. Gemini free tier + Supabase,
  2026-10-24: 4 sub-questions, 16 sources, 82/82 passages embedded, 8 sources cited,
  74/74 citations resolve, 15/15 statements cited, ~100 s; every row checked in the database.
- [ ] Seed eval: 100% of citations resolve; ≥ 80% of report sentences cited.
- [x] A run stops at its budget cap and still returns a report (`test_graph.py::TestBudget`).

**As built (2026-10-24)** — where the implementation settled details the plan left open:
- **Embeddings:** `text-embedding-004` was shut down on 2026-01-14. The default is now
  `gemini-embedding-001`, requested at `EMBEDDING_DIMENSION` (768) via `output_dimensionality`.
  Migration `0002` adds `passages.embedding_model`, backfilling existing rows with `text-embedding-004`.
- **Storage seam:** nodes persist through a `ResearchStore` protocol (`agents/.../storage.py`):
  `SqlResearchStore` in the backend, `InMemoryResearchStore` for tests, evals, and `--store memory`.
  `AgentRuntime` moved into the agents package (`runtime.py`); the backend re-exports it.
- **Fan-out:** `Send` gives one web-researcher branch per sub-question. Branches claim URLs per run,
  so a page is fetched once and other branches read different candidates. The store de-duplicates
  by URL or content hash.
- **Web researcher prompt:** picks which search results to read (labels only); search order is the fallback.
- **Synthesizer:** retrieval is scoped to the run, not the sub-question, so a page found by one
  branch serves every sub-question. Statement-level citations, one retry, then uncited statements
  are dropped. If the budget is spent or the model fails, an extractive report quotes passages
  instead, so a run always stores a report.
- **Budget:** `BudgetTracker` wraps the usage sink. Research stops at (1 − `BUDGET_SYNTHESIS_RESERVE`)
  of either cap; the synthesizer makes no model call once the hard cap is spent.
- **Not yet:** `overall_confidence` stays `null` until M5's claim scoring (the DB column stores 0.0).
  `SYNTH_MIN_SIMILARITY` defaults to 0; tune it with the seed eval.
- **Runner:** `backend/scripts/run_mvp.py` calls `app/services/research_runner.execute_research`,
  which M4's job runner will reuse.
- **Search:** `duckduckgo-search` silently returned zero results, so it was replaced by `ddgs`
  (same API). DuckDuckGo alone also returned nothing from the dev machine, so the default is
  `SEARCH_BACKEND=auto` (several engines, merged), `SEARCH_REGION=us-en` (`wt-wt` breaks the other
  engines), and `SEARCH_MAX_ATTEMPTS=2`. Live check with real search and fetching (LLM and
  embeddings faked): 2 sub-questions → 8 sources, 40 passages in ~10 s, with 403/429 pages
  replaced by backups.
- **Free-tier rate limits:** Gemini's free tier caps embeddings at 30k tokens/min; one quick run
  embeds ~60k. Retries now honour the server's "retry in Ns" (capped by `LLM_MAX_RETRY_DELAY`,
  60 s), and `AGENT_TIMEOUT` defaults to 240 s so a branch outlives the wait. Before this fix,
  1 s/2 s retries silently lost ~40% of embeddings in the first live run.
- **Supabase connection:** `DATABASE_URL` values in Supabase's plain `postgresql://` form are switched
  to `postgresql+asyncpg://` automatically. Every engine comes from `app.db.session.build_engine`,
  which disables asyncpg prepared-statement caching on the transaction pooler (port 6543).
  Alembic escapes `%` in percent-encoded passwords. Settings read `backend/.env` (or the root
  `.env`) by absolute path, and the test suite never reads either.

---

### ✅ M4 — API, background jobs, durability

- Endpoints under `/api/v1/research`:
  - `POST /` → 202 + `run_id`
  - `GET /` (paginated), `GET /{id}`, `DELETE /{id}` (cascade)
  - `GET /{id}/events` (SSE), `/report`, `/sources`, `/usage`
  - `POST /{id}/cancel`
- **Job runner** — asyncio task registry with cancellation and a global concurrency cap.
- **Durability** — on startup, runs left in `queued`/`running` are marked `failed`.
  (The LangGraph Postgres checkpointer moved back to M8, with resume: it adds two
  dependencies and does nothing until a run can resume.)
- **SSE** — events written to `run_events` and streamed from there, so reconnects replay with `Last-Event-ID`. Single worker for now; document the limit.
- **Security** — API-key header, per-key/IP rate limiting, input length limits.
- **Tracing** — one trace per run, one span per node. (LLM-call spans need
  `langsmith` as a declared dependency; deferred.)

**Done when**
- [x] A run can be started, streamed, cancelled and fetched over HTTP
  (`test_api_research.py`; `test_api_server.py` shows events arriving live over a real socket).
- [x] Killing the server mid-run leaves no run stuck in `running`: graceful shutdown records
  "interrupted"; the startup sweep catches a hard kill (`TestLifecycle`).
- [x] API tests cover every endpoint, with the mocked LLM.
- [ ] Live check against Supabase after `alembic upgrade head` (migration 0003).

**As built (2026-10-27)**
- **Event order:** migration `0003` adds `run_events.seq` (1, 2, 3, ... per run; unique with
  `run_id`; existing rows numbered by `(ts, id)`). Timestamps could not order a stream: concurrent
  branches commit out of timestamp order, so a cursor on `ts` skips late commits.
  `RunEventRecorder` (`services/run_events.py`) is the only writer: it assigns `seq` and commits
  under a per-run lock, so commit order equals `seq` order, then wakes listeners in the process.
  The SSE id is `seq`; `Last-Event-ID` (or `?after=`) resumes exactly after it.
- **Lifecycle as events:** each status change is a `run_status` event committed in the same
  transaction as the update. A terminal status is final (`UPDATE … WHERE status IN (…)`), so a
  late "completed" cannot overwrite "cancelled", and a cancelled queued run never starts.
  Streams end on the terminal `run_status`; runs from before M4 end on their row's status.
- **Stream:** reads always go to the database by `seq` — replay, live delivery, and reconnects
  are one code path. The in-process notifier only saves polling; `SSE_POLL_INTERVAL_SECONDS`
  catches events written by another process. Each read runs as a shielded task, so a client
  hanging up mid-read does not tear down a pooled connection. FastAPI's built-in SSE
  (`fastapi.sse`) sends keep-alive pings; no SSE dependency was added.
- **Job runner** (`workers/job_runner.py`): `MAX_CONCURRENT_RUNS` execute, up to
  `MAX_PENDING_RUNS` wait (503 `queue_full` beyond). Interrupts carry a reason: a client cancel
  records `cancelled`, shutdown records `failed` with "Interrupted: …". The job itself
  (`ResearchJobs.run`) moves queued → running → completed/failed; `execute_run` never touches status.
- **Per-run options:** `research_runs.options` (migration `0003`) stores request options apart from
  `user_context`. `max_sources` becomes `ResearchConfig.max_sources_per_run`, a run-wide cap the
  web researcher enforces with slots reserved before each store (released on duplicates and
  failures); a branch that finds the cap reached is `skipped`. Default `MAX_SOURCES_PER_RUN=0` (none).
- **Progress per step:** derived from events, in the order `researchmind_agents.graph.pipeline_steps`
  gives: `pending` / `running` / `completed` / `incomplete` / `skipped`. A fan-out step stays
  `running` until the next step starts, so a branch finishing early does not mark it done.
- **Shutdown:** uvicorn waits for open connections before running the app's shutdown, and an
  SSE stream stays open until its run ends — so the Dockerfile and compose file pass
  `--timeout-graceful-shutdown 5`, after which uvicorn cancels the streams and the lifespan
  interrupts runs. `stop_grace_period` (30 s) covers both waits.
- **Failure paths:** any error in the job — including reading the run or marking it running — ends
  the run `failed`; recording the outcome is retried with backoff. A run interrupted before its
  task's first step is still recorded (an unstarted coroutine never runs its `except`).
- **Errors and limits:** one envelope for `ApiError`, FastAPI's HTTP and validation errors, and
  unhandled exceptions. Settings-dependent limits (question length, context size, `max_sources`)
  are checked in `ResearchService.validate` and reported in the same `details` shape as Pydantic's.
  The body limit checks `Content-Length` and counts chunked bodies. The API key is checked
  first (401/403 before any 503), in constant time; failed attempts count against the client IP.
  Rate-limit keys are a SHA-256 fingerprint of the key, else the client IP. Cursors and offsets
  are capped at the INTEGER range. Cancel answers 409 if the run finished before it took effect.
- **App factory:** `create_app(settings=, services=)`; settings resolve at startup, not import.
  Without `DATABASE_URL` the app starts and research routes answer 503. `/ready` pings the DB.
- **Tests:** a file-backed SQLite harness with the agents' fakes (`conftest.py::make_api`). The
  gate holds runs at the planner's model call: cancelling a run while a branch is mid-query can
  leave aiosqlite holding a write lock until GC (asyncpg closes the socket, so Postgres is fine).
- **Known limits:** `max_sources` is an upper bound, not a target. A slot is reserved before the
  store de-duplicates, so in a narrow race a run can stop one or two sources short of the cap
  (never over it); fixing that means moving the cap check into `save_source`. Runs started by
  `run_mvp.py` look orphaned to an API process starting on the same database.
- **Independent review (2026-10-27):** a separate review pass found nine issues; eight were fixed
  with tests (shutdown with open streams, failure paths, unstarted-task cancel, cursor bounds,
  slot release on timeout, cancel-after-completion, auth ordering and throttling, FastAPI floor);
  the `max_sources` race above is documented instead.
- **Not yet:** `/graph` (M6); resume from checkpoint and multi-worker fan-out (M8); CORS still
  permissive (M8).

---

### 📋 M5 — Specialised agents & verification

- Researchers: `TechnicalResearcher`, `MarketResearcher`, `AcademicResearcher` (arXiv, Semantic Scholar).
- **ClaimExtractor** — atomic claims linked to passage IDs.
- **FactChecker** — verdict (`supported` / `weak` / `unsupported` / `mixed`) per claim against its evidence.
- **Scoring (`reliability.py`)**
  - Source reliability (0–1): domain type, recency, authority. Domain-type classification from a versioned list in `agents/researchmind_agents/data/domains.yaml` plus TLD rules.
  - Claim confidence (0–1): source reliability × independent corroboration × verdict.

**Done when:** on 20 hand-labelled claims, verdict agreement ≥ 80% and confidence within ±0.15 of the label; scoring functions have pure unit tests.

---

### 📋 M6 — Contradictions, loop, evidence graph, modes

- **ContradictionDetector** — opposing claims on the same topic plus a likely cause (conditions, dates, versions, datasets).
- **SufficiencyChecker** — if gaps remain and `iteration < max_iterations` and budget allows, loop to the planner with gap-filling sub-questions.
- **`GET /{id}/graph`** — React Flow–compatible `{nodes, edges}` (`decomposes`, `supports`, `contradicts`, `cites`).
- **Modes**

  | Mode | Agents | Iterations | Target |
  |---|---|---|---|
  | `quick` | planner, web, synthesizer | 1 | ~45 s |
  | `deep` | all except academic | ≤ 3 | ~5 min |
  | `academic` | all, academic first | ≤ 3 | ~6 min |
  | `decision` | all, market first, mandatory recommendation | ≤ 3 | ~5 min |

**Done when:** each mode meets its latency target on the seed eval at p50; contradiction detection finds ≥ 70% of seeded contradictions.

---

### 📋 M7 — Evaluation framework

- Grow the seed set to 30–50 questions across all modes, with gold sources and expected sub-topics.
- Compare three systems: baseline LLM (no retrieval), single-agent RAG, and ResearchMind.
- Metrics:
  - Retrieval: P@K, R@K, MRR.
  - Answers: faithfulness, citation precision, completeness.
  - Operations: hallucination rate, latency, tokens, cost.
- CLI: `python -m researchmind_agents.evals run --suite default` → JSON + markdown comparison table.

**Done when:** ResearchMind beats both baselines on faithfulness and citation precision; results committed under `evals/results/`.

---

### 📋 M8 — Hardening & deployment

- Resume failed or aborted runs from the Postgres checkpointer.
- Multi-worker support: SSE fan-out via Postgres `LISTEN/NOTIFY`.
- SSRF: pin the connection to the validated IP (closes the DNS-rebinding gap between check and connect).
- Multi-stage production `Dockerfile`; Nginx config; AWS EC2/ECS guide with managed Postgres.
- README: architecture diagram, curl examples, local setup.

**Done when:** the deployed instance passes a smoke test of the full API; a run survives a worker restart.

---

## 6. Directory layout

```
C:\ResearchMind\
├── AGENTS.md                 # Operating rules
├── IMPLEMENTATION_PLAN.md    # This file
├── ResearchMind_PRD.md       # Product requirements
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── alembic/versions/     # 0001_initial_schema.py (+ 0002 in M3)
│   ├── app/
│   │   ├── main.py
│   │   ├── core/             # Settings, logging, security
│   │   ├── api/v1/           # health (+ research in M4)
│   │   ├── db/               # session, shared DDL
│   │   ├── models/           # research, evidence, claims
│   │   ├── repositories/     # async CRUD + vector search
│   │   └── services/         # agent_runtime (+ job runner, scoring glue)
│   └── tests/
├── agents/
│   ├── researchmind_agents/
│   │   ├── config.py         # AgentConfig dataclasses
│   │   ├── state.py          # ResearchState + schemas (M3)
│   │   ├── graph.py          # build_research_graph (M3)
│   │   ├── llm/              # base, providers, factory, pricing, embeddings
│   │   ├── tools/            # search, page_extract, vector_search
│   │   ├── prompts/          # (M3)
│   │   ├── nodes/            # (M3)
│   │   └── evals/            # seed set (M3), full harness (M7)
│   └── tests/
└── frontend/                 # Future React Flow UI
```

---

## 7. Verification

```bash
.venv\Scripts\python -m pytest agents/tests/ backend/tests/ -q
.venv\Scripts\python -m ruff check agents/ backend/
.venv\Scripts\python -m mypy agents/researchmind_agents/ backend/app/ --ignore-missing-imports
```

All three must pass before a milestone is marked done. Tests use mocks and need no API keys.
