# ResearchMind — Master Implementation Plan

> **Status:** Active implementation
> **Done:** M0, M1, M2, M4, M5, M6 (offline acceptance; live checks pending: migration 0003 on
> Supabase, the M5 verification eval, and the M6 contradiction eval and latency targets)
> **Open:** M3 seed eval, M5 verification eval, M6 contradiction eval and per-mode latency
> (all need a live `GEMINI_API_KEY` run)
> **Next:** M7 — evaluation framework
> **Last updated:** 2026-10-28
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

### ✅ M5 — Specialised agents & verification

- Researchers: `TechnicalResearcher`, `MarketResearcher`, `AcademicResearcher` (arXiv, Semantic Scholar).
- **ClaimExtractor** — atomic claims linked to passage IDs.
- **FactChecker** — verdict (`supported` / `weak` / `unsupported` / `mixed`) per claim against its evidence.
- **Scoring (`reliability.py`)**
  - Source reliability (0–1): domain type, recency, authority. Domain-type classification from a versioned list in `agents/researchmind_agents/data/domains.json` plus TLD rules.
  - Claim confidence (0–1): source reliability × independent corroboration × verdict.

**Done when**
- [x] Scoring functions have pure unit tests (`test_reliability.py`).
- [x] Claims always link to passages (`test_graph.py::TestVerifiedModes`, `test_api_research.py::TestVerifiedRun`).
- [x] The report includes verdicts and confidence (key findings + `overall_confidence`, deep and academic).
- [x] 20 hand-labelled claims (`evals/claims_gold.jsonl`); with the labelled stances, scoring reproduces
  every verdict and every confidence within ±0.15 (MAE 0.022) — `test_check_verification.py`.
- [ ] Live: verdict agreement ≥ 80% and confidence within ±0.15 for ≥ 80% of the gold claims, with the
  real model — `backend/scripts/eval_verification.py`.

**As built (2026-10-27)**
- **Modes:** `deep` (web, technical, market researchers) and `academic` (adds the academic researcher,
  favoured) run the verified pipeline, single pass. `quick` is unchanged and unverified. `decision`,
  contradictions, and the loop stay in M6. Modes live in `modes.py`; the graph is compiled per mode
  (`build_research_graph(runtime, mode=...)`), and `pipeline_steps(mode)` drives API progress.
- **Routing:** the planner (prompt v2) tags each sub-question with a domain from those the mode allows;
  unknown or disallowed domains fall back to the mode's default, and a mode that favours a domain gets at
  least one sub-question in it. `PlannedSubQuestion.domain` is the one optional field in an LLM schema
  (defaults to `web`), so quick-mode plans and older outputs still validate.
- **Specialists:** web, technical, and market are one pipeline with a `ResearcherProfile`: agent name
  (events and usage), selection guidance, and source types read first when the model's picks run out.
  Search results show each site's type to the model.
- **Academic:** `tools/scholarly.py` queries arXiv (Atom XML) and Semantic Scholar (JSON) over httpx with
  per-provider request spacing, a byte cap, retries, a TTL cache, and merge/dedupe by arXiv ID / DOI /
  title. Papers are stored as sources whose text is title + abstract (no page fetch); typed
  `peer_reviewed` with a venue, else `preprint`. With no usable papers the branch falls back to web
  research that prefers journals, archives, and `.edu` sites.
- **Claim extraction:** passages are retrieved per sub-question with the synthesizer's retrieval (moved
  to `nodes/retrieval.py`), each read once for its best-matching sub-question; one call per
  sub-question. The model cites passage labels (`P1`…); claims without a valid label are dropped;
  identical wording is merged with all its passages.
- **Fact checking:** each claim is checked against its own passages plus similar passages from other
  sources (vector search, new sources first), as excerpts around the claim (`tools/excerpts.py`, 700
  chars) — a check reads a few sentences, not whole chunks. The model returns a verdict and a stance per
  excerpt; `reconcile_verdict` makes the verdict agree with the stances (no support → unsupported; any
  contradiction → mixed). Claims the model did not judge are stored `unverified`, confidence 0. Claims
  and `claim_evidence` rows are written in one transaction; `claim_verified` is streamed per claim.
- **Scoring:** weights are `ScoringConfig` fields set from `RELIABILITY_*` / `CONFIDENCE_*` settings
  (type and verdict tables as JSON overrides). Corroboration is a noisy-OR over independent sites with
  `source_weight` 0.8 — a first draft let one highly cited paper score 1.0 on its own; now a single source
  reaches at most 0.8 × its reliability. `domains.json` replaced the planned `domains.yaml` to avoid adding
  PyYAML. Sources are scored in every mode (the `/sources` endpoint shows them); `published_at` comes from
  page meta tags / JSON-LD or the paper index.
- **Synthesis:** verified modes write from usable claims only (`C1`…), and each statement's source labels
  are derived from its claims (`ReportStatement.claims`, hidden from the model's schema). The report adds
  `key_findings` (verdict, confidence, citations, evidence with stance and excerpt), `verification`
  counts, and `overall_confidence`. If no claim is usable the report falls back to passages with a
  limitation and no confidence; on model failure it lists the claims extractively.
- **Gold set:** the 20 claims use fictional products and studies on `.example` hosts, so the model cannot
  answer from memory and no text is attributed to a real organisation. Labels: 8 supported, 4 weak,
  4 mixed, 4 unsupported. They are hand-written — spot-check them before relying on the live numbers.
- **Not yet:** contradiction records and the penalty they feed (`unresolved_contradictions` is 0 until
  M6); per-mode planner bounds (PRD's 4–8 sub-questions for deep) and latency targets are M6.
  (All three landed in M6.)

---

### ✅ M6 — Contradictions, loop, evidence graph, modes

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

**Done when**
- [x] A seeded conflicting-source run produces a contradiction with its cause; the detector saw the
  dates that explain it (`test_graph.py::TestContradictions`).
- [x] The loop terminates at `max_iterations` with gaps in every round (`test_graph.py::TestLoop`),
  and five rounds fit the recursion limit.
- [x] The graph payload validates against its schema (`backend/tests/test_api_graph.py`).
- [x] Contradiction gold set: 15 cases, 14 seeded contradictions across all seven causes, 3 cases with
  none. With the labelled answers, the harness finds 14/14 with no false positives
  (`test_check_contradictions.py`).
- [ ] Live: contradiction detection finds ≥ 70% of the seeded contradictions —
  `backend/scripts/eval_contradictions.py`.
- [ ] Live: each mode meets its latency target at p50 on the seed eval.

**As built (2026-10-28)**
- **Graph:** verified modes are now `... -> fact_checker -> contradiction_detector -> sufficiency_checker`,
  which routes back to the planner or on to the synthesizer. Each pass is a research round
  (`iteration` 0, 1, ...). Only the round's new sub-questions fan out; a gap round that adds nothing
  goes straight to the synthesizer. `run_research` sets `max_iterations` from `MAX_ITERATIONS` for
  looping modes (quick runs one), and the run config's `recursion_limit` grows with it.
- **Rounds stay incremental:** sub-questions and claims record their round. The claim extractor reads
  only the round's sub-questions and passages no earlier round read (`extracted_passage_ids`; the
  retrieval asks for extra hits to make up for skipped ones) and drops claims worded like earlier
  ones. The fact checker checks and stores only the round's claims; earlier verdicts stand. The
  detector compares pairs with at least one new claim. Nothing is re-checked.
- **Gap-filling planner** (`planner_gaps/v1`, its own `GapPlanOutput` schema): sees the sub-questions
  with their coverage and the queries already tried, plus the gaps (fenced as untrusted data — model
  gaps derive from web text). It proposes at most `PLANNER_MAX_GAP_SUB_QUESTIONS`; repeats of
  sub-questions or tried queries are dropped. A sub-question filling a sub-question's gap gets
  `parent_id`, and coverage, report sections, and confidence count it toward that top-level
  sub-question — otherwise a filled gap would stay "uncovered" forever. Rows are stored with
  `save_plan(iteration=...)`, which also sets `research_runs.iteration`; the plan JSON keeps every
  sub-question plus a `rounds` log. Model failure, a spent budget, or a failed store add nothing.
- **Sufficiency** (`coverage.py`, `nodes/sufficiency.py`): rules first — each top-level sub-question
  needs `SUFFICIENCY_MIN_CLAIMS_PER_SUB_QUESTION` (2) usable claims from
  `SUFFICIENCY_MIN_SITES_PER_SUB_QUESTION` (2) independent sites. Only when the rules pass and
  another round could follow does the model judge (`sufficiency/v1`), naming up to
  `SUFFICIENCY_MAX_GAPS` missing aspects; unresolved contradictions are shown to it. A round may start
  only below `max_iterations`, with research budget left, and before the mode's
  `LOOP_DEADLINE_SECONDS` (deep 150 s, academic 180 s, decision 150 s: half the latency targets).
  Gaps left when the loop stops become report limitations.
- **Contradictions** (`contradiction_detector/v1`): candidates are deterministic — usable claim pairs
  with cosine ≥ `CONTRADICTION_MIN_SIMILARITY` (0.7) or linked by the fact checker (one claim's
  passage contradicts the other), skipping pairs from one passage; at most `CONTRADICTION_MAX_PAIRS`,
  linked first. Pairs are joined into topics of ≤ `CONTRADICTION_MAX_GROUP_SIZE` claims, several per
  call. Each claim is shown with its supporting excerpts and their sources' type and date. A record
  needs two or more claims of one topic. Causes: `conditions`, `date`, `version`, `dataset`,
  `methodology`, `definition`, `unexplained`; *resolved* is decided by code from
  `CONFIDENCE_RESOLVED_CAUSES` (default: the first four and `definition` — the claims differ in scope).
  Records go to `contradictions` / `contradiction_claims` (no migration needed: M1 created the
  tables; `likely_cause` holds the category).
- **Report:** `contradictory_evidence` (X1, ... with claim labels, citations, cause, resolved) and, in
  decision mode, `recommendation`. Claims in a contradiction are always among the labelled claims, so
  both sides can be shown. Overall confidence now applies the unresolved-contradiction penalty
  (`CONFIDENCE_CONTRADICTION_PENALTY` each, up to the cap). The verified prompt is
  `synthesizer_verified/v2` (contradictions, user context); decision mode uses
  `synthesizer_decision/v1`, whose recommendation is required: a missing one triggers the citation
  retry, then a limitation.
- **Modes:** `decision` offers every researcher with market favoured (PRD "All"; the academic
  researcher only gets sub-questions tagged `academic`). The first plan asks for 2–5 sub-questions in
  quick mode and `PLANNER_VERIFIED_MIN/MAX_SUB_QUESTIONS` (4–8) otherwise.
- **Evidence graph** (`app/schemas/graph.py`, `app/services/evidence_graph.py`): typed nodes
  (per-type `meta`, so claim nodes always carry verdict and confidence) and `{id, from, to, type,
  meta}` edges; the model rejects duplicate IDs and dangling edges. Includes passages linked as
  supporting or contradicting evidence plus the report's cited passages (all a quick run has), their
  sources, and every report-cited source. Built from stored rows, so it works mid-run.
- **Contradiction eval:** `evals/contradictions_gold.jsonl` (synthetic, `.example` hosts) and
  `evals/check_contradictions.py` report recall, cause agreement, precision, false positives, and
  candidate recall. Offline, the bag-of-words fake embeddings miss two paraphrased pairs at 0.7
  (12/14 candidates); the harness test uses 0.45 and finds 14/14. The live script reports the real
  candidate recall for tuning the threshold.
- **Not yet:** live runs of the contradiction eval and the per-mode latency check; LLM-call tracing
  spans (M4 note) and checkpoint resume (M8) are unchanged.

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
