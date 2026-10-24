# ResearchMind — Master Implementation Plan

> **Status:** Active implementation
> **Done:** M0, M1, M2 (188 tests collected, ruff + mypy clean)
> **Next:** M3 — MVP graph
> **Last updated:** 2026-10-22
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
| Web search | DuckDuckGo (`duckduckgo-search`, renamed upstream to `ddgs`) | Behind a `SearchProvider` interface from M3; Tavily/Brave/SearXNG as alternatives |
| Scraping | httpx + BeautifulSoup4 | SSRF-safe, robots-aware |
| Background jobs | In-process asyncio tasks | Single API worker assumed until M8 |
| Realtime | SSE | |
| Tracing | LangSmith or OpenTelemetry | From M4 |
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
- Unit tests for every node with mocked LLM/tools; one graph test end-to-end with fakes.
- `run_mvp.py` produces a cited report against live APIs.
- Seed eval: 100% of citations resolve; ≥ 80% of report sentences cited.
- A run stops at its budget cap and still returns a report.

---

### 📋 M4 — API, background jobs, durability

- Endpoints under `/api/v1/research`:
  - `POST /` → 202 + `run_id`
  - `GET /` (paginated), `GET /{id}`, `DELETE /{id}` (cascade)
  - `GET /{id}/events` (SSE), `/report`, `/sources`, `/usage`
  - `POST /{id}/cancel`
- **Job runner** — asyncio task registry with cancellation and a global concurrency cap.
- **Durability (moved forward from M8)**
  - LangGraph Postgres checkpointer.
  - On startup, runs left in `running` are marked `failed` (or resumed once resumption lands).
- **SSE** — events written to `run_events` and streamed from there, so reconnects replay with `Last-Event-ID`. Single worker for now; document the limit.
- **Security** — API-key header, per-key/IP rate limiting, input length limits.
- **Tracing** — one trace per run, one span per node and LLM call.

**Done when:** a run can be started, streamed, cancelled and fetched over HTTP; killing the server mid-run leaves no run stuck in `running`; API tests cover every endpoint.

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
