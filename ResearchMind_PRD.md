# ResearchMind — Backend & Agents PRD

**Version:** 1.0 | **Scope:** `backend/` and `agents/` only (frontend is a later phase) | **Build tool:** Antigravity

---

## 1. Product summary

ResearchMind is a multi-agent research system. Given a complex question, it decomposes the question into sub-questions, researches them in parallel with specialized agents, extracts and verifies claims, detects contradictions between sources, and produces a citation-backed report with confidence scores and an evidence graph.

**One-liner:** a mini AI research team that shows its evidence instead of just answering.

### Goals
1. A working, end-to-end research pipeline exposed through a clean REST + streaming API.
2. A real LangGraph workflow (branching, parallel fan-out, loops), not sequential LLM calls.
3. Every claim in a report traceable to a source passage.
4. A built-in evaluation harness so the project has measurable results for the final-year report.
5. An API contract stable enough that the frontend can be built later without backend changes.

### Non-goals (for this PRD)
- Any frontend code (only the API contract it will consume).
- Multi-tenant billing, payments, or enterprise SSO.
- Training or fine-tuning models.
- Scraping sites that disallow it; use search APIs and controlled page extraction.

---

## 2. Users and key scenarios

| User | Scenario |
|---|---|
| Developer / student | "Is RAG better than fine-tuning for enterprise document QA?" → gets a cited report |
| Tech decision-maker | "PostgreSQL or MongoDB for an AI SaaS in 2026?" → gets a recommendation with trade-offs and limitations |
| Researcher | Academic mode: papers, methodology, datasets prioritized |

---

## 3. Repository layout

```text
researchmind/
├── AGENTS.md                # Antigravity rules (see section 13)
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/            # config, logging, security, deps
│   │   ├── api/v1/          # routers: research, sessions, graph, health
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic request/response models
│   │   ├── services/        # research_service, retrieval_service, cost_tracker
│   │   ├── workers/         # background job runners
│   │   └── db/              # session, migrations (Alembic)
│   ├── tests/
│   └── pyproject.toml
├── agents/
│   ├── researchmind_agents/
│   │   ├── state.py         # shared LangGraph state (Pydantic)
│   │   ├── graph.py         # graph assembly
│   │   ├── nodes/           # planner, web, technical, market, academic, fact_checker, contradiction, synthesizer
│   │   ├── tools/           # search, page_extract, vector_search
│   │   ├── prompts/         # one file per agent, versioned
│   │   ├── llm/             # provider abstraction
│   │   └── evals/           # evaluation harness
│   ├── tests/
│   └── pyproject.toml
└── frontend/                # empty placeholder, built later
```

`agents` is an installable Python package imported by `backend`. The backend never contains prompt text or agent logic.

---

## 4. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| API | FastAPI, Pydantic v2 | async throughout |
| Orchestration | LangGraph | checkpointing enabled |
| LLM access | Provider-agnostic wrapper | configurable via env; support at least two providers |
| Embeddings | Configurable (BGE local or API embeddings) | dimension set in config |
| Database | PostgreSQL 16 + pgvector | SQLAlchemy 2.0 async, Alembic migrations |
| Cache / queue | Redis | job queue, caching, run state, rate limiting |
| Background jobs | ARQ (or Celery if preferred) | research runs execute off the request path |
| Web research | Search API (e.g. Tavily or similar) + controlled page extraction | respect robots.txt, timeouts, size limits |
| Containers | Docker, docker-compose | api, worker, postgres, redis |
| Testing | pytest, pytest-asyncio, httpx | |
| Quality | ruff, mypy, pre-commit | |

---

## 5. System architecture

```text
Client ──REST/SSE──► FastAPI ──enqueue──► Redis queue ──► Worker
                        │                                   │
                        │                          LangGraph workflow
                        │                          (agents package)
                        ▼                                   │
                   PostgreSQL ◄─────────────────────────────┘
                (runs, sources, chunks+vectors,
                 claims, edges, reports, costs)
```

- The API creates a `research_run`, enqueues it, and returns immediately with a `run_id`.
- The worker executes the graph and writes progress events to Redis (pub/sub) and Postgres.
- The SSE endpoint streams those events to any client.

---

## 6. Agent workflow (LangGraph)

```text
START → Planner → [fan-out: Web | Technical | Market | Academic] → Evidence Store
      → Claim Extractor → Fact Checker → Contradiction Detector
      → Sufficiency Check ──(insufficient & iterations < max)──► Planner (gap-filling sub-questions)
                          └─(sufficient or cap reached)──► Synthesizer → END
```

### 6.1 Shared state (`state.py`)
```text
run_id, question, mode, user_context
plan: {objective, sub_questions[], domains[]}
sources[]: {id, url, title, published_at, domain_type, reliability_score}
passages[]: {id, source_id, text, embedding_ref, sub_question_id}
claims[]: {id, text, sub_question_id, evidence[], verdict, confidence}
contradictions[]: {claim_ids[], explanation, likely_cause}
iteration, max_iterations
report: {...}
usage: {tokens_in, tokens_out, cost_usd, per_agent{}}
events[]
```

### 6.2 Agents

| Agent | Input | Output | Key requirements |
|---|---|---|---|
| **Planner** | question, mode | plan with 4–8 sub-questions and domains | structured JSON output validated by Pydantic; on re-entry, receives gaps and proposes only new sub-questions |
| **Web Research** | sub-question | sources + passages | search, dedupe by URL/content hash, extract relevant passages, store metadata |
| **Technical** | sub-question | sources + passages | focus on docs, benchmarks, engineering sources |
| **Market / Cost** | sub-question | sources + passages | pricing, adoption, vendor lock-in, talent availability |
| **Academic** | sub-question | sources + passages | papers, methodology, datasets; only active in Academic mode or when planner requests |
| **Claim Extractor** | passages | atomic claims linked to passage IDs | one claim per statement; no claim without a supporting passage |
| **Fact Checker** | claims + evidence | verdict (`supported`, `weak`, `unsupported`, `mixed`) + confidence | checks what source actually says, conditions/context, recency, source reliability |
| **Contradiction Detector** | verified claims | contradiction records | groups claims on the same topic, explains *why* they disagree (different dataset, version, conditions, date) |
| **Sufficiency Check** | whole state | `sufficient` / `needs_more` + gap list | deterministic rules first (coverage per sub-question, min sources), LLM judgment second |
| **Synthesizer** | verified state | report | uses only verified claims; every statement carries citation IDs |

### 6.3 Behavioral rules (all agents)
- Structured outputs only (Pydantic schemas); retry with validation error feedback up to 2 times.
- No fabricated URLs: citations must come from the sources table.
- Per-agent timeouts, retries with backoff, and graceful degradation (a failed agent marks its sub-question as incomplete; the run continues).
- Parallel branches run concurrently with a configurable concurrency cap.
- All LLM calls go through the provider wrapper, which records tokens and cost.

### 6.4 Research modes

| Mode | Agents | Max iterations | Target |
|---|---|---|---|
| `quick` | Planner, Web, Synthesizer | 1 | under ~45 s |
| `deep` | All except Academic | 3 | under ~5 min |
| `academic` | All, Academic prioritized | 3 | under ~6 min |
| `decision` | All, Market/Cost prioritized, recommendation section required | 3 | under ~5 min |

---

## 7. Data model (PostgreSQL)

| Table | Key fields |
|---|---|
| `research_runs` | id, question, mode, status, user_context (jsonb), plan (jsonb), iteration, created_at, started_at, finished_at, error |
| `sub_questions` | id, run_id, text, domain, status |
| `sources` | id, run_id, url, title, published_at, domain_type, reliability_score, content_hash |
| `passages` | id, source_id, sub_question_id, text, embedding vector(N), chunk_index |
| `claims` | id, run_id, sub_question_id, text, verdict, confidence |
| `claim_evidence` | claim_id, passage_id, stance (`supports`/`contradicts`/`neutral`) |
| `contradictions` | id, run_id, explanation, likely_cause |
| `contradiction_claims` | contradiction_id, claim_id |
| `reports` | id, run_id, content (jsonb), markdown, overall_confidence, version |
| `run_events` | id, run_id, ts, agent, type, payload (jsonb) |
| `llm_usage` | id, run_id, agent, model, tokens_in, tokens_out, cost_usd, latency_ms |

Requirements: Alembic migrations, HNSW or IVFFlat index on `passages.embedding`, foreign keys with cascade on run deletion.

---

## 8. Source reliability and confidence scoring

**Reliability score (0–1)** per source, computed from:
- domain type (peer-reviewed/official docs/standards body > reputable publication > company blog > forum/unknown)
- publication recency relative to the question
- presence of citations/methodology (for papers)

Make the weighting configurable in one place (`reliability.py`) and unit-tested.

**Claim confidence (0–1):** function of number of independent sources, their reliability, agreement ratio, and fact-checker verdict.

**Overall report confidence:** weighted aggregate of claim confidences, penalized for unresolved contradictions and uncovered sub-questions. The formula must be documented and deterministic, not an LLM-invented number.

---

## 9. API specification (v1)

All responses JSON unless noted. Errors use a consistent envelope `{error: {code, message, details}}`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/research` | Create a run. Body: `{question, mode, user_context?, max_sources?}`. Returns `{run_id, status}` (202) |
| GET | `/api/v1/research/{run_id}` | Run status, plan, progress per step |
| GET | `/api/v1/research/{run_id}/events` | **SSE stream** of agent events (`agent_started`, `agent_finished`, `source_found`, `claim_verified`, `contradiction_found`, `report_ready`, `error`) |
| GET | `/api/v1/research/{run_id}/report` | Final report (JSON + markdown) |
| GET | `/api/v1/research/{run_id}/graph` | Evidence graph as `{nodes[], edges[]}` for the future React Flow UI |
| GET | `/api/v1/research/{run_id}/sources` | Sources with reliability scores |
| GET | `/api/v1/research/{run_id}/usage` | Token and cost breakdown per agent |
| POST | `/api/v1/research/{run_id}/cancel` | Cancel a running job |
| GET | `/api/v1/research` | List runs (paginated) |
| DELETE | `/api/v1/research/{run_id}` | Delete a run and its data |
| GET | `/health`, `/ready` | Liveness and readiness |

### Graph payload shape
```text
nodes: {id, type: question|sub_question|claim|source|passage, label, meta}
edges: {from, to, type: decomposes|supports|contradicts|cites}
```
Claim nodes carry `verdict` and `confidence`.

### Report JSON sections
`executive_summary, methodology, key_findings[], evidence[], contradictory_evidence[], analysis, recommendation (decision mode), limitations, sources[], overall_confidence`

---

## 10. Non-functional requirements

| Area | Requirement |
|---|---|
| Reliability | A single agent failure must not crash a run; runs are resumable from LangGraph checkpoints |
| Performance | Parallel fan-out; embeddings batched; search results cached in Redis (TTL configurable) |
| Cost control | Per-run token and USD budget cap; run stops gracefully and synthesizes with what exists when exceeded |
| Security | Secrets via env only; input length limits; rate limiting per IP/API key; SSRF protection on page extraction (block private IP ranges); prompt-injection hygiene (retrieved text treated as data, never as instructions) |
| Auth | API-key header for MVP; structured so JWT/user accounts can be added later |
| Observability | Structured JSON logs with `run_id`; per-agent latency and cost in `llm_usage`; optional LangSmith/OpenTelemetry hook |
| Config | Single `Settings` class (pydantic-settings); every model name, limit, and threshold configurable |
| Testing | Unit tests for scoring and parsing; agent tests with recorded/mocked LLM and search responses; API integration tests; CI-runnable without real API keys |

---

## 11. Evaluation framework (`agents/researchmind_agents/evals/`)

Purpose: produce the experimental results for the project report.

**Benchmark set:** 30–50 questions across the four modes, with hand-labeled expected sub-topics and a small set of gold sources.

**Systems compared**
1. Baseline LLM (single prompt, no retrieval)
2. Single-agent RAG
3. Full multi-agent system

**Metrics**

| Category | Metrics |
|---|---|
| Retrieval | Precision@K, Recall@K, MRR |
| Answer quality | Faithfulness (claims supported by cited passages), relevance, citation accuracy, completeness |
| Agent performance | Task completion rate, iterations per run, hallucination rate, latency, tokens and cost per run |

**Deliverables:** CLI (`python -m researchmind_agents.evals run --suite default`) that outputs a JSON results file and a markdown comparison table. LLM-as-judge prompts must be versioned and a sample manually spot-checked.

---

## 12. Milestones

Each milestone is a self-contained Antigravity task with acceptance criteria. Do them in order and verify before moving on.

### M0 — Foundation
- Monorepo scaffold, `pyproject.toml` for both packages, docker-compose (api, worker, postgres+pgvector, redis), `.env.example`, ruff/mypy/pytest setup, `AGENTS.md`.
- **Accept:** `docker compose up` starts all services; `/health` returns 200; `pytest` passes.

### M1 — Data layer
- SQLAlchemy models, Alembic migration for all tables in section 7, repository layer, pgvector index.
- **Accept:** migration applies cleanly on empty DB; CRUD tests pass; vector similarity query works in a test.

### M2 — LLM and tool layer
- Provider-agnostic LLM wrapper with structured output, retries, usage and cost recording. Search tool, page extractor (with SSRF guard), embedding service, vector search tool.
- **Accept:** all tools have mocked unit tests; wrapper records usage rows; extractor rejects private IPs.

### M3 — MVP graph (Phase 1)
- State, Planner, Web Research, RAG retrieval, Synthesizer, minimal graph; `quick` mode.
- **Accept:** a script runs a question end-to-end and returns a report with citations pointing only to stored sources.

### M4 — API and background jobs
- Endpoints from section 9 (research create/status/report/events SSE/cancel/list/delete), ARQ worker, Redis event pub/sub, error envelope, API-key auth, rate limiting.
- **Accept:** POST creates a run; SSE streams events live; report endpoint returns the result; cancel stops a run; integration tests pass with mocked LLM.

### M5 — Specialized agents and verification (Phase 2)
- Technical, Market/Cost, Academic agents; Claim Extractor; Fact Checker; confidence scoring and source reliability (section 8).
- **Accept:** claims always link to passages; scoring unit tests pass; report includes verdicts and confidence.

### M6 — Contradictions, loop, graph (Phase 3)
- Contradiction Detector with explanations, Sufficiency Check with gap-filling loop, `/graph` endpoint, all four modes, budget caps.
- **Accept:** seeded conflicting-source test produces a contradiction with a plausible cause; loop terminates at max iterations; graph payload validates against schema.

### M7 — Evaluation (Phase 4)
- Benchmark set, three-system comparison, metrics CLI, results markdown.
- **Accept:** one command produces the comparison table for the report.

### M8 — Hardening and deployment (Phase 5)
- Checkpoint resume, structured logging, production Dockerfiles, Nginx config, deployment notes for AWS (EC2/ECS) with managed Postgres, README with architecture diagram and API docs.
- **Accept:** fresh-clone setup works from README alone; a killed worker resumes a run from checkpoint.

---

## 13. Antigravity working rules (put in `AGENTS.md`)

1. Read this PRD before starting any task; work only on the current milestone.
2. Backend contains no prompt text or agent logic; agents package contains no HTTP or DB-session code (it talks to storage through interfaces passed in).
3. Every function that calls an LLM returns a validated Pydantic object.
4. Never hard-code model names, API keys, or thresholds; use `Settings`.
5. Write tests alongside code; a milestone is done only when its acceptance criteria pass.
6. Before finishing a task, run ruff, mypy, and pytest and report results.
7. Use mocks for LLM/search in tests; real API calls only in clearly marked optional integration tests.
8. Ask before adding a new dependency; prefer the stack in section 4.
9. Commit per milestone with a descriptive message.

### Suggested kickoff prompt for Antigravity
> Read `ResearchMind_PRD.md` and `AGENTS.md`. Implement **Milestone M0** only. Create the plan first, show me the file tree you intend to create, then implement, run the checks, and summarize what passed. Do not start M1.

(Repeat with the next milestone number each time.)

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| LLM hallucinated citations | Citations validated against stored source IDs; unsupported claims dropped from report |
| Runaway cost or loops | Budget cap, max iterations, per-agent timeouts |
| Search API quota/rate limits | Redis cache, backoff, concurrency cap |
| Prompt injection from web pages | Treat retrieved text as data; strip instructions; never let retrieved content change tool calls |
| Inconsistent structured output | Pydantic validation with retry-on-error |
| Scope creep | Milestone order is fixed; frontend and auth upgrades are out of scope until M8 is done |

---

## 15. Open decisions (choose before M0)

1. Primary LLM provider and model for agents vs. a cheaper model for extraction/judging.
2. Search API provider.
3. Embedding model and dimension (affects the `passages.embedding` column).
4. ARQ vs. Celery for background jobs.
5. Hosting target for the first deployment (EC2 vs. ECS).
