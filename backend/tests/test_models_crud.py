"""
CRUD and cascade deletion tests for all 11 SQLAlchemy models and repositories.
"""


import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.claims_repository import (
    ClaimEvidenceRepository,
    ClaimRepository,
    ContradictionRepository,
)
from app.repositories.evidence_repository import (
    PassageRepository,
    SourceRepository,
)
from app.repositories.research_repository import (
    LLMUsageRepository,
    ReportRepository,
    ResearchRunRepository,
    RunEventRepository,
    SubQuestionRepository,
)


@pytest.mark.asyncio
async def test_research_run_crud(db_session: AsyncSession):
    """Test full CRUD lifecycle for research_runs."""
    repo = ResearchRunRepository(db_session)

    # 1. Create
    run = await repo.create(
        question="Is RAG better than fine-tuning for document QA?",
        mode="deep",
        user_context={"audience": "engineers"},
        plan={"steps": ["search", "synthesize"]},
    )
    assert run.id is not None
    assert run.question == "Is RAG better than fine-tuning for document QA?"
    assert run.mode == "deep"
    assert run.status == "queued"
    assert run.iteration == 0

    # 2. Read
    fetched = await repo.get_by_id(run.id)
    assert fetched is not None
    assert fetched.id == run.id
    assert fetched.user_context["audience"] == "engineers"

    # 3. Update status
    updated = await repo.update_status(run.id, status="running")
    assert updated is not None
    assert updated.status == "running"
    assert updated.started_at is not None

    # 4. Count
    total = await repo.count()
    assert total >= 1

    # 5. Delete
    deleted = await repo.delete(run.id)
    assert deleted is True
    assert await repo.get_by_id(run.id) is None


@pytest.mark.asyncio
async def test_sub_questions_repository(db_session: AsyncSession):
    """Test batch sub-question creation and retrieval."""
    run_repo = ResearchRunRepository(db_session)
    sq_repo = SubQuestionRepository(db_session)

    run = await run_repo.create(question="Database comparison")

    items = [
        {"text": "What is Postgres performance?", "domain": "technical"},
        {"text": "What is MongoDB pricing?", "domain": "market"},
    ]
    sub_questions = await sq_repo.create_batch(run.id, items)
    assert len(sub_questions) == 2
    assert sub_questions[0].domain == "technical"
    assert sub_questions[1].domain == "market"

    # Fetch by run
    listed = await sq_repo.get_by_run(run.id)
    assert len(listed) == 2
    assert listed[0].text == "What is Postgres performance?"


@pytest.mark.asyncio
async def test_sources_and_passages(db_session: AsyncSession):
    """Test source creation, deduplication, and passage chunk storage."""
    run_repo = ResearchRunRepository(db_session)
    src_repo = SourceRepository(db_session)
    pass_repo = PassageRepository(db_session)

    run = await run_repo.create(question="Cloud architecture")

    # Create source
    source = await src_repo.create(
        run_id=run.id,
        url="https://aws.amazon.com/whitepapers",
        title="AWS Architecture",
        domain_type="official_docs",
        reliability_score=0.95,
        content_hash="abc123hash",
    )
    assert source.id is not None
    assert source.reliability_score == 0.95

    # Check deduplication helper
    existing = await src_repo.get_by_url_or_hash(
        run.id,
        url="https://aws.amazon.com/whitepapers",
        content_hash="abc123hash",
    )
    assert existing is not None
    assert existing.id == source.id

    # Create passages
    passages = await pass_repo.create_batch([
        {
            "source_id": source.id,
            "text": "Serverless architectures reduce infrastructure operational overhead.",
            "embedding": [0.1] * 768,
            "chunk_index": 0,
        },
        {
            "source_id": source.id,
            "text": "Cold starts can impact latency for infrequently used serverless functions.",
            "embedding": [0.2] * 768,
            "chunk_index": 1,
        },
    ])
    assert len(passages) == 2
    assert passages[0].chunk_index == 0
    assert passages[1].chunk_index == 1


@pytest.mark.asyncio
async def test_claims_and_evidence_linking(db_session: AsyncSession):
    """Test claims creation, fact-checking verdicts, and claim_evidence stances."""
    run_repo = ResearchRunRepository(db_session)
    src_repo = SourceRepository(db_session)
    pass_repo = PassageRepository(db_session)
    claim_repo = ClaimRepository(db_session)
    evidence_repo = ClaimEvidenceRepository(db_session)

    run = await run_repo.create(question="Python performance")
    source = await src_repo.create(run_id=run.id, url="https://python.org")
    passage = await pass_repo.create(
        source_id=source.id,
        text="Python 3.13 introduces an experimental JIT compiler.",
        chunk_index=0,
    )

    # Create claim
    claim = await claim_repo.create(
        run_id=run.id,
        text="Python 3.13 has an experimental JIT compiler.",
        verdict="unverified",
        confidence=0.0,
    )

    # Link evidence
    link = await evidence_repo.link_evidence(
        claim_id=claim.id,
        passage_id=passage.id,
        stance="supports",
    )
    assert link.stance == "supports"

    # Verify verdict update
    updated_claim = await claim_repo.update_verdict(
        claim.id,
        verdict="supported",
        confidence=0.92,
    )
    assert updated_claim is not None
    assert updated_claim.verdict == "supported"
    assert updated_claim.confidence == 0.92

    # Fetch claims for run with evidence
    claims = await claim_repo.list_by_run(run.id, include_evidence=True)
    assert len(claims) == 1
    assert len(claims[0].evidence_links) == 1
    assert claims[0].evidence_links[0].passage.text == passage.text


@pytest.mark.asyncio
async def test_contradiction_detection_models(db_session: AsyncSession):
    """Test contradiction recording and linking to conflicting claims."""
    run_repo = ResearchRunRepository(db_session)
    claim_repo = ClaimRepository(db_session)
    contra_repo = ContradictionRepository(db_session)

    run = await run_repo.create(question="Database benchmark")
    claim_a = await claim_repo.create(
        run_id=run.id,
        text="PostgreSQL 16 throughput reached 100k TPS on Benchmark X.",
        verdict="supported",
        confidence=0.85,
    )
    claim_b = await claim_repo.create(
        run_id=run.id,
        text="PostgreSQL 16 throughput capped at 45k TPS on Benchmark X.",
        verdict="supported",
        confidence=0.80,
    )

    contradiction = await contra_repo.create_with_claims(
        run_id=run.id,
        explanation="Conflicting throughput numbers reported for Benchmark X.",
        likely_cause="Different hardware specifications (NVMe SSD vs SATA)",
        claim_ids=[claim_a.id, claim_b.id],
    )
    assert contradiction.id is not None

    # Query contradictions for run with preloaded claim links
    contradictions = await contra_repo.list_by_run(run.id)
    assert len(contradictions) == 1
    assert len(contradictions[0].claim_links) == 2


@pytest.mark.asyncio
async def test_reports_and_events(db_session: AsyncSession):
    """Test report creation and versioning, and event logging."""
    run_repo = ResearchRunRepository(db_session)
    report_repo = ReportRepository(db_session)
    event_repo = RunEventRepository(db_session)

    run = await run_repo.create(question="AI Safety")

    # Log events
    ev1 = await event_repo.log_event(
        run_id=run.id,
        agent="planner",
        event_type="agent_started",
        payload={"sub_questions_count": 3},
    )
    assert ev1.agent == "planner"

    events = await event_repo.list_by_run(run.id)
    assert len(events) == 1
    assert events[0].type == "agent_started"

    # Save reports
    rep1 = await report_repo.create(
        run_id=run.id,
        content={"summary": "Initial draft"},
        markdown="# Draft Report",
        overall_confidence=0.75,
        version=1,
    )
    rep2 = await report_repo.create(
        run_id=run.id,
        content={"summary": "Final report"},
        markdown="# Final Report",
        overall_confidence=0.91,
        version=2,
    )
    assert rep1.version == 1
    assert rep2.version == 2

    latest = await report_repo.get_latest_by_run(run.id)
    assert latest is not None
    assert latest.version == 2
    assert latest.overall_confidence == 0.91


@pytest.mark.asyncio
async def test_llm_usage_summary(db_session: AsyncSession):
    """Test recording tokens and cost per agent with summary aggregation."""
    run_repo = ResearchRunRepository(db_session)
    usage_repo = LLMUsageRepository(db_session)

    run = await run_repo.create(question="Token tracking test")

    await usage_repo.record_usage(
        run_id=run.id,
        agent="planner",
        model="gemini-3.1-flash-lite",
        tokens_in=500,
        tokens_out=200,
        cost_usd=0.0003,
        latency_ms=800,
    )
    await usage_repo.record_usage(
        run_id=run.id,
        agent="web_researcher",
        model="gemini-3.1-flash-lite",
        tokens_in=1500,
        tokens_out=600,
        cost_usd=0.0009,
        latency_ms=1200,
    )
    await usage_repo.record_usage(
        run_id=run.id,
        agent="planner",
        model="gemini-3.1-flash-lite",
        tokens_in=400,
        tokens_out=150,
        cost_usd=0.0002,
        latency_ms=600,
    )

    summary = await usage_repo.get_summary_by_run(run.id)
    assert summary["total_tokens_in"] == 2400
    assert summary["total_tokens_out"] == 950
    assert round(summary["total_cost_usd"], 4) == 0.0014
    assert "planner" in summary["per_agent"]
    assert summary["per_agent"]["planner"]["tokens_in"] == 900
    assert summary["per_agent"]["web_researcher"]["tokens_in"] == 1500


@pytest.mark.asyncio
async def test_cascade_delete_on_run_deletion(db_session: AsyncSession):
    """Test that deleting a research_run cascades to all 10 child tables."""
    run_repo = ResearchRunRepository(db_session)
    sq_repo = SubQuestionRepository(db_session)
    src_repo = SourceRepository(db_session)
    pass_repo = PassageRepository(db_session)
    claim_repo = ClaimRepository(db_session)
    evidence_repo = ClaimEvidenceRepository(db_session)
    report_repo = ReportRepository(db_session)
    event_repo = RunEventRepository(db_session)
    usage_repo = LLMUsageRepository(db_session)

    # Create parent run
    run = await run_repo.create(question="Cascade deletion test")

    # Populate child tables
    sq = await sq_repo.create(run_id=run.id, text="Sub q")
    src = await src_repo.create(run_id=run.id, url="https://test.com")
    pas = await pass_repo.create(source_id=src.id, text="Passage text")
    cl = await claim_repo.create(run_id=run.id, text="Claim text")
    await evidence_repo.link_evidence(claim_id=cl.id, passage_id=pas.id)
    await report_repo.create(run_id=run.id, markdown="Report", overall_confidence=0.8)
    await event_repo.log_event(run_id=run.id, agent="test", event_type="start")
    await usage_repo.record_usage(
        run_id=run.id,
        agent="test",
        model="m",
        tokens_in=10,
        tokens_out=10,
        cost_usd=0.01,
        latency_ms=100,
    )

    # Store IDs
    sq_id = sq.id
    src_id = src.id
    pas_id = pas.id
    cl_id = cl.id

    # Delete run with cascade
    deleted = await run_repo.delete(run.id)
    assert deleted is True

    # Verify all children are deleted
    assert await sq_repo.get_by_id(sq_id) is None
    assert await src_repo.get_by_id(src_id) is None
    assert await pass_repo.get_by_id(pas_id) is None
    assert await claim_repo.get_by_id(cl_id) is None
    assert await report_repo.count(run_id=run.id) == 0
    assert await event_repo.count(run_id=run.id) == 0
    assert await usage_repo.count(run_id=run.id) == 0
