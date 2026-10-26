"""
Agent runtime assembly.

This is the seam between the two packages. The agents package knows nothing
about SQLAlchemy or ``Settings``; this module builds its config from
``Settings``, binds the usage sink to ``LLMUsageRepository`` so every LLM call
lands in ``llm_usage`` (PRD s6.3, M2 acceptance), and hands the graph a SQL
``ResearchStore`` plus a vector search function. Assembly itself is
``researchmind_agents.runtime.build_runtime``, so the budget tracker is wired
in front of the usage sink the same way in every caller.

Nothing here contains prompt text or agent logic (AGENTS.md rule 2).
"""

from __future__ import annotations

import logging
import uuid

from researchmind_agents.config import AgentConfig
from researchmind_agents.llm.base import LLMUsage, UsageSink
from researchmind_agents.llm.embeddings import EmbeddingService
from researchmind_agents.runtime import AgentRuntime, LLMFactory, build_runtime
from researchmind_agents.storage import ResearchStore
from researchmind_agents.tools.page_extract import PageExtractTool
from researchmind_agents.tools.search import SearchTool
from researchmind_agents.tools.vector_search import VectorSearchTool, make_vector_search_fn
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.repositories.evidence_repository import PassageRepository
from app.repositories.research_repository import LLMUsageRepository
from app.services.research_store import SqlResearchStore, make_session_vector_search_fn

logger = logging.getLogger(__name__)

__all__ = [
    "AgentRuntime",
    "build_agent_config",
    "build_agent_runtime",
    "make_usage_sink",
    "record_usage",
]


def build_agent_config(settings: Settings | None = None) -> AgentConfig:
    """Translate ``Settings`` into the agents package's own config object."""
    resolved = settings or get_settings()
    return AgentConfig.from_mapping(resolved.agent_config_mapping())


async def record_usage(session: AsyncSession, usage: LLMUsage) -> None:
    """
    Persist one :class:`LLMUsage` as an ``llm_usage`` row.

    Usage without a ``run_id`` cannot be attributed (the column is NOT NULL and
    foreign-keyed), so it is logged and dropped rather than failing the call.
    """
    if not usage.run_id:
        logger.debug(
            "Skipping usage row without run_id (agent=%s model=%s)",
            usage.agent,
            usage.model,
        )
        return

    repo = LLMUsageRepository(session)
    row = usage.as_usage_row()
    await repo.record_usage(
        run_id=uuid.UUID(str(row["run_id"])),
        agent=row["agent"],
        model=row["model"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        cost_usd=row["cost_usd"],
        latency_ms=row["latency_ms"],
    )


def make_usage_sink(session_factory: async_sessionmaker[AsyncSession]) -> UsageSink:
    """
    Build a usage sink that writes each call to its own short-lived session.

    A dedicated session (rather than the caller's) means accounting commits
    even if the surrounding research transaction later rolls back — the tokens
    were spent either way.
    """

    async def sink(usage: LLMUsage) -> None:
        async with session_factory() as session:
            await record_usage(session, usage)
            await session.commit()

    return sink


def build_agent_runtime(
    *,
    settings: Settings | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    session: AsyncSession | None = None,
    store: ResearchStore | None = None,
    llm_factory: LLMFactory | None = None,
    embeddings: EmbeddingService | None = None,
    search: SearchTool | None = None,
    page_extract: PageExtractTool | None = None,
) -> AgentRuntime:
    """
    Assemble an :class:`AgentRuntime` for one run.

    * *session_factory* enables usage recording, a :class:`SqlResearchStore`,
      and vector search with a fresh session per call.
    * *session* wires vector search to that one session instead (kept for
      callers that already hold one).
    * Without either, the runtime uses the agents package's in-memory store,
      which also serves vector search — enough for tests and DB-less scripts.

    *store*, *llm_factory*, and the tool arguments override the defaults; tests
    use them to inject fakes.
    """
    config = build_agent_config(settings)

    usage_sink = make_usage_sink(session_factory) if session_factory is not None else None
    if usage_sink is None:
        logger.warning(
            "Agent runtime built without a session factory; LLM usage will not be recorded"
        )

    if store is None and session_factory is not None:
        store = SqlResearchStore(session_factory)

    vector_search: VectorSearchTool | None = None
    if session is not None:
        vector_search = VectorSearchTool(make_vector_search_fn(PassageRepository(session)))
    elif session_factory is not None:
        vector_search = VectorSearchTool(make_session_vector_search_fn(session_factory))

    return build_runtime(
        config,
        store=store,
        usage_sink=usage_sink,
        vector_search=vector_search,
        llm_factory=llm_factory,
        embeddings=embeddings,
        search=search,
        page_extract=page_extract,
    )
