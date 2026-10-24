"""
Agent runtime assembly.

This is the seam between the two packages. The agents package knows nothing
about SQLAlchemy or ``Settings``; this module builds its config from
``Settings``, constructs the provider and tools, and binds the usage sink to
``LLMUsageRepository`` so every LLM call lands in ``llm_usage``
(PRD s6.3, M2 acceptance).

Nothing here contains prompt text or agent logic (AGENTS.md rule 2).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from researchmind_agents.config import AgentConfig
from researchmind_agents.llm.base import BaseLLMProvider, LLMUsage, UsageSink
from researchmind_agents.llm.embeddings import EmbeddingService
from researchmind_agents.llm.factory import build_llm_provider
from researchmind_agents.tools.page_extract import PageExtractTool
from researchmind_agents.tools.search import SearchTool
from researchmind_agents.tools.vector_search import VectorSearchTool, make_vector_search_fn
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.repositories.evidence_repository import PassageRepository
from app.repositories.research_repository import LLMUsageRepository

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


@dataclass
class AgentRuntime:
    """Everything the graph needs, already wired to config and storage."""

    config: AgentConfig
    llm: BaseLLMProvider
    embeddings: EmbeddingService
    search: SearchTool
    page_extract: PageExtractTool
    vector_search: VectorSearchTool | None = None

    async def aclose(self) -> None:
        """Release HTTP clients held by the tools."""
        await self.page_extract.aclose()
        closer = getattr(self.llm, "aclose", None)
        if closer is not None:
            await closer()


def build_agent_runtime(
    *,
    settings: Settings | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    session: AsyncSession | None = None,
) -> AgentRuntime:
    """
    Assemble an :class:`AgentRuntime`.

    Pass *session_factory* to enable usage recording, and *session* to enable
    vector search. Both are optional so the runtime can also be built in tests
    and scripts with no database.
    """
    config = build_agent_config(settings)

    usage_sink = make_usage_sink(session_factory) if session_factory is not None else None
    if usage_sink is None:
        logger.warning(
            "Agent runtime built without a session factory; LLM usage will not be recorded"
        )

    vector_search = None
    if session is not None:
        vector_search = VectorSearchTool(make_vector_search_fn(PassageRepository(session)))

    return AgentRuntime(
        config=config,
        llm=build_llm_provider(config.llm, usage_sink),
        embeddings=EmbeddingService(config.embedding),
        search=SearchTool(config.search),
        page_extract=PageExtractTool(config.page_extract),
        vector_search=vector_search,
    )
