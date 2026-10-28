"""
The runtime a research graph runs against.

Everything a node needs — config, LLM, embeddings, tools, store, budget — in
one object, built once per run. The backend builds it from ``Settings`` with a
SQL store (``app.services.agent_runtime``); tests, evals, and DB-less scripts
build it here with :func:`build_runtime`.

The budget tracker is wired in front of the usage sink, so it must be created
before the LLM provider. That is why providers are passed as a factory.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from researchmind_agents.budget import BudgetTracker
from researchmind_agents.config import AgentConfig, LLMConfig
from researchmind_agents.llm.base import BaseLLMProvider, UsageSink
from researchmind_agents.llm.embeddings import EmbeddingService
from researchmind_agents.llm.factory import build_llm_provider
from researchmind_agents.storage import InMemoryResearchStore, ResearchStore
from researchmind_agents.tools.page_extract import PageExtractTool
from researchmind_agents.tools.scholarly import ScholarlySearchTool
from researchmind_agents.tools.search import SearchTool
from researchmind_agents.tools.vector_search import VectorSearchTool

__all__ = ["AgentRuntime", "LLMFactory", "build_runtime", "utc_now"]

LLMFactory = Callable[[LLMConfig, UsageSink | None], BaseLLMProvider]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class AgentRuntime:
    """Everything the graph needs, already wired to config and storage."""

    config: AgentConfig
    llm: BaseLLMProvider
    embeddings: EmbeddingService
    search: SearchTool
    page_extract: PageExtractTool
    store: ResearchStore
    budget: BudgetTracker
    scholarly: ScholarlySearchTool
    vector_search: VectorSearchTool | None = None
    # Reference time for source recency (PRD s8); injectable for tests.
    now: Callable[[], datetime] = utc_now
    # Monotonic seconds, for the run's elapsed time (the loop deadline);
    # injectable for tests. The run starts when the runtime is built.
    clock: Callable[[], float] = time.monotonic
    started: float = field(default=-1.0)
    # Normalised URLs a research branch has claimed in this run. Branches skip
    # claimed URLs and read other candidates instead of fetching a page twice.
    # Per-run scratch only: not checkpointed, so a resumed run may refetch.
    claimed_urls: set[str] = field(default_factory=set)
    # Sources stored (or being stored) in this run, counted against
    # ``config.research.max_sources_per_run``. Per-run scratch, like the above.
    source_slots: int = 0

    def __post_init__(self) -> None:
        if self.started < 0:
            self.started = self.clock()

    def elapsed(self) -> float:
        """Seconds since the run started."""
        return max(0.0, self.clock() - self.started)

    async def aclose(self) -> None:
        """Release HTTP clients held by the tools."""
        await self.page_extract.aclose()
        await self.scholarly.aclose()
        closer = getattr(self.llm, "aclose", None)
        if closer is not None:
            await closer()


def build_runtime(
    config: AgentConfig,
    *,
    store: ResearchStore | None = None,
    usage_sink: UsageSink | None = None,
    vector_search: VectorSearchTool | None = None,
    llm_factory: LLMFactory | None = None,
    embeddings: EmbeddingService | None = None,
    search: SearchTool | None = None,
    page_extract: PageExtractTool | None = None,
    scholarly: ScholarlySearchTool | None = None,
    now: Callable[[], datetime] | None = None,
    clock: Callable[[], float] | None = None,
) -> AgentRuntime:
    """
    Assemble a runtime for one run.

    With no *store*, an :class:`InMemoryResearchStore` is used, and it also
    serves vector search unless *vector_search* is given. Any tool can be
    overridden, which is how tests inject fakes; *now* fixes the clock used
    for source recency, *clock* the one used for the run's elapsed time.
    """
    budget = BudgetTracker.from_config(config)
    resolved_store: ResearchStore = store if store is not None else InMemoryResearchStore()

    if vector_search is None and isinstance(resolved_store, InMemoryResearchStore):
        vector_search = VectorSearchTool(resolved_store.vector_search)

    factory = llm_factory or build_llm_provider
    return AgentRuntime(
        config=config,
        llm=factory(config.llm, budget.wrap(usage_sink)),
        embeddings=embeddings or EmbeddingService(config.embedding),
        search=search or SearchTool(config.search),
        page_extract=page_extract or PageExtractTool(config.page_extract),
        store=resolved_store,
        budget=budget,
        scholarly=scholarly or ScholarlySearchTool(config.academic),
        vector_search=vector_search,
        now=now or utc_now,
        clock=clock or time.monotonic,
    )
