"""
DuckDuckGo web search tool.

Includes an in-process TTL cache: the PRD asks for cached search results with a
configurable TTL (s10), and Redis was dropped from the stack in favour of
in-process state, so the cache lives here. It also keeps the free DuckDuckGo
endpoint from being hit repeatedly with the same query during a gap-filling
loop.

No LLM calls, no DB access.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from duckduckgo_search import DDGS

from researchmind_agents.config import SearchConfig

logger = logging.getLogger(__name__)

__all__ = ["SearchResult", "SearchTool"]


@dataclass
class SearchResult:
    """A single web search result."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = "duckduckgo"
    rank: int = 0
    extra: dict = field(default_factory=dict)


class _TTLCache:
    """Small insertion-ordered TTL cache; evicts oldest entries when full."""

    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max(1, max_entries)
        self._entries: OrderedDict[tuple, tuple[float, list[SearchResult]]] = OrderedDict()

    def get(self, key: tuple) -> list[SearchResult] | None:
        if self._ttl <= 0:
            return None
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.monotonic() - stored_at > self._ttl:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return list(value)

    def set(self, key: tuple, value: list[SearchResult]) -> None:
        if self._ttl <= 0:
            return
        self._entries[key] = (time.monotonic(), list(value))
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class SearchTool:
    """
    Async DuckDuckGo search wrapper.

    Usage::

        tool = SearchTool(SearchConfig(max_results=10, timeout=30))
        results = await tool.search("LangGraph tutorial")

    Returns an empty list on any error rather than raising, so a failed search
    degrades its sub-question instead of killing the run (PRD s6.3).
    """

    def __init__(self, config: SearchConfig | None = None) -> None:
        self._config = config or SearchConfig()
        self._cache = _TTLCache(
            self._config.cache_ttl_seconds,
            self._config.cache_max_entries,
        )

    @property
    def config(self) -> SearchConfig:
        return self._config

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        use_cache: bool = True,
    ) -> list[SearchResult]:
        """Run a text search and return ranked results, served from cache when fresh."""
        limit = max_results or self._config.max_results
        key = (
            query.strip().lower(),
            limit,
            self._config.region,
            self._config.safe_search,
        )

        if use_cache:
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("Search cache hit query=%r (%d results)", query, len(cached))
                return cached

        def call() -> list[dict]:
            with DDGS(timeout=self._config.timeout) as ddgs:
                return list(
                    ddgs.text(
                        query,
                        region=self._config.region,
                        safesearch=self._config.safe_search,
                        max_results=limit,
                    )
                )

        try:
            raw: list[dict] = await asyncio.to_thread(call)
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash the run
            logger.warning("Search failed for query=%r: %s", query, exc)
            return []

        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
                source="duckduckgo",
                rank=rank,
                extra={"raw": item},
            )
            for rank, item in enumerate(raw)
        ]

        if use_cache and results:
            self._cache.set(key, results)

        logger.debug("Search query=%r returned %d results", query, len(results))
        return results

    async def search_multi(
        self,
        queries: list[str],
        *,
        max_results_each: int | None = None,
        concurrency: int | None = None,
    ) -> dict[str, list[SearchResult]]:
        """
        Run several queries with bounded concurrency.

        Returns a mapping of query -> results. The concurrency bound matters on
        the free endpoint, which rate-limits aggressively.
        """
        if not queries:
            return {}

        semaphore = asyncio.Semaphore(max(1, concurrency or self._config.max_concurrency))

        async def run(query: str) -> list[SearchResult]:
            async with semaphore:
                return await self.search(query, max_results=max_results_each)

        all_results = await asyncio.gather(*(run(q) for q in queries))
        return dict(zip(queries, all_results, strict=True))
