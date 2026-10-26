"""
Web search tool.

:class:`SearchTool` is what nodes call. It delegates the actual search to a
:class:`SearchProvider` — :class:`DDGSProvider` by default — so switching to
Tavily, Brave, or SearXNG means adding a provider, not touching nodes.

The tool adds what every provider needs:

* an in-process TTL cache — the PRD asks for cached search results with a
  configurable TTL (s10), and Redis was dropped from the stack in favour of
  in-process state. It also keeps a free endpoint from being hit repeatedly
  with the same query;
* a concurrency cap shared by every caller of the tool instance, so parallel
  graph branches cannot stampede a rate-limited endpoint;
* retries: a provider error is retried up to ``max_attempts`` times in total;
* degrade-don't-crash: errors that outlast the retries become an empty list.

No LLM calls, no DB access.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol

from ddgs import DDGS

from researchmind_agents.config import SearchConfig

logger = logging.getLogger(__name__)

__all__ = ["SearchResult", "SearchProvider", "DDGSProvider", "SearchTool"]


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


class SearchProvider(Protocol):
    """A search backend. Raise on failure; :class:`SearchTool` handles degradation."""

    @property
    def name(self) -> str: ...

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]: ...


class DDGSProvider:
    """
    Keyless web search via ``ddgs``.

    ``ddgs`` replaced the frozen ``duckduckgo-search`` package, which by late
    2026 silently returned no results. It is a metasearch library: with
    ``backend="auto"`` it queries several engines (DuckDuckGo, Bing, Brave,
    Mojeek, Wikipedia, ...) and merges their results, which survives one
    engine rate-limiting or blocking far better than DuckDuckGo alone. It
    raises when no engine returns anything.
    """

    def __init__(self, config: SearchConfig | None = None) -> None:
        self._config = config or SearchConfig()

    @property
    def name(self) -> str:
        return f"ddgs:{self._config.backend}"

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        def call() -> list[dict]:
            with DDGS(timeout=self._config.timeout) as ddgs:
                return list(
                    ddgs.text(
                        query,
                        region=self._config.region,
                        safesearch=self._config.safe_search,
                        max_results=max_results,
                        backend=self._config.backend,
                    )
                )

        raw: list[dict] = await asyncio.to_thread(call)
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
                source=self.name,
                rank=rank,
                extra={"raw": item},
            )
            for rank, item in enumerate(raw)
        ]


class SearchTool:
    """
    Cached, concurrency-capped search over a :class:`SearchProvider`.

    Usage::

        tool = SearchTool(SearchConfig(max_results=10, timeout=30))
        results = await tool.search("LangGraph tutorial")

    Returns an empty list on any error rather than raising, so a failed search
    degrades its sub-question instead of killing the run (PRD s6.3).
    """

    def __init__(
        self,
        config: SearchConfig | None = None,
        provider: SearchProvider | None = None,
    ) -> None:
        self._config = config or SearchConfig()
        self._provider: SearchProvider = provider or DDGSProvider(self._config)
        self._cache = _TTLCache(
            self._config.cache_ttl_seconds,
            self._config.cache_max_entries,
        )
        # Shared by every caller, unlike search_multi's per-call bound.
        self._semaphore = asyncio.Semaphore(max(1, self._config.max_concurrency))

    @property
    def config(self) -> SearchConfig:
        return self._config

    @property
    def provider(self) -> SearchProvider:
        return self._provider

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
            self._provider.name,
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

        attempts = max(1, self._config.max_attempts)
        results: list[SearchResult] = []
        for attempt in range(1, attempts + 1):
            try:
                async with self._semaphore:
                    results = await self._provider.search(query, max_results=limit)
                break
            except Exception as exc:  # noqa: BLE001 - degrade, don't crash the run
                if attempt == attempts:
                    logger.warning(
                        "Search failed for query=%r after %d attempt(s): %s", query, attempts, exc
                    )
                    return []
                logger.info("Search attempt %d failed for query=%r: %s", attempt, query, exc)

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
