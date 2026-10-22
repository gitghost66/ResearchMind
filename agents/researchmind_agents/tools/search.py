"""
DuckDuckGo web search tool.

Returns a list of SearchResult objects — no LLM calls, no DB access.
Respects the configured timeout and result cap from Settings.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single web search result."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = "duckduckgo"
    rank: int = 0
    extra: dict = field(default_factory=dict)


class SearchTool:
    """
    Async DuckDuckGo search wrapper.

    Usage::

        tool = SearchTool(max_results=10, timeout=30)
        results = await tool.search("LangGraph tutorial")
    """

    def __init__(
        self,
        max_results: int = 10,
        timeout: int = 30,
        region: str = "wt-wt",
        safe_search: str = "off",
    ) -> None:
        self._max_results = max_results
        self._timeout = timeout
        self._region = region
        self._safe_search = safe_search

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """
        Run a DuckDuckGo text search and return ranked results.

        Runs the blocking DDGS call in a thread to stay non-blocking.
        Returns an empty list on any error rather than raising (so the
        research pipeline can continue with partial data).
        """
        n = max_results or self._max_results

        def _sync_search() -> list[dict]:
            with DDGS(timeout=self._timeout) as ddgs:
                return list(
                    ddgs.text(
                        query,
                        region=self._region,
                        safesearch=self._safe_search,
                        max_results=n,
                    )
                )

        try:
            raw: list[dict] = await asyncio.to_thread(_sync_search)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DuckDuckGo search failed for query=%r: %s", query, exc)
            return []

        results: list[SearchResult] = []
        for rank, item in enumerate(raw):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                    source="duckduckgo",
                    rank=rank,
                    extra={"raw": item},
                )
            )

        logger.debug("Search query=%r returned %d results", query, len(results))
        return results

    async def search_multi(
        self,
        queries: list[str],
        *,
        max_results_each: int | None = None,
    ) -> dict[str, list[SearchResult]]:
        """
        Run multiple queries concurrently.

        Returns a mapping of query → results.
        """
        tasks = [self.search(q, max_results=max_results_each) for q in queries]
        all_results = await asyncio.gather(*tasks)
        return dict(zip(queries, all_results, strict=False))
