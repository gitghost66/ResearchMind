"""
M2 tests — web search tool and its TTL cache.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from researchmind_agents.config import SearchConfig
from researchmind_agents.tools.search import SearchResult, SearchTool

FAKE_RESULTS = [
    {"title": "Title 1", "href": "https://example.com/1", "body": "Snippet 1"},
    {"title": "Title 2", "href": "https://example.com/2", "body": "Snippet 2"},
]


def patch_ddgs(results: Any = None, side_effect: Any = None) -> Any:
    """Patch the DDGS context manager used inside the search tool."""
    patcher = patch("researchmind_agents.tools.search.DDGS")
    mock_ddgs = patcher.start()
    if side_effect is not None:
        mock_ddgs.side_effect = side_effect
    else:
        instance = mock_ddgs.return_value.__enter__.return_value
        instance.text.return_value = results if results is not None else FAKE_RESULTS
    return patcher, mock_ddgs


def tool(**overrides: Any) -> SearchTool:
    return SearchTool(SearchConfig(max_results=5, timeout=10, **overrides))


class TestSearch:
    async def test_maps_results_and_ranks(self) -> None:
        patcher, _ = patch_ddgs()
        try:
            results = await tool().search("LangGraph tutorial")
        finally:
            patcher.stop()

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].title == "Title 1"
        assert results[0].url == "https://example.com/1"
        assert results[0].snippet == "Snippet 1"
        assert [r.rank for r in results] == [0, 1]

    async def test_passes_configured_parameters(self) -> None:
        patcher, mock_ddgs = patch_ddgs()
        try:
            await tool(region="us-en", safe_search="moderate").search("q", max_results=3)
        finally:
            patcher.stop()

        kwargs = mock_ddgs.return_value.__enter__.return_value.text.call_args.kwargs
        assert kwargs["region"] == "us-en"
        assert kwargs["safesearch"] == "moderate"
        assert kwargs["max_results"] == 3

    async def test_returns_empty_on_error(self) -> None:
        """A failed search degrades its sub-question; it must not raise."""
        patcher, _ = patch_ddgs(side_effect=Exception("network timeout"))
        try:
            results = await tool().search("anything")
        finally:
            patcher.stop()

        assert results == []

    async def test_search_multi_maps_every_query(self) -> None:
        patcher, _ = patch_ddgs(results=[FAKE_RESULTS[0]])
        try:
            mapping = await tool().search_multi(["q1", "q2"])
        finally:
            patcher.stop()

        assert set(mapping) == {"q1", "q2"}
        assert len(mapping["q1"]) == 1

    async def test_search_multi_empty_input(self) -> None:
        assert await tool().search_multi([]) == {}


class TestCache:
    async def test_second_identical_query_is_served_from_cache(self) -> None:
        """PRD s10: search results are cached with a configurable TTL."""
        search_tool = tool(cache_ttl_seconds=600)
        patcher, mock_ddgs = patch_ddgs()
        try:
            first = await search_tool.search("same query")
            second = await search_tool.search("same query")
        finally:
            patcher.stop()

        assert [r.url for r in first] == [r.url for r in second]
        assert mock_ddgs.call_count == 1
        assert search_tool.cache_size == 1

    async def test_cache_key_ignores_case_and_padding(self) -> None:
        search_tool = tool(cache_ttl_seconds=600)
        patcher, mock_ddgs = patch_ddgs()
        try:
            await search_tool.search("Same Query")
            await search_tool.search("  same query  ")
        finally:
            patcher.stop()

        assert mock_ddgs.call_count == 1

    async def test_different_limit_is_a_different_entry(self) -> None:
        search_tool = tool(cache_ttl_seconds=600)
        patcher, mock_ddgs = patch_ddgs()
        try:
            await search_tool.search("q", max_results=2)
            await search_tool.search("q", max_results=3)
        finally:
            patcher.stop()

        assert mock_ddgs.call_count == 2

    async def test_expired_entry_is_refetched(self) -> None:
        search_tool = tool(cache_ttl_seconds=600)
        patcher, mock_ddgs = patch_ddgs()
        try:
            await search_tool.search("q")
            # Age the entry past its TTL.
            key, (_, value) = next(iter(search_tool._cache._entries.items()))
            search_tool._cache._entries[key] = (-10_000.0, value)
            await search_tool.search("q")
        finally:
            patcher.stop()

        assert mock_ddgs.call_count == 2

    async def test_ttl_zero_disables_cache(self) -> None:
        search_tool = tool(cache_ttl_seconds=0)
        patcher, mock_ddgs = patch_ddgs()
        try:
            await search_tool.search("q")
            await search_tool.search("q")
        finally:
            patcher.stop()

        assert mock_ddgs.call_count == 2
        assert search_tool.cache_size == 0

    async def test_use_cache_false_bypasses_cache(self) -> None:
        search_tool = tool(cache_ttl_seconds=600)
        patcher, mock_ddgs = patch_ddgs()
        try:
            await search_tool.search("q")
            await search_tool.search("q", use_cache=False)
        finally:
            patcher.stop()

        assert mock_ddgs.call_count == 2

    async def test_eviction_respects_max_entries(self) -> None:
        search_tool = tool(cache_ttl_seconds=600, cache_max_entries=2)
        patcher, _ = patch_ddgs()
        try:
            for i in range(5):
                await search_tool.search(f"query {i}")
        finally:
            patcher.stop()

        assert search_tool.cache_size == 2

    async def test_empty_results_are_not_cached(self) -> None:
        """A transient failure must not be remembered as "no results"."""
        search_tool = tool(cache_ttl_seconds=600)
        patcher, _ = patch_ddgs(results=[])
        try:
            await search_tool.search("q")
        finally:
            patcher.stop()

        assert search_tool.cache_size == 0

    async def test_clear_cache(self) -> None:
        search_tool = tool(cache_ttl_seconds=600)
        patcher, _ = patch_ddgs()
        try:
            await search_tool.search("q")
            assert search_tool.cache_size == 1
            search_tool.clear_cache()
        finally:
            patcher.stop()

        assert search_tool.cache_size == 0
