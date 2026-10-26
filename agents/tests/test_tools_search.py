"""
Tests for the web search tool: provider interface, ddgs provider, TTL cache,
shared concurrency cap, and retries.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

from researchmind_agents.config import SearchConfig
from researchmind_agents.tools.search import (
    DDGSProvider,
    SearchResult,
    SearchTool,
)

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
            await tool(region="uk-en", safe_search="moderate", backend="brave,mojeek").search(
                "q", max_results=3
            )
        finally:
            patcher.stop()

        kwargs = mock_ddgs.return_value.__enter__.return_value.text.call_args.kwargs
        assert kwargs["region"] == "uk-en"
        assert kwargs["safesearch"] == "moderate"
        assert kwargs["max_results"] == 3
        assert kwargs["backend"] == "brave,mojeek"

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


# ===========================================================================
# Provider abstraction (M3)
# ===========================================================================


class FakeProvider:
    """Records calls; optionally slow or failing."""

    def __init__(self, name: str = "fake", delay: float = 0.0, error: Exception | None = None) -> None:
        self._name = name
        self._delay = delay
        self._error = error
        self.calls: list[tuple[str, int]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    @property
    def name(self) -> str:
        return self._name

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        self.calls.append((query, max_results))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._error is not None:
                raise self._error
            return [SearchResult(title=query, url=f"https://{self._name}.example/{query}", source=self._name)]
        finally:
            self.in_flight -= 1


class TestProviders:
    def test_ddgs_is_the_default(self) -> None:
        search_tool = tool()
        assert isinstance(search_tool.provider, DDGSProvider)
        # The backend is part of the name, so it is part of the cache key.
        assert search_tool.provider.name == "ddgs:auto"

    async def test_injected_provider_is_used(self) -> None:
        provider = FakeProvider()
        search_tool = SearchTool(SearchConfig(max_results=4), provider=provider)

        results = await search_tool.search("rag")

        assert search_tool.provider is provider
        assert provider.calls == [("rag", 4)]
        assert results[0].source == "fake"

    async def test_ddgs_provider_labels_results(self) -> None:
        patcher, _ = patch_ddgs()
        try:
            results = await DDGSProvider(SearchConfig(backend="bing")).search("q", max_results=2)
        finally:
            patcher.stop()
        assert all(r.source == "ddgs:bing" for r in results)

    async def test_provider_error_degrades_to_empty(self) -> None:
        provider = FakeProvider(error=RuntimeError("quota"))
        search_tool = SearchTool(SearchConfig(), provider=provider)

        assert await search_tool.search("q") == []
        assert search_tool.cache_size == 0

    async def test_cache_key_includes_provider_name(self) -> None:
        search_tool = SearchTool(SearchConfig(cache_ttl_seconds=600), provider=FakeProvider("one"))
        await search_tool.search("q")
        key = next(iter(search_tool._cache._entries))
        assert key[0] == "one"

        # Swapping the provider must not serve the other provider's cached results.
        other = FakeProvider("two")
        search_tool._provider = other
        results = await search_tool.search("q")
        assert other.calls == [("q", search_tool.config.max_results)]
        assert results[0].source == "two"

    async def test_concurrency_is_capped_across_callers(self) -> None:
        """Parallel graph branches share one tool; the cap must hold across them."""
        provider = FakeProvider(delay=0.02)
        search_tool = SearchTool(SearchConfig(max_concurrency=2, cache_ttl_seconds=0), provider=provider)

        await asyncio.gather(*(search_tool.search(f"q{i}") for i in range(6)))

        assert len(provider.calls) == 6
        assert provider.max_in_flight == 2

    async def test_search_multi_respects_shared_cap(self) -> None:
        provider = FakeProvider(delay=0.02)
        search_tool = SearchTool(SearchConfig(max_concurrency=1, cache_ttl_seconds=0), provider=provider)

        await asyncio.gather(
            search_tool.search_multi(["a", "b"], concurrency=4),
            search_tool.search_multi(["c", "d"], concurrency=4),
        )

        assert provider.max_in_flight == 1

    async def test_cache_hits_skip_the_provider(self) -> None:
        provider = FakeProvider()
        search_tool = SearchTool(SearchConfig(cache_ttl_seconds=600), provider=provider)
        await search_tool.search("q")
        await search_tool.search("Q ")
        assert len(provider.calls) == 1


class FlakyProvider:
    """Fails the first *failures* calls, then succeeds."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    @property
    def name(self) -> str:
        return "flaky"

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("No results found.")
        return [SearchResult(title=query, url="https://ok.example", source=self.name)]


class TestRetries:
    async def test_error_is_retried_and_success_is_returned(self) -> None:
        provider = FlakyProvider(failures=1)
        search_tool = SearchTool(SearchConfig(max_attempts=2), provider=provider)

        results = await search_tool.search("q")

        assert provider.calls == 2
        assert [r.url for r in results] == ["https://ok.example"]

    async def test_gives_up_after_max_attempts(self) -> None:
        provider = FlakyProvider(failures=10)
        search_tool = SearchTool(SearchConfig(max_attempts=3), provider=provider)

        assert await search_tool.search("q") == []
        assert provider.calls == 3

    async def test_single_attempt_does_not_retry(self) -> None:
        provider = FlakyProvider(failures=1)
        search_tool = SearchTool(SearchConfig(max_attempts=1), provider=provider)

        assert await search_tool.search("q") == []
        assert provider.calls == 1

    async def test_empty_results_are_not_retried(self) -> None:
        patcher, mock_ddgs = patch_ddgs(results=[])
        try:
            results = await tool(max_attempts=3).search("q")
        finally:
            patcher.stop()
        assert results == []
        assert mock_ddgs.return_value.__enter__.return_value.text.call_count == 1
