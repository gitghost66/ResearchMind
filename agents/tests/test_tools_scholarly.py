"""
M5 tests — scholarly search (arXiv, Semantic Scholar) for the academic researcher.

Every HTTP call goes to an ``httpx.MockTransport``; nothing touches the network.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest

from researchmind_agents.config import AcademicSearchConfig
from researchmind_agents.tools.scholarly import (
    ArxivProvider,
    Paper,
    ScholarlySearchTool,
    SemanticScholarProvider,
    arxiv_search_query,
    build_scholarly_providers,
    merge_papers,
    parse_arxiv_feed,
    parse_semantic_scholar,
)

CONFIG = AcademicSearchConfig(
    arxiv_url="https://arxiv.test/api/query",
    semantic_scholar_url="https://s2.test/graph/v1/paper/search",
    arxiv_min_interval=0.0,
    semantic_scholar_min_interval=0.0,
    max_attempts=2,
)

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>arXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2401.01234v3</id>
    <published>2024-01-02T18:00:00Z</published>
    <title>Retrieval-Augmented
      Generation for Enterprise QA</title>
    <summary>  We study retrieval augmented generation
      on proprietary documents.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:journal_ref>ACL 2024</arxiv:journal_ref>
    <arxiv:doi>10.1000/rag.2024</arxiv:doi>
  </entry>
  <entry>
    <id>https://arxiv.org/abs/2305.00001v1</id>
    <published>2023-05-01T00:00:00Z</published>
    <title>Fine-tuning Large Language Models</title>
    <summary>A survey of fine-tuning methods.</summary>
    <author><name>Grace Hopper</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2309.99999v1</id>
    <title>   </title>
    <summary>No title, so skipped.</summary>
  </entry>
</feed>
"""


def s2_payload(*items: dict[str, Any]) -> dict[str, Any]:
    return {"total": len(items), "offset": 0, "data": list(items)}


S2_ITEMS = [
    {
        "paperId": "abc",
        "title": "Retrieval-Augmented Generation for Enterprise QA",
        "abstract": None,
        "tldr": {"model": "tldr@v2", "text": "RAG beats fine-tuning on fresh documents."},
        "url": "https://www.semanticscholar.org/paper/abc",
        "year": 2024,
        "venue": "ACL",
        "publicationDate": "2024-01-02",
        "citationCount": 42,
        "externalIds": {"ArXiv": "2401.01234", "DOI": "10.1000/rag.2024"},
        "authors": [{"authorId": "1", "name": "Ada Lovelace"}],
    },
    {
        "paperId": "def",
        "title": "Domain Adaptation by Fine-Tuning",
        "abstract": "We fine-tune models on domain data.",
        "url": "https://www.semanticscholar.org/paper/def",
        "year": 2022,
        "venue": "",
        "publicationDate": None,
        "citationCount": 7,
        "externalIds": {"DOI": "10.2000/FT.2022"},
        "authors": [],
    },
    {
        "paperId": "ghi",
        "title": "A Paper Without Identifiers",
        "abstract": "Plain abstract.",
        "url": "https://www.semanticscholar.org/paper/ghi",
        "year": None,
        "externalIds": {},
    },
    {"paperId": "nope", "title": "", "abstract": "no title"},
]


Handler = Callable[[httpx.Request], httpx.Response]


def client_for(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


def router(requests: list[httpx.Request] | None = None, **routes: Handler) -> Handler:
    """Dispatch by host: ``arxiv=...`` for arxiv.test, ``s2=...`` for s2.test."""

    def handle(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        name = "arxiv" if request.url.host == "arxiv.test" else "s2"
        return routes[name](request)

    return handle


def arxiv_ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=ARXIV_FEED.encode(), headers={"content-type": "application/atom+xml"})


def s2_ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=s2_payload(*S2_ITEMS))


def build_tool(handler: Handler, config: AcademicSearchConfig = CONFIG) -> ScholarlySearchTool:
    return ScholarlySearchTool(config, build_scholarly_providers(config, client_for(handler)))


# ===========================================================================
# arXiv
# ===========================================================================


class TestArxivQuery:
    def test_significant_words_are_anded(self) -> None:
        query = arxiv_search_query("What is the effect of RAG versus fine-tuning on QA accuracy?")
        assert query == "all:rag AND all:fine AND all:tuning AND all:accuracy"

    def test_terms_are_capped_and_deduplicated(self) -> None:
        query = arxiv_search_query("alpha beta gamma delta alpha epsilon zeta theta iota")
        assert query.count("all:") == 6
        assert query.count("all:alpha") == 1

    def test_no_significant_words_falls_back_to_the_query(self) -> None:
        assert arxiv_search_query("is it") == "all:is it"


class TestArxivParsing:
    def test_entries_are_parsed(self) -> None:
        papers = parse_arxiv_feed(ARXIV_FEED)

        assert [p.arxiv_id for p in papers] == ["2401.01234", "2305.00001"]
        first = papers[0]
        assert first.title == "Retrieval-Augmented Generation for Enterprise QA"
        assert first.abstract == "We study retrieval augmented generation on proprietary documents."
        assert first.url == "https://arxiv.org/abs/2401.01234"
        assert first.authors == ["Ada Lovelace", "Alan Turing"]
        assert first.published_at == "2024-01-02T18:00:00+00:00"
        assert first.year == 2024
        assert first.venue == "ACL 2024"
        assert first.doi == "10.1000/rag.2024"
        assert first.provider == "arxiv"
        assert [p.rank for p in papers] == [0, 1]

    def test_missing_optional_fields_are_empty(self) -> None:
        second = parse_arxiv_feed(ARXIV_FEED)[1]
        assert second.venue == ""
        assert second.doi == ""
        assert second.citation_count is None

    async def test_provider_sends_the_api_parameters(self) -> None:
        requests: list[httpx.Request] = []
        provider = ArxivProvider(CONFIG, client_for(router(requests, arxiv=arxiv_ok)))

        papers = await provider.search("RAG enterprise documents", max_results=1)

        assert len(papers) == 1
        params = requests[0].url.params
        assert params["search_query"] == "all:rag AND all:enterprise AND all:documents"
        assert params["max_results"] == "1"
        assert params["start"] == "0"
        assert params["sortBy"] == "relevance"

    async def test_malformed_feed_raises(self) -> None:
        provider = ArxivProvider(
            CONFIG, client_for(router(arxiv=lambda _: httpx.Response(200, content=b"<feed><oops")))
        )
        # The provider raises; the tool turns it into "no papers" (TestTool).
        with pytest.raises(ET.ParseError):
            await provider.search("rag", max_results=5)


# ===========================================================================
# Semantic Scholar
# ===========================================================================


class TestSemanticScholarParsing:
    def test_items_are_parsed(self) -> None:
        papers = parse_semantic_scholar(s2_payload(*S2_ITEMS))

        assert [p.title for p in papers] == [
            "Retrieval-Augmented Generation for Enterprise QA",
            "Domain Adaptation by Fine-Tuning",
            "A Paper Without Identifiers",
        ]
        first, second, third = papers
        # No abstract: the TLDR stands in.
        assert first.abstract == "RAG beats fine-tuning on fresh documents."
        assert first.arxiv_id == "2401.01234"
        assert first.url == "https://arxiv.org/abs/2401.01234"
        assert first.citation_count == 42
        assert first.published_at == "2024-01-02T00:00:00+00:00"
        assert first.authors == ["Ada Lovelace"]
        # DOI but no arXiv ID: the DOI resolver is the URL; the year gives the date.
        assert second.url == "https://doi.org/10.2000/FT.2022"
        assert second.published_at == "2022-01-01T00:00:00+00:00"
        assert second.year == 2022
        # Neither: Semantic Scholar's own page.
        assert third.url == "https://www.semanticscholar.org/paper/ghi"
        assert third.published_at == ""
        assert third.year is None

    def test_unexpected_payloads_give_no_papers(self) -> None:
        assert parse_semantic_scholar({"message": "Too Many Requests"}) == []
        assert parse_semantic_scholar([]) == []
        assert parse_semantic_scholar({"data": ["junk", None]}) == []

    async def test_api_key_header_only_when_configured(self) -> None:
        requests: list[httpx.Request] = []
        handler = router(requests, s2=s2_ok)

        await SemanticScholarProvider(CONFIG, client_for(handler)).search("rag", max_results=3)
        keyed = replace(CONFIG, semantic_scholar_api_key="secret")
        await SemanticScholarProvider(keyed, client_for(handler)).search("rag", max_results=3)

        assert "x-api-key" not in requests[0].headers
        assert requests[1].headers["x-api-key"] == "secret"
        params = requests[0].url.params
        assert params["query"] == "rag"
        assert params["limit"] == "3"
        assert "citationCount" in params["fields"]

    async def test_results_are_capped_at_max_results(self) -> None:
        provider = SemanticScholarProvider(CONFIG, client_for(router(s2=s2_ok)))
        assert len(await provider.search("rag", max_results=2)) == 2


# ===========================================================================
# Shared HTTP behaviour
# ===========================================================================


class TestHttp:
    async def test_oversized_response_is_rejected(self) -> None:
        small = replace(CONFIG, max_response_bytes=100)
        provider = ArxivProvider(small, client_for(router(arxiv=arxiv_ok)))
        with pytest.raises(ValueError, match="exceeds"):
            await provider.search("rag", max_results=5)

    async def test_http_error_raises(self) -> None:
        provider = SemanticScholarProvider(
            CONFIG, client_for(router(s2=lambda _: httpx.Response(429)))
        )
        with pytest.raises(httpx.HTTPStatusError):
            await provider.search("rag", max_results=5)

    async def test_redirects_are_not_followed(self) -> None:
        def redirect(_: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

        provider = ArxivProvider(CONFIG, client_for(router(arxiv=redirect)))
        with pytest.raises(httpx.HTTPStatusError):
            await provider.search("rag", max_results=5)

    async def test_owned_client_is_created_lazily_and_closed(self) -> None:
        provider = ArxivProvider(CONFIG)
        assert provider._client is None
        client = provider._ensure_client()
        assert client.headers["User-Agent"] == CONFIG.user_agent
        await provider.aclose()
        assert client.is_closed
        assert provider._client is None

    async def test_injected_client_is_not_closed(self) -> None:
        client = client_for(router(arxiv=arxiv_ok))
        provider = ArxivProvider(CONFIG, client)
        await provider.aclose()
        assert not client.is_closed
        assert await provider.search("rag", max_results=5)


# ===========================================================================
# Tool: merging, degradation, cache
# ===========================================================================


class TestTool:
    async def test_merges_providers_round_robin_and_dedupes(self) -> None:
        tool = build_tool(router(arxiv=arxiv_ok, s2=s2_ok))

        papers = await tool.search("rag fine tuning")

        keys = [p.key for p in papers]
        assert len(keys) == len(set(keys))
        # arXiv rank 0, then S2 rank 0 (a duplicate of it), arXiv rank 1, S2 rank 1, ...
        assert [p.title for p in papers] == [
            "Retrieval-Augmented Generation for Enterprise QA",
            "Fine-tuning Large Language Models",
            "Domain Adaptation by Fine-Tuning",
            "A Paper Without Identifiers",
        ]
        assert [p.rank for p in papers] == [0, 1, 2, 3]

    async def test_duplicate_fills_fields_the_first_copy_lacks(self) -> None:
        tool = build_tool(router(arxiv=arxiv_ok, s2=s2_ok))

        rag = (await tool.search("rag"))[0]

        assert rag.provider == "arxiv"  # first copy wins
        assert rag.abstract.startswith("We study")  # its own abstract is kept
        assert rag.citation_count == 42  # filled from Semantic Scholar

    async def test_failing_provider_degrades_to_the_others(self) -> None:
        tool = build_tool(router(arxiv=lambda _: httpx.Response(503), s2=s2_ok))

        papers = await tool.search("rag")

        assert papers
        assert {p.provider for p in papers} == {"semantic_scholar"}

    async def test_every_provider_failing_gives_an_empty_list(self) -> None:
        def boom(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        assert await build_tool(router(arxiv=boom, s2=boom)).search("rag") == []

    async def test_each_provider_is_retried_up_to_max_attempts(self) -> None:
        requests: list[httpx.Request] = []
        tool = build_tool(router(requests, arxiv=lambda _: httpx.Response(500), s2=s2_ok))

        await tool.search("rag")

        assert sum(1 for r in requests if r.url.host == "arxiv.test") == CONFIG.max_attempts

    async def test_a_retry_can_succeed(self) -> None:
        calls = {"n": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500) if calls["n"] == 1 else arxiv_ok(request)

        tool = build_tool(router(arxiv=flaky, s2=lambda _: httpx.Response(200, json=s2_payload())))

        papers = await tool.search("rag")

        assert [p.arxiv_id for p in papers] == ["2401.01234", "2305.00001"]

    async def test_cache_hit_skips_the_providers(self) -> None:
        requests: list[httpx.Request] = []
        tool = build_tool(router(requests, arxiv=arxiv_ok, s2=s2_ok))

        first = await tool.search("RAG")
        second = await tool.search("  rag ")

        assert [p.key for p in first] == [p.key for p in second]
        assert len(requests) == 2  # one per provider, first call only
        assert tool.cache_size == 1

    async def test_empty_results_are_not_cached(self) -> None:
        requests: list[httpx.Request] = []
        empty = router(
            requests,
            arxiv=lambda _: httpx.Response(200, content=b'<feed xmlns="http://www.w3.org/2005/Atom"/>'),
            s2=lambda _: httpx.Response(200, json=s2_payload()),
        )
        tool = build_tool(empty)

        await tool.search("rag")
        await tool.search("rag")

        assert len(requests) == 4

    async def test_cache_disabled_with_zero_ttl(self) -> None:
        requests: list[httpx.Request] = []
        no_cache = replace(CONFIG, cache_ttl_seconds=0)
        tool = build_tool(router(requests, arxiv=arxiv_ok, s2=s2_ok), no_cache)

        await tool.search("rag")
        await tool.search("rag")

        assert len(requests) == 4

    async def test_search_multi(self) -> None:
        tool = build_tool(router(arxiv=arxiv_ok, s2=s2_ok))

        results = await tool.search_multi(["rag", "fine tuning"])

        assert set(results) == {"rag", "fine tuning"}
        assert all(results.values())
        assert await tool.search_multi([]) == {}

    async def test_aclose_closes_owned_clients_only(self) -> None:
        injected = client_for(router(arxiv=arxiv_ok, s2=s2_ok))
        tool = ScholarlySearchTool(CONFIG, build_scholarly_providers(CONFIG, injected))
        await tool.aclose()
        assert not injected.is_closed

        owned = ScholarlySearchTool(CONFIG)
        clients = [p._ensure_client() for p in owned.providers]  # type: ignore[attr-defined]
        await owned.aclose()
        assert all(c.is_closed for c in clients)


class TestProviders:
    def test_default_providers(self) -> None:
        names = [p.name for p in build_scholarly_providers(AcademicSearchConfig())]
        assert names == ["arxiv", "semantic_scholar"]

    def test_tool_builds_providers_from_config(self) -> None:
        tool = ScholarlySearchTool(replace(CONFIG, providers=("semantic_scholar",)))
        assert [p.name for p in tool.providers] == ["semantic_scholar"]

    def test_unknown_provider_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="Unknown academic search provider"):
            build_scholarly_providers(replace(CONFIG, providers=("arxiv", "openalex")))


class TestPaper:
    def test_key_prefers_arxiv_then_doi_then_title(self) -> None:
        assert Paper(title="X", arxiv_id="2401.01234", doi="10.1/A").key == "arxiv:2401.01234"
        assert Paper(title="X", doi="10.1/AbC").key == "doi:10.1/abc"
        assert Paper(title="  Deep  Learning: A Survey! ").key == "title:deep learning a survey"

    def test_merge_does_not_mutate_inputs(self) -> None:
        a = Paper(title="Same", doi="10.1/x", provider="a")
        b = Paper(title="Same", doi="10.1/x", provider="b", citation_count=5, venue="V")

        merged = merge_papers([[a], [b]])

        assert len(merged) == 1
        assert merged[0].citation_count == 5
        assert merged[0].venue == "V"
        assert a.citation_count is None and a.venue == ""

    def test_merge_of_nothing(self) -> None:
        assert merge_papers([]) == []
        assert merge_papers([[], []]) == []
