"""
M2 tests — LLM client, embedding service, search tool, page extractor,
and vector search tool.

All LLM / network calls are mocked; no real API keys required.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from researchmind_agents.llm.client import GeminiClient, LLMResponse, LLMUsage
from researchmind_agents.llm.embeddings import EmbeddingResult, EmbeddingService
from researchmind_agents.tools.page_extract import PageContent, PageExtractTool, SSRFError
from researchmind_agents.tools.search import SearchResult, SearchTool
from researchmind_agents.tools.vector_search import (
    VectorSearchResult,
    VectorSearchTool,
    make_vector_search_fn,
)


# ===========================================================================
# Helpers / fixtures
# ===========================================================================


def _make_genai_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
    """Build a fake google-genai GenerateContentResponse."""
    part = MagicMock()
    part.text = text

    candidate = MagicMock()
    candidate.content.parts = [part]

    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = completion_tokens
    usage.total_token_count = prompt_tokens + completion_tokens

    resp = MagicMock()
    resp.candidates = [candidate]
    resp.usage_metadata = usage
    resp.parsed = None
    return resp


def _make_embed_response(vector: list[float]) -> MagicMock:
    """Build a fake google-genai EmbedContentResponse."""
    embedding = MagicMock()
    embedding.values = vector

    resp = MagicMock()
    resp.embeddings = [embedding]
    resp.usage_metadata = None
    return resp


# ===========================================================================
# GeminiClient tests
# ===========================================================================


class TestGeminiClient:
    @pytest.fixture
    def client(self) -> GeminiClient:
        return GeminiClient(api_key="test-key", model="gemini-3.1-flash-lite")

    @pytest.mark.asyncio
    async def test_generate_plain_text(self, client: GeminiClient) -> None:
        """generate() returns LLMResponse with text and usage populated."""
        fake_resp = _make_genai_response("Hello world")

        with patch.object(
            client._client.models,
            "generate_content",
            return_value=fake_resp,
        ):
            result = await client.generate("Say hello")

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello world"
        assert result.parsed is None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.usage.model == "gemini-3.1-flash-lite"

    @pytest.mark.asyncio
    async def test_generate_structured_valid_json(self, client: GeminiClient) -> None:
        """generate_structured() parses JSON into a Pydantic model."""

        class Answer(BaseModel):
            score: int
            reason: str

        payload = json.dumps({"score": 8, "reason": "solid evidence"})
        fake_resp = _make_genai_response(payload)

        with patch.object(
            client._client.models,
            "generate_content",
            return_value=fake_resp,
        ):
            result = await client.generate_structured("Rate this", Answer)

        assert isinstance(result.parsed, Answer)
        assert result.parsed.score == 8
        assert result.parsed.reason == "solid evidence"

    @pytest.mark.asyncio
    async def test_generate_structured_invalid_json_raises(self, client: GeminiClient) -> None:
        """generate_structured() raises ValidationError on bad JSON."""
        from pydantic import ValidationError

        class Answer(BaseModel):
            score: int

        fake_resp = _make_genai_response("not json at all {{")
        fake_resp.parsed = None

        with patch.object(
            client._client.models,
            "generate_content",
            return_value=fake_resp,
        ):
            with pytest.raises((ValidationError, ValueError)):
                await client.generate_structured("Rate", Answer)

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, client: GeminiClient) -> None:
        """Client retries on rate-limit errors then succeeds."""
        fake_resp = _make_genai_response("Retry worked")
        call_count = 0

        def _side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("429 quota exceeded — rate limit")
            return fake_resp

        with patch.object(
            client._client.models,
            "generate_content",
            side_effect=_side_effect,
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):  # don't actually sleep
                result = await client.generate("test")

        assert result.text == "Retry worked"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_transient_error(self, client: GeminiClient) -> None:
        """Client does NOT retry on non-transient (e.g. auth) errors."""
        call_count = 0

        def _side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            raise Exception("invalid API key — authentication failed")

        with patch.object(
            client._client.models,
            "generate_content",
            side_effect=_side_effect,
        ):
            with pytest.raises(Exception, match="invalid API key"):
                await client.generate("test")

        assert call_count == 1

    def test_usage_from_metadata_none(self) -> None:
        """LLMUsage.from_metadata handles None gracefully."""
        usage = LLMUsage.from_metadata(None, model="m", latency_ms=5.0)
        assert usage.total_tokens == 0
        assert usage.model == "m"


# ===========================================================================
# EmbeddingService tests
# ===========================================================================


class TestEmbeddingService:
    @pytest.fixture
    def service(self) -> EmbeddingService:
        return EmbeddingService(api_key="test-key", model="text-embedding-004", dimension=768)

    @pytest.mark.asyncio
    async def test_embed_query(self, service: EmbeddingService) -> None:
        vector = [0.1] * 768
        fake_resp = _make_embed_response(vector)

        with patch.object(service._client.models, "embed_content", return_value=fake_resp):
            result = await service.embed_query("What is RAG?")

        assert isinstance(result, EmbeddingResult)
        assert len(result.vector) == 768
        assert result.vector[0] == pytest.approx(0.1)
        assert result.model == "text-embedding-004"

    @pytest.mark.asyncio
    async def test_embed_document(self, service: EmbeddingService) -> None:
        vector = [0.5] * 768
        fake_resp = _make_embed_response(vector)

        with patch.object(service._client.models, "embed_content", return_value=fake_resp):
            result = await service.embed_document("Some long document...")

        assert len(result.vector) == 768

    @pytest.mark.asyncio
    async def test_embed_documents_batch(self, service: EmbeddingService) -> None:
        """embed_documents returns one result per input text."""
        texts = ["doc1", "doc2", "doc3"]
        vector = [0.2] * 768
        fake_resp = _make_embed_response(vector)

        with patch.object(service._client.models, "embed_content", return_value=fake_resp):
            results = await service.embed_documents(texts)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, EmbeddingResult)

    @pytest.mark.asyncio
    async def test_embed_documents_empty(self, service: EmbeddingService) -> None:
        results = await service.embed_documents([])
        assert results == []

    def test_dimension_property(self, service: EmbeddingService) -> None:
        assert service.dimension == 768


# ===========================================================================
# SearchTool tests
# ===========================================================================


class TestSearchTool:
    @pytest.fixture
    def tool(self) -> SearchTool:
        return SearchTool(max_results=5, timeout=10)

    @pytest.mark.asyncio
    async def test_search_returns_results(self, tool: SearchTool) -> None:
        fake_results = [
            {"title": "Title 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Title 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]

        with patch("researchmind_agents.tools.search.DDGS") as MockDDGS:
            instance = MockDDGS.return_value.__enter__.return_value
            instance.text.return_value = fake_results

            results = await tool.search("LangGraph tutorial")

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].title == "Title 1"
        assert results[0].url == "https://example.com/1"
        assert results[0].snippet == "Snippet 1"
        assert results[0].rank == 0
        assert results[1].rank == 1

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_error(self, tool: SearchTool) -> None:
        """Search gracefully returns [] on network errors."""
        with patch("researchmind_agents.tools.search.DDGS") as MockDDGS:
            MockDDGS.side_effect = Exception("network timeout")
            results = await tool.search("anything")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_multi(self, tool: SearchTool) -> None:
        fake = [{"title": "T", "href": "https://x.com", "body": "s"}]

        with patch("researchmind_agents.tools.search.DDGS") as MockDDGS:
            instance = MockDDGS.return_value.__enter__.return_value
            instance.text.return_value = fake

            mapping = await tool.search_multi(["q1", "q2"])

        assert set(mapping.keys()) == {"q1", "q2"}
        assert len(mapping["q1"]) == 1


# ===========================================================================
# PageExtractTool tests
# ===========================================================================


class TestPageExtractTool:
    @pytest.fixture
    def tool(self) -> PageExtractTool:
        return PageExtractTool(timeout=5, max_size_bytes=512_000)

    def test_validate_url_blocks_non_http(self, tool: PageExtractTool) -> None:
        with pytest.raises(ValueError, match="scheme"):
            tool._validate_url("ftp://evil.com/file")

    def test_validate_url_blocks_no_host(self, tool: PageExtractTool) -> None:
        with pytest.raises(ValueError, match="host"):
            tool._validate_url("http:///no-host")

    @pytest.mark.asyncio
    async def test_ssrf_blocks_localhost(self, tool: PageExtractTool) -> None:
        """Check SSRF detection blocks loopback."""
        with patch("socket.getaddrinfo") as mock_ga:
            mock_ga.return_value = [
                (None, None, None, None, ("127.0.0.1", 80))
            ]
            with pytest.raises(SSRFError):
                await tool._check_ssrf("http://localhost/admin")

    @pytest.mark.asyncio
    async def test_ssrf_blocks_private_ip(self, tool: PageExtractTool) -> None:
        with patch("socket.getaddrinfo") as mock_ga:
            mock_ga.return_value = [
                (None, None, None, None, ("192.168.1.100", 80))
            ]
            with pytest.raises(SSRFError):
                await tool._check_ssrf("http://internal.corp/secret")

    @pytest.mark.asyncio
    async def test_ssrf_allows_public_ip(self, tool: PageExtractTool) -> None:
        with patch("socket.getaddrinfo") as mock_ga:
            mock_ga.return_value = [
                (None, None, None, None, ("93.184.216.34", 80))  # example.com
            ]
            # Should not raise
            await tool._check_ssrf("http://example.com")

    def test_clean_html_extracts_text(self, tool: PageExtractTool) -> None:
        html = """
        <html>
          <head><title>Test Page</title></head>
          <body>
            <nav>Navigation</nav>
            <main>
              <h1>Article Title</h1>
              <p>This is the article content.</p>
              <script>alert('bad')</script>
            </main>
          </body>
        </html>
        """
        result = tool._clean_html(html)
        assert result["title"] == "Test Page"
        assert "Article Title" in result["text"]
        assert "article content" in result["text"]
        # Scripts should be stripped
        assert "alert" not in result["text"]
        # Nav should be stripped
        assert "Navigation" not in result["text"]

    @pytest.mark.asyncio
    async def test_extract_returns_error_on_ssrf(self, tool: PageExtractTool) -> None:
        """extract() returns PageContent with error set instead of raising."""
        with patch("socket.getaddrinfo") as mock_ga:
            mock_ga.return_value = [(None, None, None, None, ("10.0.0.1", 80))]
            result = await tool.extract("http://internal/secret")

        assert isinstance(result, PageContent)
        assert not result.ok
        assert "SSRF" in result.error

    @pytest.mark.asyncio
    async def test_extract_returns_error_on_http_failure(self, tool: PageExtractTool) -> None:
        """extract() returns error PageContent on HTTP errors."""
        with patch("socket.getaddrinfo") as mock_ga:
            mock_ga.return_value = [(None, None, None, None, ("93.184.216.34", 80))]

            with patch("httpx.AsyncClient") as MockClient:
                MockClient.return_value.__aenter__.side_effect = Exception("connection refused")
                result = await tool.extract("http://example.com/page")

        assert not result.ok
        assert result.error != ""

    @pytest.mark.asyncio
    async def test_extract_full_success(self, tool: PageExtractTool) -> None:
        """Happy-path: mock HTTP response, verify PageContent populated."""
        html = b"<html><head><title>My Page</title></head><body><main><p>Hello ResearchMind!</p></main></body></html>"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = html

        with patch("socket.getaddrinfo") as mock_ga:
            mock_ga.return_value = [(None, None, None, None, ("93.184.216.34", 80))]

            with patch("httpx.AsyncClient") as MockClient:
                mock_ctx = AsyncMock()
                MockClient.return_value.__aenter__ = mock_ctx
                mock_ctx.return_value.get = AsyncMock(return_value=mock_response)
                MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

                result = await tool.extract("http://example.com/page")

        assert result.title == "My Page"
        assert "Hello ResearchMind" in result.text
        assert result.ok


# ===========================================================================
# VectorSearchTool tests
# ===========================================================================


class TestVectorSearchTool:
    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        expected = [
            VectorSearchResult(
                passage_id="p1",
                source_id="s1",
                text="Some passage text",
                score=0.95,
                metadata={},
            )
        ]

        async def fake_fn(vector: list[float], top_k: int, run_id: str | None) -> list[VectorSearchResult]:
            return expected

        tool = VectorSearchTool(search_fn=fake_fn)
        results = await tool.search([0.1] * 768, top_k=5, run_id="run-123")

        assert len(results) == 1
        assert results[0].passage_id == "p1"
        assert results[0].score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_error(self) -> None:
        async def failing_fn(vector: list[float], top_k: int, run_id: str | None) -> list[VectorSearchResult]:
            raise RuntimeError("db connection lost")

        tool = VectorSearchTool(search_fn=failing_fn)
        results = await tool.search([0.0] * 768, top_k=3)

        assert results == []

    @pytest.mark.asyncio
    async def test_make_vector_search_fn_adapter(self) -> None:
        """make_vector_search_fn correctly adapts a mock PassageRepository."""

        class FakePassage:
            id = "p-abc"
            source_id = "src-1"
            text = "passage text"
            chunk_index = 0
            token_count = 42
            _similarity_score = 0.88

        mock_repo = AsyncMock()
        mock_repo.search_similar.return_value = [FakePassage()]

        fn = make_vector_search_fn(mock_repo)
        results = await fn([0.1] * 768, 5, "run-xyz")

        assert len(results) == 1
        assert results[0].passage_id == "p-abc"
        assert results[0].score == pytest.approx(0.88)
        assert results[0].text == "passage text"
        mock_repo.search_similar.assert_called_once_with(
            embedding=[0.1] * 768, limit=5, run_id="run-xyz"
        )
