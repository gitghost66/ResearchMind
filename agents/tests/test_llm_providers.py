"""
M2 tests — provider-agnostic LLM wrapper.

Covers the pieces M2's acceptance criteria name: structured output, retries,
and usage/cost recording. All network calls are mocked; no API keys required.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from researchmind_agents.config import LLMConfig
from researchmind_agents.llm.base import (
    LLMUsage,
    StructuredOutputError,
    TransientLLMError,
    extract_json,
)
from researchmind_agents.llm.client import GeminiClient, GeminiProvider
from researchmind_agents.llm.factory import available_providers, build_llm_provider
from researchmind_agents.llm.openai_compatible import OpenAICompatibleProvider
from researchmind_agents.llm.pricing import compute_cost_usd, resolve_pricing


class Answer(BaseModel):
    score: int
    reason: str


def make_genai_response(
    text: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    """Build a fake google-genai GenerateContentResponse."""
    part = MagicMock()
    part.text = text

    candidate = MagicMock()
    candidate.content.parts = [part]
    candidate.finish_reason = "STOP"

    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = completion_tokens
    usage.total_token_count = prompt_tokens + completion_tokens

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = usage
    response.parsed = None
    return response


def gemini(**overrides: Any) -> GeminiProvider:
    """A Gemini provider with retry sleeps kept short and no usage sink."""
    overrides.setdefault("api_key", "test-key")
    overrides.setdefault("model", "gemini-3.1-flash-lite")
    overrides.setdefault("base_delay", 0.0)
    return GeminiProvider(LLMConfig(**overrides))


# ===========================================================================
# Plain generation and usage recording
# ===========================================================================


class TestGenerate:
    async def test_returns_text_and_usage(self) -> None:
        provider = gemini()
        with patch.object(
            provider._client.models, "generate_content", return_value=make_genai_response("Hi")
        ):
            result = await provider.generate("Say hi", agent="planner", run_id="run-1")

        assert result.text == "Hi"
        assert result.parsed is None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.usage.model == "gemini-3.1-flash-lite"
        assert result.usage.provider == "gemini"
        assert result.usage.agent == "planner"
        assert result.usage.run_id == "run-1"
        assert result.usage.latency_ms >= 0

    async def test_usage_pushed_to_sink(self) -> None:
        """M2 acceptance: the wrapper records usage rows."""
        recorded: list[LLMUsage] = []

        async def sink(usage: LLMUsage) -> None:
            recorded.append(usage)

        config = LLMConfig(api_key="k", model="gemini-3.1-flash-lite", base_delay=0.0)
        provider = GeminiProvider(config, usage_sink=sink)

        with patch.object(
            provider._client.models, "generate_content", return_value=make_genai_response("ok")
        ):
            await provider.generate("prompt", agent="web_research", run_id="run-7")

        assert len(recorded) == 1
        row = recorded[0].as_usage_row()
        assert row["run_id"] == "run-7"
        assert row["agent"] == "web_research"
        assert row["model"] == "gemini-3.1-flash-lite"
        assert row["tokens_in"] == 10
        assert row["tokens_out"] == 20
        assert isinstance(row["latency_ms"], int)

    async def test_sink_failure_does_not_break_the_call(self) -> None:
        """Accounting is best-effort; it must never fail a research run."""

        async def failing_sink(usage: LLMUsage) -> None:
            raise RuntimeError("db down")

        config = LLMConfig(api_key="k", base_delay=0.0)
        provider = GeminiProvider(config, usage_sink=failing_sink)

        with patch.object(
            provider._client.models, "generate_content", return_value=make_genai_response("ok")
        ):
            result = await provider.generate("prompt", run_id="r")

        assert result.text == "ok"

    async def test_configured_price_produces_cost(self) -> None:
        provider = gemini(price_per_1m_input=1.0, price_per_1m_output=2.0)
        with patch.object(
            provider._client.models, "generate_content", return_value=make_genai_response("ok")
        ):
            result = await provider.generate("prompt", run_id="r")

        # 10 in @ $1/1M + 20 out @ $2/1M
        assert result.usage.cost_usd == pytest.approx((10 * 1.0 + 20 * 2.0) / 1_000_000)

    async def test_unpriced_model_costs_zero(self) -> None:
        """Free-tier usage genuinely costs nothing; cost is never invented."""
        provider = gemini(model="some-unknown-free-model")
        with patch.object(
            provider._client.models, "generate_content", return_value=make_genai_response("ok")
        ):
            result = await provider.generate("prompt", run_id="r")

        assert result.usage.cost_usd == 0.0

    async def test_empty_response_is_transient(self) -> None:
        """A blocked or truncated candidate must not be silently returned as ''."""
        provider = gemini(max_attempts=1)
        blank = make_genai_response("")
        blank.candidates[0].content.parts = []

        with (
            patch.object(provider._client.models, "generate_content", return_value=blank),
            pytest.raises(TransientLLMError),
        ):
            await provider.generate("prompt")


# ===========================================================================
# Structured output
# ===========================================================================


class TestStructuredOutput:
    async def test_valid_json_is_parsed(self) -> None:
        provider = gemini()
        payload = json.dumps({"score": 8, "reason": "solid evidence"})

        with patch.object(
            provider._client.models, "generate_content", return_value=make_genai_response(payload)
        ):
            result = await provider.generate_structured("Rate this", Answer, run_id="r")

        assert isinstance(result.parsed, Answer)
        assert result.parsed.score == 8
        assert result.parsed.reason == "solid evidence"

    async def test_markdown_fenced_json_is_recovered(self) -> None:
        """Small free-tier models routinely wrap JSON in fences."""
        provider = gemini()
        fenced = '```json\n{"score": 5, "reason": "ok"}\n```'

        with patch.object(
            provider._client.models, "generate_content", return_value=make_genai_response(fenced)
        ):
            result = await provider.generate_structured("Rate", Answer)

        assert result.parsed is not None
        assert result.parsed.score == 5

    async def test_retries_with_validation_feedback(self) -> None:
        """PRD s6.3: retry with the validation error, up to structured_retries."""
        provider = gemini(structured_retries=2)
        prompts: list[str] = []
        responses = [
            make_genai_response("not json at all {{"),
            make_genai_response(json.dumps({"score": 3, "reason": "fixed"})),
        ]

        def side_effect(*_: Any, **kwargs: Any) -> Any:
            prompts.append(kwargs["contents"][0])
            return responses[len(prompts) - 1]

        with patch.object(
            provider._client.models, "generate_content", side_effect=side_effect
        ):
            result = await provider.generate_structured("Rate this", Answer)

        assert result.parsed is not None
        assert result.parsed.score == 3
        assert len(prompts) == 2
        # The repair prompt carries the schema and quotes the rejected output.
        assert "RETRY" in prompts[1]
        assert "Answer" in prompts[1]
        assert "not json at all" in prompts[1]

    async def test_every_attempt_records_usage(self) -> None:
        """Each repair attempt is a real billed call, so each gets its own row."""
        recorded: list[LLMUsage] = []

        async def sink(usage: LLMUsage) -> None:
            recorded.append(usage)

        provider = GeminiProvider(
            LLMConfig(api_key="k", base_delay=0.0, structured_retries=2),
            usage_sink=sink,
        )
        responses = [
            make_genai_response("garbage"),
            make_genai_response(json.dumps({"score": 1, "reason": "r"})),
        ]

        with patch.object(
            provider._client.models, "generate_content", side_effect=responses
        ):
            result = await provider.generate_structured("Rate", Answer, run_id="r")

        assert len(recorded) == 2
        # The returned usage aggregates both attempts.
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 40

    async def test_raises_after_exhausting_retries(self) -> None:
        provider = gemini(structured_retries=2)

        with patch.object(
            provider._client.models,
            "generate_content",
            side_effect=[make_genai_response("nope") for _ in range(3)],
        ) as mock_call, pytest.raises(StructuredOutputError) as exc_info:
            await provider.generate_structured("Rate", Answer)

        assert mock_call.call_count == 3  # 1 initial + 2 repairs
        assert exc_info.value.attempts == 3

    async def test_uses_sdk_parsed_object_as_fallback(self) -> None:
        provider = gemini()
        response = make_genai_response("<<unparseable>>")
        response.parsed = {"score": 9, "reason": "from sdk"}

        with patch.object(provider._client.models, "generate_content", return_value=response):
            result = await provider.generate_structured("Rate", Answer)

        assert result.parsed is not None
        assert result.parsed.reason == "from sdk"


# ===========================================================================
# Transport retries
# ===========================================================================


class TestRetries:
    async def test_retries_on_rate_limit_then_succeeds(self) -> None:
        provider = gemini()
        calls = 0

        def side_effect(*_: Any, **__: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise Exception("429 quota exceeded - rate limit")
            return make_genai_response("Retry worked")

        with patch.object(
            provider._client.models, "generate_content", side_effect=side_effect
        ):
            result = await provider.generate("test")

        assert result.text == "Retry worked"
        assert calls == 2

    async def test_does_not_retry_auth_errors(self) -> None:
        provider = gemini()
        calls = 0

        def side_effect(*_: Any, **__: Any) -> Any:
            nonlocal calls
            calls += 1
            raise Exception("invalid API key - authentication failed")

        with patch.object(
            provider._client.models, "generate_content", side_effect=side_effect
        ), pytest.raises(Exception, match="invalid API key"):
            await provider.generate("test")

        assert calls == 1

    async def test_transient_match_is_word_bounded(self) -> None:
        """"accurate" contains "rate" but is not a rate-limit error."""
        provider = gemini()
        calls = 0

        def side_effect(*_: Any, **__: Any) -> Any:
            nonlocal calls
            calls += 1
            raise ValueError("the response was not accurate for request 1500")

        with patch.object(
            provider._client.models, "generate_content", side_effect=side_effect
        ), pytest.raises(ValueError):
            await provider.generate("test")

        assert calls == 1

    async def test_retries_on_http_status_code(self) -> None:
        """Transience is read from the status code, not just the message."""
        provider = gemini()
        calls = 0
        request = httpx.Request("POST", "https://example.invalid")

        def side_effect(*_: Any, **__: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise httpx.HTTPStatusError(
                    "server blew up",
                    request=request,
                    response=httpx.Response(503, request=request),
                )
            return make_genai_response("recovered")

        with patch.object(
            provider._client.models, "generate_content", side_effect=side_effect
        ):
            result = await provider.generate("test")

        assert result.text == "recovered"
        assert calls == 2


# ===========================================================================
# OpenAI-compatible provider (second provider — PRD s4)
# ===========================================================================


class TestOpenAICompatibleProvider:
    def build(self, handler: Any) -> OpenAICompatibleProvider:
        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        config = LLMConfig(
            provider="openai_compatible",
            model="meta-llama/llama-3.3-70b-instruct:free",
            base_url="https://openrouter.example/api/v1",
            api_key="or-key",
            base_delay=0.0,
        )
        return OpenAICompatibleProvider(config, client=client)

    async def test_generate_parses_openai_shape(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                    },
                },
            )

        provider = self.build(handler)
        result = await provider.generate("hi", agent="synthesizer", run_id="r2")

        assert result.text == "hello"
        assert result.usage.prompt_tokens == 7
        assert result.usage.completion_tokens == 3
        assert result.usage.provider == "openai_compatible"
        assert seen["url"] == "https://openrouter.example/api/v1/chat/completions"
        assert seen["auth"] == "Bearer or-key"
        assert seen["body"]["model"] == "meta-llama/llama-3.3-70b-instruct:free"

    async def test_structured_request_asks_for_json_object(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"score":4,"reason":"y"}'}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        provider = self.build(handler)
        result = await provider.generate_structured("Rate", Answer)

        assert result.parsed is not None
        assert result.parsed.score == 4
        assert seen["body"]["response_format"] == {"type": "json_object"}
        # The schema is injected via the system message for endpoints that
        # don't implement strict json_schema mode.
        assert "Answer" in seen["body"]["messages"][0]["content"]

    async def test_retries_on_429(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls < 2:
                return httpx.Response(429, json={"error": "slow down"})
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        provider = self.build(handler)
        result = await provider.generate("hi")

        assert result.text == "ok"
        assert calls == 2

    async def test_empty_content_is_transient(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
            )

        transport = httpx.MockTransport(handler)
        config = LLMConfig(
            provider="openai_compatible",
            base_url="https://x.example/v1",
            max_attempts=1,
            base_delay=0.0,
        )
        provider = OpenAICompatibleProvider(
            config, client=httpx.AsyncClient(transport=transport)
        )

        with pytest.raises(TransientLLMError):
            await provider.generate("hi")


# ===========================================================================
# Factory
# ===========================================================================


class TestFactory:
    def test_builds_gemini(self) -> None:
        provider = build_llm_provider(LLMConfig(provider="gemini", api_key="k"))
        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"

    def test_builds_openai_compatible(self) -> None:
        provider = build_llm_provider(
            LLMConfig(provider="openrouter", base_url="https://x.example/v1", api_key="k")
        )
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_unknown_provider_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            build_llm_provider(LLMConfig(provider="not-a-provider"))

    def test_at_least_two_providers_available(self) -> None:
        """PRD s4 requires support for at least two providers."""
        names = available_providers()
        assert "gemini" in names
        assert "openai_compatible" in names

    def test_legacy_alias_still_resolves(self) -> None:
        assert GeminiClient is GeminiProvider


# ===========================================================================
# Pricing and JSON salvage helpers
# ===========================================================================


class TestPricing:
    def test_explicit_price_wins_over_table(self) -> None:
        pricing = resolve_pricing("gemini-2.5-flash", input_per_1m=5.0, output_per_1m=6.0)
        assert pricing.input_per_1m == 5.0
        assert pricing.output_per_1m == 6.0

    def test_longest_prefix_wins(self) -> None:
        lite = resolve_pricing("gemini-2.5-flash-lite-001")
        flash = resolve_pricing("gemini-2.5-flash-001")
        assert lite != flash

    def test_unknown_model_is_free(self) -> None:
        assert resolve_pricing("mystery-model").is_free

    def test_compute_cost(self) -> None:
        pricing = resolve_pricing("x", input_per_1m=10.0, output_per_1m=30.0)
        cost = compute_cost_usd(1_000_000, 1_000_000, pricing)
        assert cost == pytest.approx(40.0)

    def test_free_pricing_short_circuits(self) -> None:
        assert compute_cost_usd(999_999, 999_999, resolve_pricing("free-model")) == 0.0


class TestExtractJson:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"a":1}', '{"a":1}'),
            ('```json\n{"a":1}\n```', '{"a":1}'),
            ('```\n{"a":1}\n```', '{"a":1}'),
            ('Here you go: {"a":1} hope that helps', '{"a":1}'),
            ("[1,2,3]", "[1,2,3]"),
            ("", ""),
        ],
    )
    def test_recovers_payload(self, raw: str, expected: str) -> None:
        assert extract_json(raw) == expected


class TestLLMUsage:
    def test_from_metadata_handles_none(self) -> None:
        usage = LLMUsage.from_metadata(None, model="m", latency_ms=5.0)
        assert usage.total_tokens == 0
        assert usage.model == "m"

    def test_from_metadata_derives_total(self) -> None:
        metadata = MagicMock()
        metadata.prompt_token_count = 4
        metadata.candidates_token_count = 6
        metadata.total_token_count = 0
        usage = LLMUsage.from_metadata(metadata, model="m", latency_ms=1.0)
        assert usage.total_tokens == 10

    def test_as_usage_row_maps_column_names(self) -> None:
        usage = LLMUsage(
            prompt_tokens=3,
            completion_tokens=4,
            model="m",
            agent="planner",
            run_id="r",
            latency_ms=12.7,
            cost_usd=0.5,
        )
        row = usage.as_usage_row()
        assert row == {
            "run_id": "r",
            "agent": "planner",
            "model": "m",
            "tokens_in": 3,
            "tokens_out": 4,
            "cost_usd": 0.5,
            "latency_ms": 12,
        }


class TestAbstractContract:
    async def test_sink_is_optional(self) -> None:
        """A provider built without a sink still works (scripts, tests)."""
        provider = gemini()
        with patch.object(
            provider._client.models, "generate_content", return_value=make_genai_response("x")
        ):
            result = await provider.generate("p")
        assert result.text == "x"

    def test_cannot_instantiate_base_directly(self) -> None:
        from researchmind_agents.llm.base import BaseLLMProvider

        with pytest.raises(TypeError):
            BaseLLMProvider(LLMConfig())  # type: ignore[abstract]

    async def test_aclose_is_safe(self) -> None:
        provider = OpenAICompatibleProvider(
            LLMConfig(provider="openai_compatible", base_url="https://x.example/v1")
        )
        await provider.aclose()

    async def test_provided_client_is_not_closed(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        provider = OpenAICompatibleProvider(
            LLMConfig(provider="openai_compatible", base_url="https://x.example/v1"),
            client=client,
        )
        await provider.aclose()
        client.aclose.assert_not_called()
