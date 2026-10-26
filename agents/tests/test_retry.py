"""
Tests for retry waits: server-suggested delays (free-tier 429s) are honoured,
capped by max_delay, and never shorten the normal back-off.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from researchmind_agents.config import AgentConfig
from researchmind_agents.llm.base import TransientLLMError, retry_after_seconds, retry_async

# Shape of a real Gemini free-tier embedding 429 (trimmed).
GEMINI_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current "
    "quota... Please retry in 46.816531493s.', 'status': 'RESOURCE_EXHAUSTED', 'details': "
    "[{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '46s'}]}}"
)
# Shape of a real Gemini overload 503: no retry hint.
GEMINI_503 = (
    "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing "
    "high demand. Spikes in demand are usually temporary. Please try again later.', "
    "'status': 'UNAVAILABLE'}}"
)


class TestRetryAfterSeconds:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (GEMINI_429, 46.816531493),
            ("{'retryDelay': '12s'}", 12.0),
            ('"retryDelay": "7s"', 7.0),
            ("Rate limited, retry in 3s", 3.0),
            ("503 overloaded", None),
            ("retry later", None),
        ],
    )
    def test_reads_delay_from_error_text(self, message: str, expected: float | None) -> None:
        assert retry_after_seconds(TransientLLMError(message)) == expected

    def test_reads_retry_after_header(self) -> None:
        request = httpx.Request("POST", "https://api.example/v1")
        response = httpx.Response(429, headers={"Retry-After": "20"}, request=request)
        exc = httpx.HTTPStatusError("429", request=request, response=response)
        assert retry_after_seconds(exc) == 20.0

    def test_ignores_http_date_retry_after(self) -> None:
        request = httpx.Request("POST", "https://api.example/v1")
        response = httpx.Response(
            429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, request=request
        )
        exc = httpx.HTTPStatusError("429", request=request, response=response)
        assert retry_after_seconds(exc) is None


class _Flaky:
    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = errors
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


async def run_with_sleeps(factory: Any, **kwargs: Any) -> tuple[Any, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with patch("researchmind_agents.llm.base.asyncio.sleep", fake_sleep):
        result = await retry_async(factory, **kwargs)
    return result, sleeps


class TestRetryAsyncWaits:
    async def test_server_hint_replaces_short_backoff(self) -> None:
        factory = _Flaky([TransientLLMError(GEMINI_429)])
        result, sleeps = await run_with_sleeps(factory, max_attempts=3, base_delay=1.0)
        assert result == "ok"
        assert sleeps == [pytest.approx(46.816531493)]

    async def test_hint_is_capped_by_max_delay(self) -> None:
        factory = _Flaky([TransientLLMError(GEMINI_429)])
        _, sleeps = await run_with_sleeps(factory, max_attempts=2, base_delay=1.0, max_delay=30.0)
        assert sleeps == [30.0]

    async def test_hint_never_shortens_backoff(self) -> None:
        factory = _Flaky([TransientLLMError("429 retry in 0.5s"), TransientLLMError("429 retry in 0.5s")])
        _, sleeps = await run_with_sleeps(factory, max_attempts=3, base_delay=2.0)
        assert sleeps == [2.0, 4.0]

    async def test_without_hint_uses_exponential_backoff(self) -> None:
        factory = _Flaky([TransientLLMError("503 unavailable"), TransientLLMError("503 unavailable")])
        _, sleeps = await run_with_sleeps(factory, max_attempts=3, base_delay=1.0)
        assert sleeps == [1.0, 2.0]

    async def test_non_transient_error_is_not_retried(self) -> None:
        factory = _Flaky([ValueError("bad api key")])
        with pytest.raises(ValueError):
            await run_with_sleeps(factory, max_attempts=3, base_delay=1.0)
        assert factory.calls == 1


class TestOverloadWaits:
    async def test_overload_backs_off_from_overload_delay(self) -> None:
        factory = _Flaky([TransientLLMError(GEMINI_503), TransientLLMError(GEMINI_503)])
        result, sleeps = await run_with_sleeps(
            factory, max_attempts=3, base_delay=1.0, overload_delay=10.0
        )
        assert result == "ok"
        assert sleeps == [10.0, 20.0]

    async def test_overload_wait_is_capped_by_max_delay(self) -> None:
        factory = _Flaky([TransientLLMError(GEMINI_503), TransientLLMError(GEMINI_503)])
        _, sleeps = await run_with_sleeps(
            factory, max_attempts=3, base_delay=1.0, max_delay=15.0, overload_delay=10.0
        )
        assert sleeps == [10.0, 15.0]

    async def test_rate_limit_without_hint_keeps_normal_backoff(self) -> None:
        factory = _Flaky([TransientLLMError("429 too many requests")] * 2)
        _, sleeps = await run_with_sleeps(
            factory, max_attempts=3, base_delay=1.0, overload_delay=10.0
        )
        assert sleeps == [1.0, 2.0]

    async def test_server_hint_wins_over_overload_delay(self) -> None:
        factory = _Flaky([TransientLLMError("503 unavailable, retry in 3s")])
        _, sleeps = await run_with_sleeps(
            factory, max_attempts=2, base_delay=1.0, overload_delay=10.0
        )
        assert sleeps == [3.0]

    async def test_overload_is_read_from_the_status_code(self) -> None:
        request = httpx.Request("POST", "https://api.example/v1")
        response = httpx.Response(503, request=request)
        factory = _Flaky([httpx.HTTPStatusError("server blew up", request=request, response=response)])
        _, sleeps = await run_with_sleeps(
            factory, max_attempts=2, base_delay=1.0, overload_delay=10.0
        )
        assert sleeps == [10.0]


class TestConfig:
    def test_max_retry_delay_maps_to_llm_and_embeddings(self) -> None:
        config = AgentConfig.from_mapping({"llm_max_retry_delay": 15.0})
        assert config.llm.max_retry_delay == 15.0
        assert config.embedding.max_retry_delay == 15.0

    def test_overload_delay_maps_to_llm_and_embeddings(self) -> None:
        config = AgentConfig.from_mapping({"llm_overload_retry_delay": 4.0})
        assert config.llm.overload_delay == 4.0
        assert config.embedding.overload_delay == 4.0

    def test_default_overload_retries_outlast_a_short_spike(self) -> None:
        config = AgentConfig.from_mapping({})
        # 3 attempts wait 10s + 20s; a 1s/2s back-off met the same spike in live runs.
        assert config.llm.overload_delay >= 5
        # Every overload wait plus a slow generation must fit inside one node timeout.
        waits = sum(
            min(config.llm.overload_delay * 2**n, config.llm.max_retry_delay)
            for n in range(config.llm.max_attempts - 1)
        )
        assert config.agent_timeout > waits + 120

    def test_defaults_wait_out_a_free_tier_minute(self) -> None:
        config = AgentConfig.from_mapping({})
        assert config.llm.max_retry_delay >= 45
        assert config.embedding.max_retry_delay >= 45
        # A node must outlive one full rate-limit wait plus its own work.
        assert config.agent_timeout > 2 * config.embedding.max_retry_delay
