"""M4 tests — the sliding-window rate limiter, client keys, and tracing setup."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.core.rate_limit import SlidingWindowRateLimiter, client_key
from app.core.tracing import configure_tracing


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestSlidingWindow:
    def test_allows_up_to_the_limit_then_reports_the_wait(self) -> None:
        clock = Clock()
        limiter = SlidingWindowRateLimiter(3, 60, clock=clock)

        assert [limiter.hit("k") for _ in range(3)] == [None, None, None]
        clock.now += 10
        assert limiter.hit("k") == 50  # the first hit leaves the window in 50s

    def test_the_window_slides(self) -> None:
        clock = Clock()
        limiter = SlidingWindowRateLimiter(2, 60, clock=clock)
        limiter.hit("k")
        clock.now += 30
        limiter.hit("k")
        assert limiter.hit("k") is not None

        clock.now += 30.5  # the first hit has expired, the second has not
        assert limiter.hit("k") is None
        assert limiter.hit("k") is not None

    def test_rejected_requests_are_not_counted(self) -> None:
        clock = Clock()
        limiter = SlidingWindowRateLimiter(1, 10, clock=clock)
        limiter.hit("k")
        for _ in range(5):
            assert limiter.hit("k") is not None
        clock.now += 10.1
        assert limiter.hit("k") is None

    def test_keys_are_independent(self) -> None:
        limiter = SlidingWindowRateLimiter(1, 60, clock=Clock())
        assert limiter.hit("a") is None
        assert limiter.hit("b") is None
        assert limiter.hit("a") is not None

    def test_zero_disables(self) -> None:
        limiter = SlidingWindowRateLimiter(0, 60)
        assert not limiter.enabled
        assert all(limiter.hit("k") is None for _ in range(1000))

    def test_tracked_keys_are_bounded(self) -> None:
        limiter = SlidingWindowRateLimiter(1, 60, max_keys=2, clock=Clock())
        limiter.hit("a")
        limiter.hit("b")
        limiter.hit("c")  # evicts "a", the least recently seen
        assert limiter.hit("a") is None
        assert limiter.hit("c") is not None


class TestClientKey:
    def request(self, host: str | None) -> Any:
        return SimpleNamespace(client=SimpleNamespace(host=host) if host else None)

    def test_api_key_is_fingerprinted_not_stored(self) -> None:
        key = client_key(self.request("1.2.3.4"), "secret-key")
        assert key.startswith("key:")
        assert "secret" not in key
        assert key == client_key(self.request("5.6.7.8"), "secret-key")

    def test_falls_back_to_the_client_ip(self) -> None:
        assert client_key(self.request("1.2.3.4"), "") == "ip:1.2.3.4"
        assert client_key(self.request(None), "") == "ip:unknown"


class TestTracing:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # setenv first so monkeypatch records the original state and removes
        # whatever configure_tracing writes; a bare delenv of an unset name
        # records nothing, and tracing would leak into later tests.
        for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)

    def settings(self, **values: Any) -> Settings:
        return Settings(_env_file=None, **values)  # type: ignore[call-arg]

    def test_off_by_default(self) -> None:
        assert configure_tracing(self.settings()) is False
        assert "LANGSMITH_TRACING" not in os.environ

    def test_exports_settings_to_the_environment(self) -> None:
        enabled = configure_tracing(
            self.settings(
                langsmith_tracing=True,
                langsmith_api_key="ls-key",
                langsmith_project="rm-test",
                langsmith_endpoint="https://ls.example",
            )
        )
        assert enabled
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "ls-key"
        assert os.environ["LANGSMITH_PROJECT"] == "rm-test"
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://ls.example"

    def test_real_environment_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_PROJECT", "from-env")
        configure_tracing(self.settings(langsmith_tracing=True, langsmith_api_key="k"))
        assert os.environ["LANGSMITH_PROJECT"] == "from-env"

    def test_without_a_key_tracing_stays_off(self) -> None:
        assert configure_tracing(self.settings(langsmith_tracing=True)) is False
        assert "LANGSMITH_TRACING" not in os.environ
