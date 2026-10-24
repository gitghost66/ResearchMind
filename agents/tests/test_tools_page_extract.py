"""
M2 tests — page extraction.

M2's acceptance criteria call out the SSRF guard specifically, so these tests
cover not just the direct case but the ways a guard like this usually leaks:
redirects, IP literals, IPv6, and post-download size caps.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from researchmind_agents.config import PageExtractConfig
from researchmind_agents.tools.page_extract import (
    PageContent,
    PageExtractTool,
    SSRFError,
    _blocked_network,
)

PUBLIC_IP = "93.184.216.34"

# hostname -> IP returned by the patched resolver
DNS: dict[str, str] = {
    "example.com": PUBLIC_IP,
    "public.example": PUBLIC_IP,
    "evil.example": PUBLIC_IP,
    "internal.corp": "192.168.1.100",
    "localhost": "127.0.0.1",
    "metadata.example": "169.254.169.254",
}


def fake_getaddrinfo(host: str, *_: Any, **__: Any) -> list[tuple]:
    if host not in DNS:
        raise socket.gaierror(f"unknown host {host}")
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (DNS[host], 0))]


@pytest.fixture(autouse=True)
def patched_dns() -> Any:
    with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
        yield


def build_tool(
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
    **overrides: Any,
) -> PageExtractTool:
    """A tool wired to a MockTransport, with robots off unless asked for."""
    overrides.setdefault("respect_robots", False)
    config = PageExtractConfig(timeout=5, **overrides)
    client = None
    if handler is not None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=False
        )
    return PageExtractTool(config, client=client)


def html_response(body: str, content_type: str = "text/html") -> httpx.Response:
    return httpx.Response(200, content=body.encode(), headers={"content-type": content_type})


def chunked_response(total_bytes: int, content_type: str = "text/html") -> httpx.Response:
    """A streaming response with no Content-Length, like a chunked server."""

    async def stream() -> AsyncIterator[bytes]:
        sent = 0
        chunk = b"<p>" + b"x" * 97
        while sent < total_bytes:
            yield chunk
            sent += len(chunk)

    return httpx.Response(200, content=stream(), headers={"content-type": content_type})


SIMPLE_PAGE = (
    "<html><head><title>My Page</title></head>"
    "<body><main><p>Hello ResearchMind!</p></main></body></html>"
)


# ===========================================================================
# URL validation
# ===========================================================================


class TestUrlValidation:
    def test_rejects_non_http_scheme(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            build_tool()._validate_url("ftp://evil.example/file")

    def test_rejects_file_scheme(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            build_tool()._validate_url("file:///etc/passwd")

    def test_rejects_missing_host(self) -> None:
        with pytest.raises(ValueError, match="host"):
            build_tool()._validate_url("http:///no-host")

    def test_rejects_blocked_ip_literal_without_dns(self) -> None:
        """An IP literal is judged directly — no resolver round-trip needed."""
        with pytest.raises(SSRFError):
            build_tool()._validate_url("http://127.0.0.1:8000/admin")

    def test_rejects_cloud_metadata_literal(self) -> None:
        with pytest.raises(SSRFError):
            build_tool()._validate_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_ipv6_loopback_literal(self) -> None:
        with pytest.raises(SSRFError):
            build_tool()._validate_url("http://[::1]/admin")

    def test_allows_public_ip_literal(self) -> None:
        build_tool()._validate_url(f"http://{PUBLIC_IP}/page")


# ===========================================================================
# SSRF resolution
# ===========================================================================


class TestSSRF:
    async def test_blocks_localhost(self) -> None:
        with pytest.raises(SSRFError):
            await build_tool()._check_ssrf("http://localhost/admin")

    async def test_blocks_private_range(self) -> None:
        with pytest.raises(SSRFError):
            await build_tool()._check_ssrf("http://internal.corp/secret")

    async def test_blocks_link_local_metadata(self) -> None:
        with pytest.raises(SSRFError):
            await build_tool()._check_ssrf("http://metadata.example/")

    async def test_allows_public_host(self) -> None:
        await build_tool()._check_ssrf("http://example.com")

    async def test_blocks_when_any_address_is_private(self) -> None:
        """A split-horizon host must not pass on the strength of one good record."""
        def mixed(host: str, *_: Any, **__: Any) -> list[tuple]:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0)),
            ]

        with patch("socket.getaddrinfo", side_effect=mixed), pytest.raises(SSRFError):
            await build_tool()._check_ssrf("http://mixed.example")

    async def test_dns_failure_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="DNS resolution failed"):
            await build_tool()._check_ssrf("http://nonexistent.invalid")

    async def test_extract_returns_error_envelope_not_exception(self) -> None:
        result = await build_tool().extract("http://internal.corp/secret")

        assert isinstance(result, PageContent)
        assert not result.ok
        assert "SSRF" in result.error

    def test_blocked_network_catches_unlisted_non_global(self) -> None:
        """The is_global backstop covers ranges the explicit list misses."""
        import ipaddress

        assert _blocked_network(ipaddress.ip_address("192.0.2.1")) is not None
        assert _blocked_network(ipaddress.ip_address(PUBLIC_IP)) is None


# ===========================================================================
# Redirects — the hole that a first-URL-only check leaves open
# ===========================================================================


class TestRedirects:
    async def test_redirect_to_loopback_is_blocked(self) -> None:
        """A public page must not be able to bounce the fetcher to localhost."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "evil.example":
                return httpx.Response(302, headers={"location": "http://127.0.0.1:8000/admin"})
            return html_response("<html><body>secret</body></html>")

        result = await build_tool(handler).extract("http://evil.example/start")

        assert not result.ok
        assert "SSRF" in result.error

    async def test_redirect_to_cloud_metadata_is_blocked(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "evil.example":
                return httpx.Response(
                    302,
                    headers={"location": "http://169.254.169.254/latest/meta-data/"},
                )
            return html_response("<html><body>creds</body></html>")

        result = await build_tool(handler).extract("http://evil.example/start")

        assert not result.ok
        assert "SSRF" in result.error

    async def test_redirect_to_private_hostname_is_blocked(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "evil.example":
                return httpx.Response(302, headers={"location": "http://internal.corp/secret"})
            return html_response("<html><body>secret</body></html>")

        result = await build_tool(handler).extract("http://evil.example/start")

        assert not result.ok
        assert "SSRF" in result.error

    async def test_follows_redirect_to_public_host(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "evil.example":
                return httpx.Response(301, headers={"location": "http://public.example/real"})
            return html_response(SIMPLE_PAGE)

        result = await build_tool(handler).extract("http://evil.example/start")

        assert result.ok
        assert result.title == "My Page"
        assert result.url == "http://evil.example/start"
        assert result.final_url == "http://public.example/real"

    async def test_relative_redirect_is_resolved(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/landing"})
            return html_response(SIMPLE_PAGE)

        result = await build_tool(handler).extract("http://example.com/start")

        assert result.ok
        assert result.final_url == "http://example.com/landing"

    async def test_redirect_loop_is_bounded(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "http://example.com/loop"})

        result = await build_tool(handler, max_redirects=3).extract("http://example.com/loop")

        assert not result.ok
        assert "redirects" in result.error

    async def test_redirect_without_location_errors(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302)

        result = await build_tool(handler).extract("http://example.com/x")

        assert not result.ok
        assert "Location" in result.error


# ===========================================================================
# Size cap and content type
# ===========================================================================


class TestSizeCap:
    async def test_declared_length_over_cap_is_rejected_before_download(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return html_response("x" * 5000)

        result = await build_tool(handler, max_size_bytes=1000).extract(
            "http://example.com/big"
        )

        assert not result.ok
        assert "Content-Length" in result.error

    async def test_chunked_body_is_truncated_at_cap(self) -> None:
        """A chunked response has no Content-Length, so the cap must be streamed."""

        def handler(request: httpx.Request) -> httpx.Response:
            return chunked_response(50_000)

        result = await build_tool(handler, max_size_bytes=2000).extract(
            "http://example.com/stream"
        )

        assert result.truncated
        assert result.html_length <= 2000

    async def test_body_under_cap_is_not_truncated(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return html_response(SIMPLE_PAGE)

        result = await build_tool(handler, max_size_bytes=100_000).extract(
            "http://example.com/small"
        )

        assert result.ok
        assert not result.truncated


class TestContentType:
    async def test_rejects_binary_content_type(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"}
            )

        result = await build_tool(handler).extract("http://example.com/doc.pdf")

        assert not result.ok
        assert "content type" in result.error

    async def test_accepts_html_with_charset(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return html_response(SIMPLE_PAGE, content_type="text/html; charset=utf-8")

        result = await build_tool(handler).extract("http://example.com/page")

        assert result.ok

    async def test_missing_content_type_is_allowed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=SIMPLE_PAGE.encode())

        result = await build_tool(handler).extract("http://example.com/page")

        assert result.ok


# ===========================================================================
# robots.txt
# ===========================================================================


class TestRobots:
    def handler_with_robots(self, robots_body: str) -> Callable[[httpx.Request], httpx.Response]:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(
                    200, content=robots_body.encode(), headers={"content-type": "text/plain"}
                )
            return html_response(SIMPLE_PAGE)

        return handler

    async def test_disallowed_path_is_refused(self) -> None:
        tool = build_tool(
            self.handler_with_robots("User-agent: *\nDisallow: /private"),
            respect_robots=True,
        )

        result = await tool.extract("http://example.com/private/page")

        assert not result.ok
        assert "robots" in result.error

    async def test_allowed_path_is_fetched(self) -> None:
        tool = build_tool(
            self.handler_with_robots("User-agent: *\nDisallow: /private"),
            respect_robots=True,
        )

        result = await tool.extract("http://example.com/public/page")

        assert result.ok

    async def test_missing_robots_is_permissive(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return html_response(SIMPLE_PAGE)

        result = await build_tool(handler, respect_robots=True).extract(
            "http://example.com/page"
        )

        assert result.ok

    async def test_robots_is_cached_per_origin(self) -> None:
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.url.path)
            if request.url.path == "/robots.txt":
                return httpx.Response(200, content=b"User-agent: *\nAllow: /")
            return html_response(SIMPLE_PAGE)

        tool = build_tool(handler, respect_robots=True)
        await tool.extract("http://example.com/a")
        await tool.extract("http://example.com/b")

        assert requests.count("/robots.txt") == 1

    async def test_robots_check_skipped_when_disabled(self) -> None:
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.url.path)
            return html_response(SIMPLE_PAGE)

        await build_tool(handler, respect_robots=False).extract("http://example.com/page")

        assert "/robots.txt" not in requests


# ===========================================================================
# Fetch and clean
# ===========================================================================


class TestFetchAndClean:
    async def test_happy_path_populates_page_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return html_response(SIMPLE_PAGE)

        result = await build_tool(handler).extract("http://example.com/page")

        assert result.ok
        assert result.title == "My Page"
        assert "Hello ResearchMind" in result.text
        assert result.status_code == 200
        assert result.text_length == len(result.text)

    async def test_http_error_returns_error_envelope(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        result = await build_tool(handler).extract("http://example.com/page")

        assert not result.ok
        assert result.error

    async def test_transport_error_returns_error_envelope(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        result = await build_tool(handler).extract("http://example.com/page")

        assert not result.ok
        assert "connection refused" in result.error

    async def test_clean_html_strips_scripts_and_chrome(self) -> None:
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
            <footer>Footer junk</footer>
          </body>
        </html>
        """
        result = PageExtractTool._clean_html(html)

        assert result["title"] == "Test Page"
        assert "Article Title" in result["text"]
        assert "article content" in result["text"]
        assert "alert" not in result["text"]
        assert "Navigation" not in result["text"]
        assert "Footer junk" not in result["text"]

    async def test_extract_many_preserves_order(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return html_response(
                f"<html><head><title>{request.url.path}</title></head>"
                "<body><main>body</main></body></html>"
            )

        urls = [f"http://example.com/p{i}" for i in range(5)]
        results = await build_tool(handler).extract_many(urls, concurrency=2)

        assert [r.title for r in results] == [f"/p{i}" for i in range(5)]

    async def test_extract_many_empty(self) -> None:
        assert await build_tool().extract_many([]) == []

    async def test_one_bad_url_does_not_fail_the_batch(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return html_response(SIMPLE_PAGE)

        results = await build_tool(handler).extract_many(
            ["http://example.com/ok", "http://127.0.0.1/admin"]
        )

        assert results[0].ok
        assert not results[1].ok

    async def test_aclose_is_safe_for_injected_client(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return html_response(SIMPLE_PAGE)

        tool = build_tool(handler)
        await tool.aclose()  # injected client is not owned, so this is a no-op
        result = await tool.extract("http://example.com/page")
        assert result.ok
