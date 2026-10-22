"""
SSRF-protected web page extraction tool.

Security measures:
- Blocks private/link-local IP ranges (SSRF prevention).
- Enforces a hard size cap on downloaded content.
- Respects configurable timeout.
- Strips scripts, styles, and HTML boilerplate; returns clean text.
- Validates URL scheme (http/https only).

No LLM calls, no DB access.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSRF block-list (private / link-local / loopback)
# ---------------------------------------------------------------------------

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # loopback
    ipaddress.ip_network("10.0.0.0/8"),         # private class A
    ipaddress.ip_network("172.16.0.0/12"),      # private class B
    ipaddress.ip_network("192.168.0.0/16"),     # private class C
    ipaddress.ip_network("169.254.0.0/16"),     # link-local
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),           # IPv6 private
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
    ipaddress.ip_network("0.0.0.0/8"),          # "this" network
    ipaddress.ip_network("100.64.0.0/10"),      # carrier-grade NAT
    ipaddress.ip_network("198.18.0.0/15"),      # benchmarking
    ipaddress.ip_network("198.51.100.0/24"),    # documentation
    ipaddress.ip_network("203.0.113.0/24"),     # documentation
    ipaddress.ip_network("224.0.0.0/4"),        # multicast
    ipaddress.ip_network("240.0.0.0/4"),        # reserved
]

_ALLOWED_SCHEMES = {"http", "https"}

# Tags whose entire subtree we drop during cleaning
_DROP_TAGS = {
    "script", "style", "noscript", "iframe", "object", "embed",
    "form", "input", "button", "select", "textarea", "nav",
    "footer", "header", "aside", "svg", "canvas",
}

_WHITESPACE_RE = re.compile(r"\s{3,}")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PageContent:
    """Cleaned text extracted from a web page."""

    url: str
    title: str
    text: str
    html_length: int = 0        # raw HTML bytes before cleaning
    text_length: int = 0        # cleaned text character count
    status_code: int = 200
    content_type: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text)


class SSRFError(ValueError):
    """Raised when a URL resolves to a blocked (private) IP address."""


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class PageExtractTool:
    """
    Download and clean a web page into plain text.

    Usage::

        tool = PageExtractTool(timeout=15, max_size_bytes=1_048_576)
        page = await tool.extract("https://example.com/article")
        print(page.text)
    """

    def __init__(
        self,
        timeout: int = 15,
        max_size_bytes: int = 1_048_576,  # 1 MB
        user_agent: str = (
            "ResearchMind/1.0 (+https://github.com/researchmind; "
            "research bot; respectful crawler)"
        ),
    ) -> None:
        self._timeout = timeout
        self._max_size_bytes = max_size_bytes
        self._user_agent = user_agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, url: str) -> PageContent:
        """
        Download *url* and return cleaned text.

        Returns a PageContent with ``error`` set (and ``ok=False``) on any
        failure — never raises, so the research pipeline can continue.
        """
        try:
            self._validate_url(url)
            await self._check_ssrf(url)
            return await self._fetch_and_clean(url)
        except SSRFError as exc:
            logger.warning("SSRF block for url=%r: %s", url, exc)
            return PageContent(url=url, title="", text="", error=f"SSRF blocked: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Page extraction failed url=%r: %s", url, exc)
            return PageContent(url=url, title="", text="", error=str(exc))

    async def extract_many(self, urls: list[str]) -> list[PageContent]:
        """Extract multiple URLs concurrently."""
        import asyncio

        tasks = [self.extract(u) for u in urls]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            raise ValueError(f"Disallowed URL scheme: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError("URL has no host")

    async def _check_ssrf(self, url: str) -> None:
        """Resolve the hostname and block if it maps to a private range."""
        import asyncio

        hostname = urlparse(url).hostname or ""
        try:
            # getaddrinfo returns [(family, type, proto, canonname, sockaddr), ...]
            infos = await asyncio.to_thread(
                socket.getaddrinfo, hostname, None, socket.AF_UNSPEC
            )
        except socket.gaierror as exc:
            raise ValueError(f"DNS resolution failed for {hostname!r}: {exc}") from exc

        for info in infos:
            sockaddr = info[4]
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            for net in _BLOCKED_NETWORKS:
                if ip in net:
                    raise SSRFError(
                        f"{hostname!r} resolves to private IP {ip_str} ({net})"
                    )

    async def _fetch_and_clean(self, url: str) -> PageContent:
        """Download the page and return cleaned text content."""
        import asyncio

        headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=self._timeout,
            headers=headers,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            raw_bytes = response.content[: self._max_size_bytes]
            html = raw_bytes.decode("utf-8", errors="replace")

        cleaned = await asyncio.to_thread(self._clean_html, html)

        return PageContent(
            url=url,
            title=cleaned["title"],
            text=cleaned["text"],
            html_length=len(html),
            text_length=len(cleaned["text"]),
            status_code=response.status_code,
            content_type=content_type,
        )

    @staticmethod
    def _clean_html(html: str) -> dict:
        """Parse HTML with BeautifulSoup and return {title, text}."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract title before stripping tags
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remove noise subtrees
        for tag in soup.find_all(_DROP_TAGS):
            tag.decompose()

        # Prefer main content areas
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"content|main|article", re.I))
            or soup.find(role="main")
            or soup.body
            or soup
        )

        raw_text = main.get_text(separator="\n")  # type: ignore[union-attr]

        # Collapse excessive whitespace
        lines = [ln.strip() for ln in raw_text.splitlines()]
        lines = [ln for ln in lines if ln]
        text = _WHITESPACE_RE.sub("  ", "\n".join(lines))

        return {"title": title, "text": text}
