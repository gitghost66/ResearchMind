"""
SSRF-protected web page extraction.

Security measures (PRD s10):

* **Scheme allowlist** — http/https only.
* **SSRF blocking** — every hop's hostname is resolved and rejected if it maps
  into a private, loopback, link-local, or cloud-metadata range. Redirects are
  followed manually so each hop is checked; letting httpx follow them would
  mean only the first URL was ever validated.
* **Streaming size cap** — the body is read incrementally and abandoned once the
  cap is hit, so a huge response can never be buffered in full.
* **Content-type allowlist** — binary payloads are rejected before parsing.
* **robots.txt** — fetched, cached, and honoured (PRD s4).

Retrieved text is data, never instructions: this module only extracts and
returns it, and never lets it influence a request.

No LLM calls, no DB access.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from researchmind_agents.config import PageExtractConfig

logger = logging.getLogger(__name__)

__all__ = ["PageContent", "PageExtractTool", "SSRFError", "RobotsDisallowedError"]

# ---------------------------------------------------------------------------
# SSRF block-list
# ---------------------------------------------------------------------------

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),          # "this" network
    ipaddress.ip_network("10.0.0.0/8"),         # private class A
    ipaddress.ip_network("100.64.0.0/10"),      # carrier-grade NAT
    ipaddress.ip_network("127.0.0.0/8"),        # loopback
    ipaddress.ip_network("169.254.0.0/16"),     # link-local (cloud metadata)
    ipaddress.ip_network("172.16.0.0/12"),      # private class B
    ipaddress.ip_network("192.0.0.0/24"),       # IETF protocol assignments
    ipaddress.ip_network("192.168.0.0/16"),     # private class C
    ipaddress.ip_network("198.18.0.0/15"),      # benchmarking
    ipaddress.ip_network("198.51.100.0/24"),    # documentation
    ipaddress.ip_network("203.0.113.0/24"),     # documentation
    ipaddress.ip_network("224.0.0.0/4"),        # multicast
    ipaddress.ip_network("240.0.0.0/4"),        # reserved
    ipaddress.ip_network("255.255.255.255/32"),  # broadcast
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("::/128"),             # IPv6 unspecified
    ipaddress.ip_network("::ffff:0:0/96"),      # IPv4-mapped IPv6
    ipaddress.ip_network("fc00::/7"),           # IPv6 unique-local
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
    ipaddress.ip_network("ff00::/8"),           # IPv6 multicast
]

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# Tags whose entire subtree is dropped during cleaning.
_DROP_TAGS = {
    "script", "style", "noscript", "iframe", "object", "embed",
    "form", "input", "button", "select", "textarea", "nav",
    "footer", "header", "aside", "svg", "canvas",
}

_WHITESPACE_RE = re.compile(r"\s{3,}")


# ---------------------------------------------------------------------------
# Errors and results
# ---------------------------------------------------------------------------


class SSRFError(ValueError):
    """Raised when a URL resolves to a blocked (non-public) IP address."""


class RobotsDisallowedError(PermissionError):
    """Raised when robots.txt disallows fetching a URL."""


@dataclass
class PageContent:
    """Cleaned text extracted from a web page."""

    url: str
    title: str = ""
    text: str = ""
    final_url: str = ""
    html_length: int = 0
    text_length: int = 0
    status_code: int = 0
    content_type: str = ""
    truncated: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class PageExtractTool:
    """
    Download and clean a web page into plain text.

    Usage::

        tool = PageExtractTool(PageExtractConfig(timeout=15, max_size_bytes=1_048_576))
        page = await tool.extract("https://example.com/article")
        if page.ok:
            print(page.text)

    ``extract`` never raises: failures come back as a :class:`PageContent` with
    ``error`` set, so one bad URL cannot end a research run.
    """

    def __init__(
        self,
        config: PageExtractConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or PageExtractConfig()
        self._client = client
        self._owns_client = client is None
        # origin -> (fetched_at, parser or None when robots.txt was unavailable)
        self._robots_cache: dict[str, tuple[float, RobotFileParser | None]] = {}

    @property
    def config(self) -> PageExtractConfig:
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, url: str) -> PageContent:
        """Download *url* (following validated redirects) and return cleaned text."""
        try:
            self._validate_url(url)
            await self._check_ssrf(url)
            if self._config.respect_robots and not await self._robots_allows(url):
                raise RobotsDisallowedError(f"robots.txt disallows {url}")
            return await self._fetch_and_clean(url)
        except SSRFError as exc:
            logger.warning("SSRF block for url=%r: %s", url, exc)
            return PageContent(url=url, error=f"SSRF blocked: {exc}")
        except RobotsDisallowedError as exc:
            logger.info("robots.txt disallowed url=%r", url)
            return PageContent(url=url, error=f"robots disallowed: {exc}")
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash the run
            logger.warning("Page extraction failed url=%r: %s", url, exc)
            return PageContent(url=url, error=str(exc))

    async def extract_many(
        self,
        urls: list[str],
        *,
        concurrency: int = 4,
    ) -> list[PageContent]:
        """Extract several URLs with bounded concurrency, preserving input order."""
        if not urls:
            return []

        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run(target: str) -> PageContent:
            async with semaphore:
                return await self.extract(target)

        return list(await asyncio.gather(*(run(u) for u in urls)))

    async def aclose(self) -> None:
        """Close the HTTP client if this tool created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_url(self, url: str) -> None:
        """Check scheme and host, and reject blocked IP literals without DNS."""
        parsed = urlparse(url)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            raise ValueError(f"Disallowed URL scheme: {parsed.scheme!r}")
        if not parsed.netloc or not parsed.hostname:
            raise ValueError("URL has no host")

        # A bare IP literal needs no DNS lookup to be judged.
        literal = _as_ip(parsed.hostname)
        if literal is not None:
            blocked = _blocked_network(literal)
            if blocked is not None:
                raise SSRFError(f"{parsed.hostname} is in blocked range {blocked}")

    async def _check_ssrf(self, url: str) -> None:
        """
        Resolve the hostname and reject it if any address is non-public.

        Every address is checked, not just the first, so a host with one public
        and one private record cannot slip through.

        Note: a determined attacker controlling DNS could still rebind between
        this check and the connection. Closing that fully requires pinning the
        connection to a vetted IP; the exposure here is limited to outbound GETs
        whose bodies are only ever returned as text.
        """
        hostname = urlparse(url).hostname or ""
        if _as_ip(hostname) is not None:
            # Already judged by _validate_url; no resolution needed.
            return

        try:
            infos = await asyncio.to_thread(
                socket.getaddrinfo, hostname, None, socket.AF_UNSPEC
            )
        except socket.gaierror as exc:
            raise ValueError(f"DNS resolution failed for {hostname!r}: {exc}") from exc

        if not infos:
            raise ValueError(f"DNS returned no addresses for {hostname!r}")

        for info in infos:
            address = str(info[4][0])
            ip = _as_ip(address)
            if ip is None:
                continue
            blocked = _blocked_network(ip)
            if blocked is not None:
                raise SSRFError(
                    f"{hostname!r} resolves to {address} in blocked range {blocked}"
                )

    # ------------------------------------------------------------------
    # robots.txt
    # ------------------------------------------------------------------

    async def _robots_allows(self, url: str) -> bool:
        """
        Check robots.txt for *url*'s origin.

        An unreachable or unparseable robots.txt is treated as permissive, which
        is the conventional reading of the standard.
        """
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        cached = self._robots_cache.get(origin)
        ttl = self._config.robots_cache_ttl_seconds
        if cached is not None and (ttl <= 0 or time.monotonic() - cached[0] <= ttl):
            parser = cached[1]
            if parser is None:
                return True
            return parser.can_fetch(self._config.user_agent, url)

        parser = await self._fetch_robots(origin)
        self._robots_cache[origin] = (time.monotonic(), parser)
        if parser is None:
            return True
        return parser.can_fetch(self._config.user_agent, url)

    async def _fetch_robots(self, origin: str) -> RobotFileParser | None:
        """Fetch and parse ``<origin>/robots.txt``; ``None`` if unavailable."""
        robots_url = f"{origin}/robots.txt"
        client = self._ensure_client()
        try:
            response = await client.get(
                robots_url,
                headers={"User-Agent": self._config.user_agent},
                follow_redirects=True,
            )
        except Exception as exc:  # noqa: BLE001 - absent robots.txt means allow
            logger.debug("robots.txt fetch failed for %s: %s", origin, exc)
            return None

        if response.status_code >= 400:
            return None

        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.parse(response.text.splitlines())
        except Exception as exc:  # noqa: BLE001 - malformed robots.txt means allow
            logger.debug("robots.txt parse failed for %s: %s", origin, exc)
            return None
        return parser

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.timeout,
                follow_redirects=False,
            )
        return self._client

    async def _fetch_and_clean(self, url: str) -> PageContent:
        """Fetch *url*, following validated redirects, and clean the result."""
        client = self._ensure_client()
        headers = {
            "User-Agent": self._config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        current = url
        for _hop in range(self._config.max_redirects + 1):
            # follow_redirects stays off so each hop is re-validated below.
            async with client.stream(
                "GET", current, headers=headers, follow_redirects=False
            ) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError(
                            f"Redirect {response.status_code} without a Location header"
                        )
                    current = _normalise_redirect(current, location)
                    self._validate_url(current)
                    await self._check_ssrf(current)
                    logger.debug("Following validated redirect to %s", current)
                    continue

                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                self._check_content_type(content_type)
                self._check_declared_length(response.headers.get("content-length"))

                body, truncated = await self._read_capped(response)
                encoding = response.charset_encoding or "utf-8"

            html = body.decode(encoding, errors="replace")
            cleaned = await asyncio.to_thread(_clean_html, html)

            return PageContent(
                url=url,
                final_url=current,
                title=cleaned["title"],
                text=cleaned["text"],
                html_length=len(body),
                text_length=len(cleaned["text"]),
                status_code=response.status_code,
                content_type=content_type,
                truncated=truncated,
            )

        raise ValueError(f"Exceeded {self._config.max_redirects} redirects for {url}")

    def _check_content_type(self, content_type: str) -> None:
        """Reject payloads that are not text we can usefully parse."""
        allowed = self._config.allowed_content_types
        if not allowed:
            return
        mime = content_type.split(";", 1)[0].strip().lower()
        if not mime:
            # Servers that omit the header are given the benefit of the doubt;
            # the size cap still bounds the damage.
            return
        if mime not in allowed:
            raise ValueError(f"Disallowed content type: {mime!r}")

    def _check_declared_length(self, content_length: str | None) -> None:
        """Reject an oversized body before downloading a single chunk of it."""
        if not content_length:
            return
        try:
            declared = int(content_length)
        except ValueError:
            return
        if declared > self._config.max_size_bytes:
            raise ValueError(
                f"Content-Length {declared} exceeds cap {self._config.max_size_bytes}"
            )

    async def _read_capped(self, response: httpx.Response) -> tuple[bytes, bool]:
        """
        Read the body incrementally, stopping at the configured cap.

        Returns ``(body, truncated)``. Reading in chunks is what makes the cap
        real: a declared-length check alone is trivially defeated by a chunked
        response.
        """
        cap = self._config.max_size_bytes
        chunks: list[bytes] = []
        total = 0
        truncated = False

        async for chunk in response.aiter_bytes():
            # Strictly greater: a body landing exactly on the cap is complete,
            # not truncated.
            if total + len(chunk) > cap:
                chunks.append(chunk[: max(0, cap - total)])
                truncated = True
                break
            chunks.append(chunk)
            total += len(chunk)

        return b"".join(chunks), truncated

    # Kept as a method for convenience; the implementation is a module function
    # so it can be handed to asyncio.to_thread without binding self.
    @staticmethod
    def _clean_html(html: str) -> dict[str, str]:
        """Parse HTML and return ``{"title": ..., "text": ...}``."""
        return _clean_html(html)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse *value* as an IP address, tolerating bracketed IPv6 literals."""
    candidate = value.strip().strip("[]")
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        return None


def _blocked_network(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Network | ipaddress.IPv6Network | None:
    """Return the blocked network containing *ip*, or ``None`` if it is public."""
    for network in _BLOCKED_NETWORKS:
        if ip.version == network.version and ip in network:
            return network
    # Catch anything the explicit list misses (e.g. new reserved ranges).
    if not ip.is_global:
        return ipaddress.ip_network(f"{ip}/{ip.max_prefixlen}")
    return None


def _normalise_redirect(current: str, location: str) -> str:
    """Resolve a possibly-relative Location header and strip any fragment."""
    absolute = urljoin(current, location.strip())
    parts = urlparse(absolute)
    return urlunparse(parts._replace(fragment=""))


def _clean_html(html: str) -> dict[str, str]:
    """Strip boilerplate and return the title plus readable text."""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main|article", re.I))
        or soup.select_one('[role="main"]')
        or soup.body
        or soup
    )

    raw_text = main.get_text(separator="\n")  # type: ignore[union-attr]

    lines = [line.strip() for line in raw_text.splitlines()]
    text = _WHITESPACE_RE.sub("  ", "\n".join(line for line in lines if line))

    return {"title": title, "text": text}
