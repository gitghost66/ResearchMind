"""
Scholarly search tool for the academic researcher.

:class:`ScholarlySearchTool` is what the academic researcher calls. It queries
every configured :class:`ScholarlyProvider` — :class:`ArxivProvider` and
:class:`SemanticScholarProvider` by default — and merges their papers, so a new
index (OpenAlex, Crossref, ...) is a new provider, not a node change.

What the tool adds on top of the providers:

* **degrade, don't crash** — each provider is retried up to ``max_attempts``
  times in total; a provider that still fails contributes no papers, and the
  tool never raises, so a dead index cannot end a research run (PRD s6.3);
* **merging** — results are interleaved by rank (round-robin) and
  de-duplicated by :attr:`Paper.key` (arXiv ID, DOI, or normalised title). The
  first copy wins, but fields it lacks (citation count, venue, DOI, date) are
  filled from later copies — arXiv knows the abstract, Semantic Scholar the
  citation count;
* **an in-process TTL cache**, like :class:`~researchmind_agents.tools.search.SearchTool`.

Each provider spaces its own requests (arXiv asks for three seconds between
calls; Semantic Scholar's keyless pool is rate-limited hard), caps the bytes
it reads, and does not follow redirects. Their hosts are fixed in config, never
taken from retrieved content, so no SSRF check is needed.

No LLM calls, no DB access.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

from researchmind_agents.config import AcademicSearchConfig

logger = logging.getLogger(__name__)

__all__ = [
    "Paper",
    "ScholarlyProvider",
    "ArxivProvider",
    "SemanticScholarProvider",
    "ScholarlySearchTool",
    "build_scholarly_providers",
    "arxiv_search_query",
    "parse_arxiv_feed",
    "parse_semantic_scholar",
    "merge_papers",
]

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

_ARXIV_ABS = re.compile(r"^https?://(?:www\.|export\.)?arxiv\.org/abs/", re.IGNORECASE)
_ARXIV_VERSION = re.compile(r"v\d+$")
_TOKEN = re.compile(r"[a-z0-9]+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Words that only add noise to an AND-ed arXiv query.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "what", "which", "who", "how", "why", "when",
        "are", "is", "was", "were", "does", "did", "can", "could", "should", "would",
        "from", "into", "about", "than", "that", "this", "these", "those", "their",
        "between", "versus", "vs", "compared", "compare", "comparison", "best",
        "current", "evidence", "effect", "effects", "main", "use", "using", "used",
    }
)
_MAX_QUERY_TERMS = 6

_S2_FIELDS = (
    "title,abstract,url,year,venue,publicationDate,citationCount,externalIds,authors,tldr"
)

# Paper fields a later duplicate may fill when the first copy lacks them.
_FILLABLE = ("abstract", "venue", "doi", "arxiv_id", "published_at", "year", "citation_count")


@dataclass
class Paper:
    """One paper as a scholarly index describes it."""

    title: str = ""
    url: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    # ISO 8601 (e.g. "2024-03-01T00:00:00+00:00"), "" if unknown.
    published_at: str = ""
    # Journal, conference, or arXiv journal_ref; "" if none.
    venue: str = ""
    citation_count: int | None = None
    doi: str = ""
    # Without its version suffix, e.g. "2401.01234".
    arxiv_id: str = ""
    provider: str = ""
    rank: int = 0

    @property
    def key(self) -> str:
        """De-duplication key: arXiv ID, else DOI, else the normalised title."""
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower()}"
        if self.doi:
            return f"doi:{self.doi.strip().lower()}"
        return "title:" + _NON_ALNUM.sub(" ", self.title.lower()).strip()


class ScholarlyProvider(Protocol):
    """A paper index. Raise on failure; :class:`ScholarlySearchTool` handles degradation."""

    @property
    def name(self) -> str: ...

    async def search(self, query: str, *, max_results: int) -> list[Paper]: ...


# ---------------------------------------------------------------------------
# Shared HTTP plumbing
# ---------------------------------------------------------------------------


class _HttpProvider:
    """Lazy client, request spacing, and a capped body read for one provider."""

    def __init__(
        self,
        config: AcademicSearchConfig | None,
        client: httpx.AsyncClient | None,
        min_interval: float,
    ) -> None:
        self._config = config or AcademicSearchConfig()
        self._client = client
        self._owns_client = client is None
        self._min_interval = max(0.0, min_interval)
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.timeout,
                headers={"User-Agent": self._config.user_agent},
                follow_redirects=False,
            )
        return self._client

    async def _get(
        self, url: str, *, params: dict[str, Any], headers: dict[str, str] | None = None
    ) -> bytes:
        """GET *url*, spaced from this provider's previous request; body capped in size."""
        client = self._ensure_client()
        async with self._lock:
            wait = self._last_request + self._min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                async with client.stream("GET", url, params=params, headers=headers) as response:
                    response.raise_for_status()
                    return await _read_capped(response, self._config.max_response_bytes)
            finally:
                self._last_request = time.monotonic()

    async def aclose(self) -> None:
        """Close the HTTP client if this provider created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


async def _read_capped(response: httpx.Response, cap: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > cap:
            raise ValueError(f"response exceeds {cap} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


def arxiv_search_query(query: str) -> str:
    """
    arXiv ``search_query`` for free text: its significant words AND-ed together.

    arXiv's query language needs field prefixes and explicit operators; plain
    text is not ranked like a web search, so stop words are dropped and at most
    six terms are kept.
    """
    terms: list[str] = []
    for token in _TOKEN.findall(query.lower()):
        if len(token) < 3 or token in _STOPWORDS or token in terms:
            continue
        terms.append(token)
        if len(terms) >= _MAX_QUERY_TERMS:
            break
    if not terms:
        return f"all:{' '.join(query.split())}"
    return " AND ".join(f"all:{term}" for term in terms)


class ArxivProvider(_HttpProvider):
    """arXiv's public Atom API (no key; at most one request every three seconds)."""

    def __init__(
        self,
        config: AcademicSearchConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved = config or AcademicSearchConfig()
        super().__init__(resolved, client, resolved.arxiv_min_interval)

    @property
    def name(self) -> str:
        return "arxiv"

    async def search(self, query: str, *, max_results: int) -> list[Paper]:
        body = await self._get(
            self._config.arxiv_url,
            params={
                "search_query": arxiv_search_query(query),
                "start": 0,
                "max_results": max(1, max_results),
                "sortBy": "relevance",
            },
        )
        return parse_arxiv_feed(body, provider=self.name)[:max_results]


def parse_arxiv_feed(body: bytes | str, *, provider: str = "arxiv") -> list[Paper]:
    """Papers from an arXiv Atom feed; entries without an ID or title are skipped."""
    # ElementTree never resolves external entities, and the bundled expat bounds
    # entity expansion ("billion laughs"). The feed also comes from a fixed,
    # configured host, not from anything a web page supplied.
    root = ET.fromstring(body)
    papers: list[Paper] = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = _text(entry.find(f"{_ATOM}id"))
        title = _squash(_text(entry.find(f"{_ATOM}title")))
        if not raw_id or not title:
            continue
        arxiv_id = _ARXIV_VERSION.sub("", _ARXIV_ABS.sub("", raw_id.strip()))
        published_at = _iso_date(_text(entry.find(f"{_ATOM}published")))
        papers.append(
            Paper(
                title=title,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                abstract=_squash(_text(entry.find(f"{_ATOM}summary"))),
                authors=[
                    _squash(_text(author.find(f"{_ATOM}name")))
                    for author in entry.findall(f"{_ATOM}author")
                    if _text(author.find(f"{_ATOM}name")).strip()
                ],
                year=_year(published_at),
                published_at=published_at,
                venue=_squash(_text(entry.find(f"{_ARXIV}journal_ref"))),
                doi=_text(entry.find(f"{_ARXIV}doi")).strip(),
                arxiv_id=arxiv_id,
                provider=provider,
                rank=len(papers),
            )
        )
    return papers


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


class SemanticScholarProvider(_HttpProvider):
    """Semantic Scholar's Graph API paper search (an API key is optional)."""

    def __init__(
        self,
        config: AcademicSearchConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved = config or AcademicSearchConfig()
        super().__init__(resolved, client, resolved.semantic_scholar_min_interval)

    @property
    def name(self) -> str:
        return "semantic_scholar"

    async def search(self, query: str, *, max_results: int) -> list[Paper]:
        headers = {}
        if self._config.semantic_scholar_api_key:
            headers["x-api-key"] = self._config.semantic_scholar_api_key
        body = await self._get(
            self._config.semantic_scholar_url,
            params={"query": " ".join(query.split()), "limit": max(1, max_results), "fields": _S2_FIELDS},
            headers=headers or None,
        )
        return parse_semantic_scholar(json.loads(body), provider=self.name)[:max_results]


def parse_semantic_scholar(payload: Any, *, provider: str = "semantic_scholar") -> list[Paper]:
    """Papers from a Graph API search response; items without a title are skipped."""
    items = payload.get("data") if isinstance(payload, dict) else None
    papers: list[Paper] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        title = _squash(str(item.get("title") or ""))
        if not title:
            continue
        external = _as_dict(item.get("externalIds"))
        arxiv_id = _ARXIV_VERSION.sub("", str(external.get("ArXiv") or "").strip())
        doi = str(external.get("DOI") or "").strip()
        tldr = _as_dict(item.get("tldr"))
        abstract = _squash(str(item.get("abstract") or tldr.get("text") or ""))

        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        elif doi:
            url = f"https://doi.org/{doi}"
        else:
            url = str(item.get("url") or "").strip()

        year = item.get("year") if isinstance(item.get("year"), int) else None
        published_at = _iso_date(str(item.get("publicationDate") or ""))
        if not published_at and year:
            published_at = _iso_date(f"{year}-01-01")
        citations = item.get("citationCount")

        papers.append(
            Paper(
                title=title,
                url=url,
                abstract=abstract,
                authors=[
                    _squash(str(author.get("name") or ""))
                    for author in item.get("authors") or []
                    if isinstance(author, dict) and author.get("name")
                ],
                year=year if year is not None else _year(published_at),
                published_at=published_at,
                venue=_squash(str(item.get("venue") or "")),
                citation_count=citations if isinstance(citations, int) else None,
                doi=doi,
                arxiv_id=arxiv_id,
                provider=provider,
                rank=len(papers),
            )
        )
    return papers


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def build_scholarly_providers(
    config: AcademicSearchConfig,
    client: httpx.AsyncClient | None = None,
) -> list[ScholarlyProvider]:
    """Providers named in ``config.providers``; an unknown name fails at startup."""
    factories: dict[
        str, Callable[[AcademicSearchConfig, httpx.AsyncClient | None], ScholarlyProvider]
    ] = {
        "arxiv": ArxivProvider,
        "semantic_scholar": SemanticScholarProvider,
    }
    providers: list[ScholarlyProvider] = []
    for name in config.providers:
        key = name.strip().lower()
        if key not in factories:
            raise ValueError(
                f"Unknown academic search provider {name!r}; available: "
                + ", ".join(sorted(factories))
            )
        providers.append(factories[key](config, client))
    return providers


class _PaperCache:
    """Insertion-ordered TTL cache; evicts the oldest entries when full."""

    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max(1, max_entries)
        self._entries: OrderedDict[tuple[Any, ...], tuple[float, list[Paper]]] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> list[Paper] | None:
        if self._ttl <= 0:
            return None
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, papers = entry
        if time.monotonic() - stored_at > self._ttl:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return list(papers)

    def set(self, key: tuple[Any, ...], papers: list[Paper]) -> None:
        if self._ttl <= 0:
            return
        self._entries[key] = (time.monotonic(), list(papers))
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


class ScholarlySearchTool:
    """
    Cached, merged search over several :class:`ScholarlyProvider` instances.

    Usage::

        tool = ScholarlySearchTool(AcademicSearchConfig())
        papers = await tool.search("retrieval augmented generation evaluation")

    Never raises: a provider that keeps failing contributes no papers.
    """

    def __init__(
        self,
        config: AcademicSearchConfig | None = None,
        providers: list[ScholarlyProvider] | None = None,
    ) -> None:
        self._config = config or AcademicSearchConfig()
        self._providers: list[ScholarlyProvider] = (
            list(providers) if providers is not None else build_scholarly_providers(self._config)
        )
        self._cache = _PaperCache(self._config.cache_ttl_seconds, self._config.cache_max_entries)

    @property
    def config(self) -> AcademicSearchConfig:
        return self._config

    @property
    def providers(self) -> list[ScholarlyProvider]:
        return list(self._providers)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    async def search(self, query: str, *, max_results: int | None = None) -> list[Paper]:
        """Papers for *query* from every provider, merged by rank and de-duplicated."""
        limit = max(1, max_results or self._config.max_results)
        key = (
            tuple(provider.name for provider in self._providers),
            query.strip().lower(),
            limit,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        per_provider = await asyncio.gather(
            *(self._search_one(provider, query, limit) for provider in self._providers)
        )
        merged = merge_papers(list(per_provider))
        if merged:
            self._cache.set(key, merged)
        return merged

    async def search_multi(self, queries: list[str]) -> dict[str, list[Paper]]:
        """Run several queries concurrently; returns query -> papers."""
        if not queries:
            return {}
        results = await asyncio.gather(*(self.search(query) for query in queries))
        return dict(zip(queries, results, strict=True))

    async def aclose(self) -> None:
        """Close the HTTP clients the providers created."""
        for provider in self._providers:
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                await closer()

    async def _search_one(self, provider: ScholarlyProvider, query: str, limit: int) -> list[Paper]:
        attempts = max(1, self._config.max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                return await provider.search(query, max_results=limit)
            except Exception as exc:  # noqa: BLE001 - degrade, don't crash the run
                first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                if attempt == attempts:
                    logger.warning(
                        "Scholarly search %s failed for query=%r after %d attempt(s): %s",
                        provider.name,
                        query,
                        attempts,
                        first_line,
                    )
                    return []
                logger.info(
                    "Scholarly search %s attempt %d failed for query=%r: %s",
                    provider.name,
                    attempt,
                    query,
                    first_line,
                )
        return []


def merge_papers(per_provider: list[list[Paper]]) -> list[Paper]:
    """
    Interleave each provider's papers by rank and drop duplicates.

    The first copy of a paper wins; empty fields on it are filled from later
    copies. Ranks are renumbered in merged order.
    """
    merged: dict[str, Paper] = {}
    for rank in range(max((len(papers) for papers in per_provider), default=0)):
        for papers in per_provider:
            if rank >= len(papers):
                continue
            paper = papers[rank]
            existing = merged.get(paper.key)
            if existing is None:
                merged[paper.key] = Paper(
                    **{f.name: getattr(paper, f.name) for f in fields(Paper)}
                )
                continue
            for name in _FILLABLE:
                if getattr(existing, name) in ("", None) and getattr(paper, name) not in ("", None):
                    setattr(existing, name, getattr(paper, name))
            if not existing.authors and paper.authors:
                existing.authors = list(paper.authors)
    result = list(merged.values())
    for index, paper in enumerate(result):
        paper.rank = index
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return element.text or ""


def _squash(text: str) -> str:
    return " ".join(text.split())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _iso_date(value: str) -> str:
    """ISO 8601 UTC string for an arXiv/S2 date, or "" if unparseable."""
    text = value.strip()
    if not text:
        return ""
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _year(published_at: str) -> int | None:
    try:
        return datetime.fromisoformat(published_at).year if published_at else None
    except ValueError:
        return None
