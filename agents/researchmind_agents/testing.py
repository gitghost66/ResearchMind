"""
Deterministic test doubles for running the graph with no network or API keys.

Shipped in the package (not under ``tests/``) so the backend's test suite can
drive the same graph through its own seam (AGENTS.md rule 7). Nothing here is
used at runtime.

* :class:`ScriptedLLM` — a real :class:`BaseLLMProvider` subclass, so retries,
  validation, usage recording, and the budget all run for real; only the
  network round-trip is replaced.
* :class:`FakeWeb` — a tiny corpus that serves both search
  (:class:`FakeSearchProvider`) and page fetches (:class:`FakePageExtract`).
* :func:`fake_embedding_service` — the real :class:`EmbeddingService` over a
  fake client producing hashed bag-of-words vectors, so similarity is
  meaningful.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from researchmind_agents.config import AgentConfig, EmbeddingConfig, LLMConfig
from researchmind_agents.llm.base import BaseLLMProvider, RawCompletion, UsageSink
from researchmind_agents.llm.embeddings import EmbeddingService
from researchmind_agents.runtime import AgentRuntime, build_runtime
from researchmind_agents.storage import ResearchStore
from researchmind_agents.tools.page_extract import PageContent, PageExtractTool
from researchmind_agents.tools.search import SearchResult, SearchTool

__all__ = [
    "Responder",
    "ScriptedLLM",
    "default_plan_response",
    "default_selection_response",
    "default_report_response",
    "FakeWeb",
    "FakeSearchProvider",
    "FakePageExtract",
    "fake_embedding_service",
    "build_fake_runtime",
]

# A canned response: literal text, a function of the prompt, or an exception to raise.
Responder = str | Callable[[str], str] | BaseException

_WORD = re.compile(r"[a-z0-9]{3,}")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def default_plan_response(prompt: str) -> str:
    """Two sub-questions built from the question in the prompt."""
    question = prompt.split("Research question:\n", 1)[-1].split("\n", 1)[0].strip()
    return json.dumps(
        {
            "objective": f"Answer: {question}",
            "sub_questions": [
                {"text": f"What are the benefits of {question}", "queries": [question, f"{question} benefits"]},
                {"text": f"What are the costs and risks of {question}", "queries": [f"{question} costs risks"]},
            ],
        }
    )


def default_selection_response(prompt: str) -> str:
    """Pick the first N labels, where N is the prompt's "Choose up to N"."""
    match = re.search(r"Choose up to (\d+)", prompt)
    limit = int(match.group(1)) if match else 3
    labels = re.findall(r"^\[(R\d+)\]", prompt, flags=re.MULTILINE)
    return json.dumps({"selected": labels[:limit]})


def default_report_response(prompt: str) -> str:
    """A report whose statements cite the labels offered in the prompt."""
    match = re.search(r"Sources \(labels: ([^)]*)\)", prompt)
    labels = [label.strip() for label in match.group(1).split(",")] if match else []
    first = labels[:1] or ["S1"]
    return json.dumps(
        {
            "title": "Test report",
            "executive_summary": [{"text": "The sources answer the question.", "citations": first}],
            "sections": [
                {
                    "heading": f"Evidence from {label}",
                    "statements": [{"text": f"Source {label} reports a finding.", "citations": [label]}],
                }
                for label in labels
            ],
            "limitations": ["This is a test report."],
        }
    )


_DEFAULT_RESPONDERS: dict[str, Responder] = {
    "PlanOutput": default_plan_response,
    "SourceSelection": default_selection_response,
    "ReportOutput": default_report_response,
}


class ScriptedLLM(BaseLLMProvider):
    """
    Replays canned responses keyed by the requested schema's class name
    (``"text"`` for free-form calls).

    A list of responders is consumed in order and its last item repeats. Every
    call reports ``tokens`` usage, so budget behaviour can be tested.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        usage_sink: UsageSink | None = None,
        *,
        responses: dict[str, Responder | list[Responder]] | None = None,
        tokens: tuple[int, int] = (100, 50),
    ) -> None:
        super().__init__(config or LLMConfig(model="scripted", base_delay=0.0), usage_sink)
        self._responses: dict[str, list[Responder]] = {
            key: [value] for key, value in _DEFAULT_RESPONDERS.items()
        }
        for key, value in (responses or {}).items():
            self._responses[key] = list(value) if isinstance(value, list) else [value]
        self._tokens = tokens
        self.calls: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    def calls_for(self, schema: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["schema"] == schema]

    async def _complete(
        self,
        *,
        prompt: str,
        system: str | None,
        temperature: float,
        max_output_tokens: int,
        json_schema: type[BaseModel] | None,
    ) -> RawCompletion:
        key = json_schema.__name__ if json_schema is not None else "text"
        self.calls.append({"schema": key, "prompt": prompt, "system": system})
        queue = self._responses.get(key)
        if not queue:
            raise AssertionError(f"ScriptedLLM has no response for {key!r}")
        responder = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(responder, BaseException):
            raise responder
        text = responder(prompt) if callable(responder) else responder
        return RawCompletion(
            text=text, prompt_tokens=self._tokens[0], completion_tokens=self._tokens[1]
        )


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------


@dataclass
class FakeWeb:
    """
    ``pages`` maps URL -> (title, text). URLs in ``broken`` fail to fetch.

    Search ranks pages by how many query words their title and text contain.
    """

    pages: dict[str, tuple[str, str]] = field(default_factory=dict)
    broken: set[str] = field(default_factory=set)
    fetched: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.queries.append(query)
        words = set(_WORD.findall(query.lower()))
        scored = []
        for url, (title, text) in self.pages.items():
            haystack = f"{title} {text}".lower()
            score = sum(1 for word in words if word in haystack)
            if score:
                scored.append((-score, url, title, text))
        scored.sort()
        return [
            SearchResult(title=title, url=url, snippet=text[:160], source="fake", rank=rank)
            for rank, (_, url, title, text) in enumerate(scored[:max_results])
        ]

    def fetch(self, url: str) -> PageContent:
        self.fetched.append(url)
        if url in self.broken or url not in self.pages:
            return PageContent(url=url, error="fetch failed")
        title, text = self.pages[url]
        return PageContent(
            url=url, final_url=url, title=title, text=text, text_length=len(text), status_code=200
        )


class FakeSearchProvider:
    def __init__(self, web: FakeWeb) -> None:
        self._web = web

    @property
    def name(self) -> str:
        return "fake"

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        return self._web.search(query, max_results)


class FakePageExtract(PageExtractTool):
    """Serves pages from a :class:`FakeWeb`; never opens a connection."""

    def __init__(self, web: FakeWeb) -> None:
        super().__init__()
        self._web = web

    async def extract(self, url: str) -> PageContent:
        return self._web.fetch(url)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _hashed_vector(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for word in _WORD.findall(text.lower()):
        index = int(hashlib.sha1(word.encode()).hexdigest(), 16) % dimension
        vector[index] += 1.0
    return vector


class _FakeEmbedModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def embed_content(self, *, model: str, contents: list[str], config: Any = None) -> Any:
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        dimension = getattr(config, "output_dimensionality", None) or 768
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=_hashed_vector(text, dimension)) for text in contents],
            usage_metadata=None,
        )


def fake_embedding_service(config: EmbeddingConfig | None = None) -> EmbeddingService:
    """The real :class:`EmbeddingService` over a deterministic in-process client."""
    return EmbeddingService(
        config or EmbeddingConfig(base_delay=0.0),
        client=SimpleNamespace(models=_FakeEmbedModels()),
    )


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def build_fake_runtime(
    config: AgentConfig | None = None,
    *,
    web: FakeWeb | None = None,
    responses: dict[str, Responder | list[Responder]] | None = None,
    tokens: tuple[int, int] = (100, 50),
    store: ResearchStore | None = None,
    usage_sink: UsageSink | None = None,
) -> AgentRuntime:
    """A runtime wired entirely to fakes (in-memory store unless *store* is given)."""
    config = config or AgentConfig()
    web = web or FakeWeb()
    return build_runtime(
        config,
        store=store,
        usage_sink=usage_sink,
        # No back-off sleeps in tests.
        llm_factory=lambda llm_config, sink: ScriptedLLM(
            replace(llm_config, base_delay=0.0), sink, responses=responses, tokens=tokens
        ),
        embeddings=fake_embedding_service(config.embedding),
        search=SearchTool(config.search, provider=FakeSearchProvider(web)),
        page_extract=FakePageExtract(web),
    )
