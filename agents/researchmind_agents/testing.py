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
* :class:`FakeScholar` — a tiny paper index for the academic researcher.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from researchmind_agents.config import AgentConfig, EmbeddingConfig, LLMConfig
from researchmind_agents.llm.base import BaseLLMProvider, RawCompletion, UsageSink
from researchmind_agents.llm.embeddings import EmbeddingService
from researchmind_agents.runtime import AgentRuntime, build_runtime
from researchmind_agents.storage import ResearchStore
from researchmind_agents.tools.page_extract import PageContent, PageExtractTool
from researchmind_agents.tools.scholarly import Paper, ScholarlySearchTool
from researchmind_agents.tools.search import SearchResult, SearchTool

__all__ = [
    "Responder",
    "ScriptedLLM",
    "default_plan_response",
    "default_selection_response",
    "default_report_response",
    "default_claims_response",
    "default_fact_check_response",
    "default_verified_report_response",
    "default_decision_report_response",
    "default_gap_plan_response",
    "default_contradiction_response",
    "default_sufficiency_response",
    "FakeWeb",
    "FakeSearchProvider",
    "FakePageExtract",
    "FakeScholar",
    "FakeScholarProvider",
    "fake_embedding_service",
    "fake_scholarly_tool",
    "build_fake_runtime",
]

# A canned response: literal text, a function of the prompt, or an exception to raise.
Responder = str | Callable[[str], str] | BaseException

_WORD = re.compile(r"[a-z0-9]{3,}")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def default_plan_response(prompt: str) -> str:
    """
    Two sub-questions built from the question in the prompt: one "web", one
    "technical" (quick mode maps both to web; academic mode makes the first
    academic).
    """
    question = prompt.split("Research question:\n", 1)[-1].split("\n", 1)[0].strip()
    return json.dumps(
        {
            "objective": f"Answer: {question}",
            "sub_questions": [
                {
                    "text": f"What are the benefits of {question}",
                    "queries": [question, f"{question} benefits"],
                    "domain": "web",
                },
                {
                    "text": f"What are the costs and risks of {question}",
                    "queries": [f"{question} costs risks"],
                    "domain": "technical",
                },
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


_PASSAGE_HEADER = re.compile(r"^\[(P\d+)\] ", re.MULTILINE)
_SENTENCE = re.compile(r"[^.!?]+[.!?]")


def default_claims_response(prompt: str) -> str:
    """One claim per passage: its first sentence, citing that passage's label."""
    match = re.search(r"List up to (\d+) claims", prompt)
    limit = int(match.group(1)) if match else 8
    claims = []
    parts = _PASSAGE_HEADER.split(prompt)
    # parts: [preamble, label1, body1, label2, body2, ...]
    for label, body in zip(parts[1::2], parts[2::2], strict=False):
        lines = body.split("\n")
        text = " ".join(line for line in lines[2:] if line and not line.startswith("UNTRUSTED"))
        sentence = _SENTENCE.search(text)
        claim = (sentence.group(0) if sentence else text[:160]).strip()
        if claim:
            claims.append({"text": claim, "passages": [label]})
    return json.dumps({"claims": claims[:limit]})


def default_fact_check_response(prompt: str) -> str:
    """Every claim supported by its first excerpt; other excerpts neutral."""
    checks: list[dict[str, Any]] = []
    for line in prompt.splitlines():
        claim = re.match(r"^\[(C\d+)\] ", line)
        evidence = re.match(r"^\s+\[(E\d+)\] ", line)
        if claim:
            checks.append(
                {"claim": claim.group(1), "verdict": "supported", "evidence": [], "reason": "Scripted."}
            )
        elif evidence and checks:
            stance = "neutral" if checks[-1]["evidence"] else "supports"
            checks[-1]["evidence"].append({"evidence": evidence.group(1), "stance": stance})
    return json.dumps({"checks": checks})


def default_verified_report_response(prompt: str) -> str:
    """A report whose statements cite the claim labels offered in the prompt."""
    match = re.search(r"claims \(labels: ([^)]*)\)", prompt)
    labels = [label.strip() for label in match.group(1).split(",")] if match else []
    first = labels[:1] or ["C1"]
    return json.dumps(
        {
            "title": "Verified test report",
            "executive_summary": [{"text": "The claims answer the question.", "claims": first}],
            "sections": [
                {
                    "heading": f"Finding {label}",
                    "statements": [{"text": f"Claim {label} holds.", "claims": [label]}],
                }
                for label in labels
            ],
            "limitations": ["This is a test report."],
        }
    )


def default_decision_report_response(prompt: str) -> str:
    """The verified test report plus a recommendation citing the first claim."""
    report = json.loads(default_verified_report_response(prompt))
    first = report["executive_summary"][0]["claims"]
    report["recommendation"] = [{"text": "Choose the option the claims favour.", "claims": first}]
    return json.dumps(report)


def default_gap_plan_response(prompt: str) -> str:
    """No new sub-questions: a gap round adds nothing unless a test scripts it."""
    return json.dumps({"sub_questions": []})


def default_contradiction_response(prompt: str) -> str:
    """No contradictions unless a test scripts them."""
    return json.dumps({"contradictions": []})


def default_sufficiency_response(prompt: str) -> str:
    return json.dumps({"sufficient": True, "gaps": [], "reason": "Scripted."})


_DEFAULT_RESPONDERS: dict[str, Responder] = {
    "PlanOutput": default_plan_response,
    "SourceSelection": default_selection_response,
    "ReportOutput": default_report_response,
    "ClaimExtractionOutput": default_claims_response,
    "FactCheckOutput": default_fact_check_response,
    "VerifiedReportOutput": default_verified_report_response,
    "DecisionReportOutput": default_decision_report_response,
    "GapPlanOutput": default_gap_plan_response,
    "ContradictionOutput": default_contradiction_response,
    "SufficiencyOutput": default_sufficiency_response,
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
        super().__init__(
            config or LLMConfig(model="scripted", base_delay=0.0, overload_delay=0.0), usage_sink
        )
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
    ``published`` maps URL -> ISO date, returned as the page's date.

    Search ranks pages by how many query words their title and text contain.
    """

    pages: dict[str, tuple[str, str]] = field(default_factory=dict)
    broken: set[str] = field(default_factory=set)
    published: dict[str, str] = field(default_factory=dict)
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
            url=url,
            final_url=url,
            title=title,
            text=text,
            text_length=len(text),
            status_code=200,
            published_at=self.published.get(url, ""),
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


@dataclass
class FakeScholar:
    """
    A paper index. Search ranks papers by how many query words their title and
    abstract contain. Set ``fail`` to make every search raise.
    """

    papers: list[Paper] = field(default_factory=list)
    fail: bool = False
    queries: list[str] = field(default_factory=list)

    def search(self, query: str, max_results: int) -> list[Paper]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("scholarly index unavailable")
        words = set(_WORD.findall(query.lower()))
        scored = []
        for index, paper in enumerate(self.papers):
            haystack = f"{paper.title} {paper.abstract}".lower()
            score = sum(1 for word in words if word in haystack)
            if score:
                scored.append((-score, index, paper))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [replace(paper, rank=rank) for rank, (_, _, paper) in enumerate(scored[:max_results])]


class FakeScholarProvider:
    def __init__(self, scholar: FakeScholar) -> None:
        self._scholar = scholar

    @property
    def name(self) -> str:
        return "fake-scholar"

    async def search(self, query: str, *, max_results: int) -> list[Paper]:
        return self._scholar.search(query, max_results)


def fake_scholarly_tool(
    config: AgentConfig | None = None, scholar: FakeScholar | None = None
) -> ScholarlySearchTool:
    """The real :class:`ScholarlySearchTool` over a :class:`FakeScholar`, with no waits."""
    academic = replace(
        (config or AgentConfig()).academic,
        arxiv_min_interval=0.0,
        semantic_scholar_min_interval=0.0,
        cache_ttl_seconds=0,
    )
    return ScholarlySearchTool(academic, providers=[FakeScholarProvider(scholar or FakeScholar())])


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
        config or EmbeddingConfig(base_delay=0.0, overload_delay=0.0),
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
    scholar: FakeScholar | None = None,
    now: Callable[[], datetime] | None = None,
    clock: Callable[[], float] | None = None,
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
            replace(llm_config, base_delay=0.0, overload_delay=0.0),
            sink,
            responses=responses,
            tokens=tokens,
        ),
        embeddings=fake_embedding_service(config.embedding),
        search=SearchTool(config.search, provider=FakeSearchProvider(web)),
        page_extract=FakePageExtract(web),
        scholarly=fake_scholarly_tool(config, scholar),
        now=now,
        clock=clock,
    )
