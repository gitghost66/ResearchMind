"""
The ``offline`` suite: the whole benchmark on test doubles (AGENTS.md rule 7).

Four questions, one per mode, about fictional products and studies on
``.example`` hosts, answered from a twelve-page synthetic corpus
(``testing.FakeWeb``). Every model call — the systems' and the judge's — is
scripted:

* the closed-book baseline cites one real corpus page, one made-up URL (which
  answers 404), and leaves one sentence uncited;
* the RAG baseline restates each source it was shown, then adds a vague
  summary that no source states;
* ResearchMind's claim extractor takes each passage's first sentence, and its
  synthesizer restates its sources (quick) or claims (verified modes);
* the judge decides support by word overlap, and coverage by keywords.

So the numbers it produces check the harness, end to end, and measure
nothing about the real systems. ``python -m researchmind_agents.evals run
--suite offline`` needs no network or API keys.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from researchmind_agents.config import AgentConfig, EvalConfig, LLMConfig
from researchmind_agents.evals.benchmark import BenchmarkQuestion
from researchmind_agents.llm.base import BaseLLMProvider, UsageSink
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.testing import FakeWeb, Responder, ScriptedLLM, build_fake_runtime
from researchmind_agents.tools.excerpts import split_sentences
from researchmind_agents.tools.page_extract import PageContent

__all__ = [
    "OFFLINE_QUESTIONS",
    "OFFLINE_PAGES",
    "OfflineWeb",
    "offline_config",
    "offline_web",
    "offline_responses",
    "judge_responses",
    "offline_runtime_factory",
    "offline_judge_factory",
]

_FILLER = (
    " This page belongs to a synthetic corpus written to exercise the ResearchMind "
    "evaluation harness without network access, so none of it describes a real product. "
    "The remaining text repeats background detail that no answer needs to cite."
)


def _page(title: str, text: str) -> tuple[str, str]:
    return title, text + _FILLER * 2


OFFLINE_PAGES: dict[str, tuple[str, str]] = {
    # quick: Lumen DB
    "https://docs.lumendb.example/releases/4.2": _page(
        "Lumen DB 4.2 release notes",
        "Lumen DB 4.2 reduced median query latency by 30 percent compared with Lumen DB 4.1 "
        "on the TPC-H benchmark. Write performance in Lumen DB 4.2 is unchanged from 4.1.",
    ),
    "https://journal.example.org/articles/lumen-reproduction": _page(
        "Reproducing Lumen DB benchmarks",
        "An independent reproduction found that Lumen DB 4.2 lowered median query latency "
        "by 29 to 31 percent relative to Lumen DB 4.1. The study ran five trials on identical hardware.",
    ),
    "https://forum.example.net/threads/lumen-upgrade": _page(
        "Upgrading to Lumen DB 4.2",
        "Several users report that upgrading from Lumen DB 4.1 to Lumen DB 4.2 took under an hour. "
        "One user saw slower bulk loads after the upgrade.",
    ),
    # deep: Kestrel payments API
    "https://docs.kestrel.example/limits": _page(
        "Kestrel payments API rate limits",
        "Kestrel payments API free-tier accounts may send up to 60 requests per minute. "
        "Paid Kestrel plans start at 600 requests per minute.",
    ),
    "https://docs.kestrel.example/pricing": _page(
        "Kestrel payments pricing",
        "Kestrel charges 2.5 percent per successful payment with no monthly fee for a small "
        "online shop. Refunds through the Kestrel payments API are free.",
    ),
    "https://news.example.com/2026/kestrel-outage": _page(
        "Kestrel payments outage",
        "The Kestrel payments API was unavailable for four hours in March 2026. Kestrel said "
        "the outage affected card payments for online shops in Europe.",
    ),
    # academic: Orrin sleep app
    "https://journal.example.org/articles/orrin-trial": _page(
        "A randomised trial of the Orrin sleep app",
        "In a randomised trial of 400 adults, the Orrin sleep app improved sleep quality scores "
        "by 12 percent over eight weeks. The trial compared the Orrin app with a sleep-hygiene leaflet.",
    ),
    "https://preprints.example.org/orrin-follow-up": _page(
        "Orrin sleep app: one-year follow-up study",
        "A one-year follow-up study of the Orrin sleep app found that sleep quality gains faded "
        "after six months. Only half of the participants still used the Orrin app after a year.",
    ),
    "https://blog.example.com/orrin-review": _page(
        "Orrin sleep app review",
        "This review of the Orrin sleep app praises its bedtime reminders and calm design. "
        "The reviewer did not measure sleep quality.",
    ),
    # decision: Vela Cloud or Nimbus Cloud
    "https://docs.velacloud.example/pricing": _page(
        "Vela Cloud pricing for small teams",
        "Vela Cloud hosts a small web app for 25 dollars a month including a managed database. "
        "Vela Cloud support answers within one business day.",
    ),
    "https://docs.nimbuscloud.example/pricing": _page(
        "Nimbus Cloud pricing for small teams",
        "Nimbus Cloud hosts a small web app for 40 dollars a month, and a managed database costs "
        "15 dollars more. Nimbus Cloud offers around-the-clock support on every plan.",
    ),
    "https://news.example.com/2026/vela-nimbus-comparison": _page(
        "Vela Cloud versus Nimbus Cloud for a small team",
        "For a small team, Vela Cloud costs less than Nimbus Cloud for a web app with a managed "
        "database. Nimbus Cloud has more regions than Vela Cloud.",
    ),
}

OFFLINE_QUESTIONS: tuple[BenchmarkQuestion, ...] = (
    BenchmarkQuestion(
        id="off-q1",
        mode="quick",
        question="How much faster is Lumen DB 4.2 than Lumen DB 4.1?",
        expected_subtopics=("query latency", "write performance", "independent reproduction"),
        gold_sources=("docs.lumendb.example/releases", "journal.example.org/articles/lumen"),
    ),
    BenchmarkQuestion(
        id="off-d1",
        mode="deep",
        question="Is the Kestrel payments API a good fit for a small online shop?",
        expected_subtopics=("rate limits", "payment pricing", "outage reliability"),
        gold_sources=("docs.kestrel.example", "news.example.com/2026/kestrel"),
    ),
    BenchmarkQuestion(
        id="off-a1",
        mode="academic",
        question="What do studies find about the Orrin sleep app and sleep quality?",
        expected_subtopics=("sleep quality outcomes", "trial design", "long-term follow-up"),
        gold_sources=("journal.example.org/articles/orrin", "preprints.example.org/orrin"),
    ),
    BenchmarkQuestion(
        id="off-x1",
        mode="decision",
        question="Should a small team host its web app on Vela Cloud or Nimbus Cloud?",
        user_context={"team": "two developers", "budget": "under 100 dollars a month"},
        expected_subtopics=("monthly cost", "managed database", "support", "clear recommendation"),
        gold_sources=("docs.velacloud.example", "docs.nimbuscloud.example"),
    ),
)


@dataclass
class OfflineWeb(FakeWeb):
    """A :class:`FakeWeb` whose unknown URLs answer like a real 404."""

    def fetch(self, url: str) -> PageContent:
        page = super().fetch(url)
        if page.error and url not in self.pages:
            return PageContent(url=url, error=f"Client error '404 Not Found' for url '{url}'")
        return page


def offline_web() -> OfflineWeb:
    return OfflineWeb(pages=dict(OFFLINE_PAGES))


def offline_config() -> AgentConfig:
    """Defaults, with scripted model names so results say what produced them."""
    return AgentConfig(
        llm=LLMConfig(model="scripted", base_delay=0.0, overload_delay=0.0),
        eval=EvalConfig(judge_model="scripted-judge"),
    )


# ---------------------------------------------------------------------------
# Scripted responses
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP_TEXT = (
    "the and for with that this from are was were has have had not but its into than then "
    "they them their there which while what when where who will would can could should may "
    "might also such these those over under more most less about after before does did each "
    "other some any all only how much"
)
_STOP = frozenset(_STOP_TEXT.split())
def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) >= 3 and w not in _STOP}


def _first_sentence(text: str) -> str:
    # The real splitter: "Lumen DB 4.2 ..." is one sentence, not "Lumen DB 4."
    sentences = split_sentences(text)
    return sentences[0].strip() if sentences else text[:160].strip()


def _question_of(prompt: str, marker: str = "Question:\n") -> str:
    return prompt.split(marker, 1)[-1].split("\n", 1)[0].strip()


def _closed_book(web: FakeWeb) -> Responder:
    def respond(prompt: str) -> str:
        question = _terms(_question_of(prompt))
        url, (_, text) = max(
            web.pages.items(), key=lambda item: len(question & _terms(f"{item[1][0]} {item[1][1]}"))
        )
        slug = next(iter(sorted(question)), "topic")
        return json.dumps(
            {
                "statements": [
                    {"text": _first_sentence(text), "sources": [url]},
                    {
                        "text": "Most industry analysts consider it the leading option in its category.",
                        "sources": [f"https://facts.example.org/{slug}-guide"],
                    },
                    {"text": "Results vary from one deployment to another.", "sources": []},
                ]
            }
        )

    return respond


_SOURCE_HEADER = re.compile(r"^\[(S\d+)\] .*\nsite: .*\n- (.*)$", re.MULTILINE)


def _rag_answer(prompt: str) -> str:
    statements: list[dict[str, Any]] = [
        {"text": _first_sentence(text), "citations": [label]}
        for label, text in _SOURCE_HEADER.findall(prompt)
    ]
    statements.append(
        {"text": "Overall, the evidence clearly settles the question.", "citations": ["S1"]}
    )
    return json.dumps({"statements": statements})


def _quick_report(prompt: str) -> str:
    sources = _SOURCE_HEADER.findall(prompt)
    first = [{"text": _first_sentence(sources[0][1]), "citations": [sources[0][0]]}] if sources else []
    return json.dumps(
        {
            "title": "Offline report",
            "executive_summary": first,
            "sections": [
                {
                    "heading": f"Evidence from {label}",
                    "statements": [{"text": _first_sentence(text), "citations": [label]}],
                }
                for label, text in sources
            ],
            "limitations": ["Scripted offline report."],
        }
    )


_PASSAGE_HEADER = re.compile(r"^\[(P\d+)\] ", re.MULTILINE)


def _claims(prompt: str) -> str:
    """One claim per passage: its first sentence, citing that passage."""
    parts = _PASSAGE_HEADER.split(prompt)
    claims = []
    # parts: [preamble, label1, body1, label2, body2, ...]; a body is title, site, text.
    for label, body in zip(parts[1::2], parts[2::2], strict=False):
        text = " ".join(line for line in body.split("\n")[2:] if "UNTRUSTED" not in line)
        if text.strip():
            claims.append({"text": _first_sentence(text), "passages": [label]})
    return json.dumps({"claims": claims})


_CLAIM_LINE = re.compile(r"^\[(C\d+)\] \([^)]*\) (.*)$", re.MULTILINE)


def _verified_report(prompt: str, *, recommendation: bool = False) -> str:
    claims = _CLAIM_LINE.findall(prompt)
    first = claims[:1]
    report: dict[str, Any] = {
        "title": "Offline verified report",
        "executive_summary": [{"text": text, "claims": [label]} for label, text in first],
        "sections": [
            {"heading": f"Finding {label}", "statements": [{"text": text, "claims": [label]}]}
            for label, text in claims
        ],
        "limitations": ["Scripted offline report."],
    }
    if recommendation:
        report["recommendation"] = [
            {"text": f"Choose on this evidence: {text}", "claims": [label]} for label, text in first
        ]
    return json.dumps(report)


def offline_responses(web: FakeWeb) -> dict[str, Responder | list[Responder]]:
    """Scripted answers for the systems (planner, researchers, ... use the testing defaults)."""
    return {
        "ClosedBookAnswer": _closed_book(web),
        "RagAnswer": _rag_answer,
        "ClaimExtractionOutput": _claims,
        "ReportOutput": _quick_report,
        "VerifiedReportOutput": _verified_report,
        "DecisionReportOutput": lambda prompt: _verified_report(prompt, recommendation=True),
    }


_STATEMENT_LINE = re.compile(r"^\[(A\d+)\] (.*)$")
_EVIDENCE_LINE = re.compile(r"^  \[(E\d+)\] ")
_TOPIC_LINE = re.compile(r"^\[(T\d+)\] (.*)$", re.MULTILINE)
_ANSWER_LINE = re.compile(r"^\[(A\d+)\] (.*)$", re.MULTILINE)


def _support(prompt: str) -> str:
    """Supports when most of the sentence's words are in the excerpt."""
    judgements: list[dict[str, Any]] = []
    lines = prompt.splitlines()
    statement_terms: set[str] = set()
    for index, line in enumerate(lines):
        statement = _STATEMENT_LINE.match(line)
        evidence = _EVIDENCE_LINE.match(line)
        if statement:
            statement_terms = _terms(statement.group(2))
            judgements.append(
                {"statement": statement.group(1), "verdict": "unsupported", "evidence": [], "reason": "Scripted."}
            )
        elif evidence and judgements and index + 1 < len(lines):
            excerpt = _terms(lines[index + 1])
            share = len(statement_terms & excerpt) / max(1, len(statement_terms))
            stance = "supports" if share >= 0.6 else "neutral"
            judgements[-1]["evidence"].append({"evidence": evidence.group(1), "stance": stance})
            if stance == "supports":
                judgements[-1]["verdict"] = "supported"
    return json.dumps({"judgements": judgements})


def _coverage(prompt: str) -> str:
    """A topic is covered when half its words appear in one sentence."""
    answer = dict(_ANSWER_LINE.findall(prompt))
    topics = _TOPIC_LINE.findall(prompt)
    topic_terms = set().union(*(_terms(t) for _, t in topics)) if topics else set()
    question = _terms(_question_of(prompt))
    covered = []
    for label, topic in topics:
        terms = _terms(topic)
        hit = any(len(terms & _terms(text)) * 2 >= len(terms) for text in answer.values())
        covered.append({"topic": label, "covered": bool(terms) and hit})
    off_topic = [label for label, text in answer.items() if not _terms(text) & (question | topic_terms)]
    return json.dumps({"topics": covered, "off_topic": off_topic})


def judge_responses() -> dict[str, Responder | list[Responder]]:
    return {"SupportJudgement": _support, "CoverageJudgement": _coverage}


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def offline_runtime_factory(config: AgentConfig) -> AgentRuntime:
    web = offline_web()
    return build_fake_runtime(config, web=web, responses=offline_responses(web))


def offline_judge_factory(config: AgentConfig, usage_sink: UsageSink | None) -> BaseLLMProvider:
    judge_config = replace(
        config.llm, model=config.eval.judge_model or config.llm.model, base_delay=0.0, overload_delay=0.0
    )
    return ScriptedLLM(judge_config, usage_sink, responses=judge_responses())
