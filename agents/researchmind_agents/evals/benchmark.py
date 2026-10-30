"""
Benchmark suites for the M7 evaluation (PRD s11).

A suite is a JSONL file of questions. Each line has:

* ``id``, ``mode`` (quick, deep, academic, decision), ``question``;
* ``expected_subtopics`` — what a complete answer covers (hand-labelled);
* ``gold_sources`` — sites or pages a careful researcher would consult,
  written without a scheme: ``"energy.gov"`` or
  ``"github.com/pgvector/pgvector"`` (hand-labelled);
* optionally ``user_context`` (decision mode), passed to every system.

Suites:

* ``default`` — ``benchmark.jsonl``: 40 questions, 10 per mode;
* ``seed`` — ``seed.jsonl``: the M3 seed set (8 quick questions, no gold
  sources, so no retrieval metrics);
* ``offline`` — a synthetic world run entirely on test doubles
  (``evals/offline.py``), so the harness runs with no network or keys.

A path to any JSONL file in this format also works as a suite.

Gold matching is deliberately lenient, since live search finds different pages
from one day to the next: a retrieved URL matches a gold source when its host
is the gold host or a subdomain of it, and its path starts with the gold path.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from researchmind_agents.modes import SUPPORTED_MODES

__all__ = [
    "BENCHMARK_PATH",
    "SEED_PATH",
    "OFFLINE_SUITE",
    "SUITES",
    "BenchmarkQuestion",
    "load_questions",
    "load_suite",
    "gold_match",
    "matching_gold",
    "suite_summary",
]

_HERE = Path(__file__).parent
BENCHMARK_PATH = _HERE / "benchmark.jsonl"
SEED_PATH = _HERE / "seed.jsonl"
OFFLINE_SUITE = "offline"

# Named suites backed by a file; "offline" is built in code (evals/offline.py).
SUITES: dict[str, Path] = {"default": BENCHMARK_PATH, "seed": SEED_PATH}


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    question: str
    mode: str = "quick"
    expected_subtopics: tuple[str, ...] = ()
    gold_sources: tuple[str, ...] = ()
    user_context: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"question {self.id!r}: unknown mode {self.mode!r}")
        if not self.question.strip():
            raise ValueError(f"question {self.id!r} is empty")


def load_questions(path: Path | str) -> list[BenchmarkQuestion]:
    """Read a JSONL suite file. Raises ``ValueError`` on a bad line or a repeated ID."""
    questions: list[BenchmarkQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            item = BenchmarkQuestion(
                id=str(raw["id"]),
                question=str(raw["question"]),
                mode=str(raw.get("mode", "quick")),
                expected_subtopics=tuple(str(t) for t in raw.get("expected_subtopics", ())),
                gold_sources=tuple(str(g) for g in raw.get("gold_sources", ())),
                user_context=dict(raw.get("user_context") or {}),
                note=str(raw.get("note", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if item.id in seen:
            raise ValueError(f"{path}:{number}: repeated question id {item.id!r}")
        seen.add(item.id)
        questions.append(item)
    return questions


def load_suite(name: str) -> list[BenchmarkQuestion]:
    """Questions of a named suite (``default``, ``seed``, ``offline``) or a JSONL path."""
    if name == OFFLINE_SUITE:
        from researchmind_agents.evals.offline import OFFLINE_QUESTIONS

        return list(OFFLINE_QUESTIONS)
    if name in SUITES:
        return load_questions(SUITES[name])
    path = Path(name)
    if path.suffix == ".jsonl" and path.is_file():
        return load_questions(path)
    known = ", ".join([*SUITES, OFFLINE_SUITE])
    raise ValueError(f"unknown suite {name!r}; use one of {known}, or a .jsonl path")


# ---------------------------------------------------------------------------
# Gold sources
# ---------------------------------------------------------------------------


def _host_and_path(value: str) -> tuple[str, str]:
    """``"https://www.Example.org/a/b/"`` or ``"example.org/a/b"`` -> ``("example.org", "/a/b")``."""
    text = value.strip()
    if "://" not in text:
        text = "https://" + text
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower()
    except ValueError:
        return "", ""
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")
    return host, path


def gold_match(url: str, gold: str) -> bool:
    """Does *url* fall under the gold source *gold* (host or subdomain, then path prefix)?"""
    url_host, url_path = _host_and_path(url)
    gold_host, gold_path = _host_and_path(gold)
    if not url_host or not gold_host:
        return False
    if url_host != gold_host and not url_host.endswith("." + gold_host):
        return False
    return url_path.lower().startswith(gold_path.lower())


def matching_gold(url: str, gold_sources: Iterable[str]) -> str | None:
    """The first gold source *url* matches, or ``None``."""
    for gold in gold_sources:
        if gold_match(url, gold):
            return gold
    return None


def suite_summary(questions: Sequence[BenchmarkQuestion]) -> dict[str, int]:
    """Questions per mode, in mode order."""
    counts: dict[str, int] = {}
    for item in questions:
        counts[item.mode] = counts.get(item.mode, 0) + 1
    return counts
