"""
Contradiction eval: does the detector find seeded contradictions, and name
their causes?

``contradictions_gold.jsonl`` holds 15 cases, each a small set of verified
claims on one question with the passages behind them: 12 cases seed 14
contradictions across every likely cause (conditions, date, version, dataset,
methodology, definition, unexplained), and 3 cases have none — compatible
claims that only look alike. Every positive case also has a claim that
conflicts with nothing. The passages are synthetic: fictional products and
studies on ``.example`` hosts, so the model cannot answer from memory and no
text is attributed to a real organisation.

:func:`evaluate_contradictions` runs the real detector
(``nodes.contradiction_detector.detect_contradictions``) on each case — the
runtime's embeddings pick the candidate pairs, its model judges them:

* **recall** — share of seeded contradictions found: a detected contradiction
  contains at least two of its claims (both, for a pair); each detection
  matches at most one seeded contradiction;
* **cause agreement** — share of found contradictions whose likely cause
  matches the label;
* **precision** — share of detected contradictions that match a seeded one;
* **false positives** — detections in the cases with no contradiction;
* **candidate recall** — share of seeded contradictions whose claims were all
  paired as candidates (a diagnostic for ``CONTRADICTION_MIN_SIMILARITY``).

The M6 bar: recall at least 70%. Offline, a scripted model that returns the
labels checks the harness and candidate selection; with a live model
(``backend/scripts/eval_contradictions.py``) it measures the detector.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchmind_agents.nodes.common import EventLog
from researchmind_agents.nodes.contradiction_detector import AGENT, detect_contradictions
from researchmind_agents.nodes.fact_checker import SourceInfo
from researchmind_agents.reliability import (
    PAPER_TYPES,
    SourceSignals,
    classify_domain,
    describes_methodology,
    source_reliability,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import ClaimItem, EvidenceItem

__all__ = [
    "GOLD_PATH",
    "EVAL_NOW",
    "TARGET",
    "GoldPassage",
    "GoldClaim",
    "GoldContradiction",
    "GoldCase",
    "CaseOutcome",
    "ContradictionResult",
    "load_gold",
    "evaluate_contradictions",
]

GOLD_PATH = Path(__file__).with_name("contradictions_gold.jsonl")
# Labels were written as of this date; recency is measured from it.
EVAL_NOW = datetime(2026, 10, 1, tzinfo=UTC)
TARGET = 0.7


@dataclass(frozen=True)
class GoldPassage:
    url: str
    title: str
    text: str
    stance: str = "supports"
    domain_type: str = ""
    published_at: str = ""


@dataclass(frozen=True)
class GoldClaim:
    id: str
    text: str
    verdict: str
    evidence: tuple[GoldPassage, ...]


@dataclass(frozen=True)
class GoldContradiction:
    claims: tuple[str, ...]
    likely_cause: str
    note: str = ""


@dataclass(frozen=True)
class GoldCase:
    id: str
    question: str
    claims: tuple[GoldClaim, ...]
    contradictions: tuple[GoldContradiction, ...]
    note: str = ""


@dataclass
class CaseOutcome:
    id: str
    # Per seeded contradiction: (claim ids, expected cause, found cause or None).
    seeded: list[tuple[list[str], str, str | None]] = field(default_factory=list)
    # Per detection: (claim ids, cause, matches a seeded contradiction).
    detected: list[tuple[list[str], str, bool]] = field(default_factory=list)
    # Seeded contradictions whose claims were all candidates.
    candidates_found: int = 0


@dataclass
class ContradictionResult:
    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def seeded(self) -> int:
        return sum(len(o.seeded) for o in self.outcomes)

    @property
    def found(self) -> int:
        return sum(1 for o in self.outcomes for _, _, got in o.seeded if got is not None)

    @property
    def recall(self) -> float:
        return self.found / self.seeded if self.seeded else 0.0

    @property
    def cause_agreement(self) -> float:
        pairs = [(want, got) for o in self.outcomes for _, want, got in o.seeded if got is not None]
        return _share(want == got for want, got in pairs)

    @property
    def precision(self) -> float:
        return _share(match for o in self.outcomes for _, _, match in o.detected)

    @property
    def false_positives(self) -> int:
        return sum(len(o.detected) for o in self.outcomes if not o.seeded)

    @property
    def candidate_recall(self) -> float:
        total = self.seeded
        return sum(o.candidates_found for o in self.outcomes) / total if total else 0.0

    def passed(self, target: float = TARGET) -> bool:
        return self.recall >= target

    def summary(self) -> dict[str, Any]:
        return {
            "cases": len(self.outcomes),
            "seeded": self.seeded,
            "found": self.found,
            "recall": round(self.recall, 4),
            "cause_agreement": round(self.cause_agreement, 4),
            "precision": round(self.precision, 4),
            "false_positives": self.false_positives,
            "candidate_recall": round(self.candidate_recall, 4),
            "passed": self.passed(),
            "outcomes": [asdict(o) for o in self.outcomes],
        }


def _share(flags: Iterable[bool]) -> float:
    values = list(flags)
    return sum(values) / len(values) if values else 0.0


def load_gold(path: Path = GOLD_PATH) -> list[GoldCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            GoldCase(
                id=raw["id"],
                question=raw["question"],
                note=raw.get("note", ""),
                claims=tuple(
                    GoldClaim(
                        id=c["id"],
                        text=c["text"],
                        verdict=c.get("verdict", "supported"),
                        evidence=tuple(GoldPassage(**e) for e in c["evidence"]),
                    )
                    for c in raw["claims"]
                ),
                contradictions=tuple(
                    GoldContradiction(
                        claims=tuple(x["claims"]),
                        likely_cause=x["likely_cause"],
                        note=x.get("note", ""),
                    )
                    for x in raw["contradictions"]
                ),
            )
        )
    return cases


async def evaluate_contradictions(
    runtime: AgentRuntime,
    cases: Sequence[GoldCase] | None = None,
    *,
    now: datetime = EVAL_NOW,
) -> ContradictionResult:
    """Run the detector on every gold case with *runtime* and compare to the labels."""
    cases = list(cases) if cases is not None else load_gold()
    result = ContradictionResult()
    for case in cases:
        run_id = f"eval-contradictions-{uuid.uuid4()}"
        log = EventLog(runtime, run_id=run_id, agent=AGENT)
        claims, sources, by_gold_id = _build(case, runtime, now)
        detection = await detect_contradictions(
            runtime,
            run_id=run_id,
            question=case.question,
            claims=claims,
            sources=sources,
            log=log,
            llm_calls=[],
        )
        gold_of = {claim_id: gold_id for gold_id, claim_id in by_gold_id.items()}
        outcome = CaseOutcome(id=case.id)
        found = [
            ([gold_of[cid] for cid in item.claim_ids], str(item.likely_cause))
            for item in detection.contradictions
        ]
        candidate_pairs = {frozenset((gold_of[p.a], gold_of[p.b])) for p in detection.candidates}
        # One detection matches at most one seeded contradiction.
        used: set[int] = set()
        for seeded in case.contradictions:
            want = set(seeded.claims)
            index = next(
                (i for i, (ids, _) in enumerate(found) if i not in used and _matches(set(ids), want)),
                None,
            )
            if index is not None:
                used.add(index)
            match = found[index][1] if index is not None else None
            outcome.seeded.append((list(seeded.claims), seeded.likely_cause, match))
            pairs = {frozenset((a, b)) for a in want for b in want if a < b}
            if pairs <= candidate_pairs:
                outcome.candidates_found += 1
        for i, (ids, cause) in enumerate(found):
            outcome.detected.append((ids, cause, i in used))
        result.outcomes.append(outcome)
    return result


def _matches(detected: set[str], seeded: set[str]) -> bool:
    """A detection matches a seeded contradiction with two or more of its claims."""
    return len(detected & seeded) >= min(2, len(seeded))


def _build(
    case: GoldCase, runtime: AgentRuntime, now: datetime
) -> tuple[list[ClaimItem], dict[str, SourceInfo], dict[str, str]]:
    claims: list[ClaimItem] = []
    sources: dict[str, SourceInfo] = {}
    by_gold_id: dict[str, str] = {}
    for gold in case.claims:
        evidence = []
        for passage in gold.evidence:
            source_id = str(uuid.uuid4())
            domain_type = passage.domain_type or classify_domain(passage.url)
            published = _date(passage.published_at)
            sources[source_id] = SourceInfo(
                source_id=source_id,
                url=passage.url,
                title=passage.title,
                domain_type=domain_type,
                reliability=source_reliability(
                    SourceSignals(
                        domain_type=domain_type,
                        published_at=published,
                        has_methodology=domain_type in PAPER_TYPES
                        and describes_methodology(passage.text),
                    ),
                    now=now,
                    config=runtime.config.scoring,
                ),
                published_at=published.isoformat() if published else None,
            )
            evidence.append(
                EvidenceItem(
                    passage_id=str(uuid.uuid4()),
                    source_id=source_id,
                    stance=passage.stance,  # type: ignore[arg-type]
                    excerpt=passage.text,
                )
            )
        claim_id = f"{case.id}-{gold.id}-{uuid.uuid4().hex[:8]}"
        by_gold_id[gold.id] = claim_id
        claims.append(
            ClaimItem(
                id=claim_id,
                text=gold.text,
                passage_ids=[e.passage_id for e in evidence if e.stance == "supports"][:1],
                evidence=evidence,
                verdict=gold.verdict,  # type: ignore[arg-type]
                confidence=0.5,
            )
        )
    return claims, sources, by_gold_id


def _date(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
