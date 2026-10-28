"""
Verification eval: does the fact checker agree with hand-labelled claims?

``claims_gold.jsonl`` holds 20 claims across the four verdicts. Each comes
with its evidence passages, each passage's source type, date, and expected
stance, the expected verdict, and a hand-labelled confidence. The passages are
synthetic: written for this eval about fictional products and studies on
``example`` hosts, so the model cannot answer from memory and no text is
attributed to a real organisation.

:func:`evaluate_verification` runs the real fact checker
(``nodes.fact_checker.check_claims``) and the real scoring on every claim:

* **verdict agreement** — share of claims whose verdict matches the label;
* **confidence within tolerance** — share whose computed confidence is
  within ``tolerance`` (0.15) of the label; also the mean absolute error.

The M5 bar: both shares at least 80%. Offline, a scripted model that returns
the labelled stances checks the scoring's calibration; with a live model
(``backend/scripts/eval_verification.py``) it measures the fact checker.
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
from researchmind_agents.nodes.fact_checker import (
    AGENT,
    CandidatePassage,
    SourceInfo,
    check_claims,
)
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
    "TOLERANCE",
    "TARGET",
    "GoldEvidence",
    "GoldClaim",
    "ClaimOutcome",
    "VerificationResult",
    "load_gold",
    "evaluate_verification",
]

GOLD_PATH = Path(__file__).with_name("claims_gold.jsonl")
# Labels were written as of this date; recency is measured from it.
EVAL_NOW = datetime(2026, 10, 1, tzinfo=UTC)
TOLERANCE = 0.15
TARGET = 0.8


@dataclass(frozen=True)
class GoldEvidence:
    url: str
    title: str
    text: str
    stance: str
    domain_type: str = ""
    published_at: str = ""


@dataclass(frozen=True)
class GoldClaim:
    id: str
    question: str
    claim: str
    verdict: str
    confidence: float
    evidence: tuple[GoldEvidence, ...]
    note: str = ""


@dataclass
class ClaimOutcome:
    id: str
    claim: str
    expected_verdict: str
    verdict: str
    expected_confidence: float
    confidence: float
    expected_stances: list[str]
    stances: list[str]

    @property
    def verdict_ok(self) -> bool:
        return self.verdict == self.expected_verdict

    @property
    def confidence_error(self) -> float:
        return abs(self.confidence - self.expected_confidence)


@dataclass
class VerificationResult:
    outcomes: list[ClaimOutcome] = field(default_factory=list)
    tolerance: float = TOLERANCE

    @property
    def verdict_agreement(self) -> float:
        return _share(o.verdict_ok for o in self.outcomes)

    @property
    def confidence_within(self) -> float:
        return _share(o.confidence_error <= self.tolerance + 1e-9 for o in self.outcomes)

    @property
    def confidence_mae(self) -> float:
        if not self.outcomes:
            return 0.0
        return round(sum(o.confidence_error for o in self.outcomes) / len(self.outcomes), 4)

    @property
    def stance_agreement(self) -> float:
        pairs = [
            (got, want)
            for o in self.outcomes
            for got, want in zip(o.stances, o.expected_stances, strict=False)
        ]
        return _share(got == want for got, want in pairs)

    def passed(self, target: float = TARGET) -> bool:
        return self.verdict_agreement >= target and self.confidence_within >= target

    def summary(self) -> dict[str, Any]:
        return {
            "claims": len(self.outcomes),
            "verdict_agreement": round(self.verdict_agreement, 4),
            "confidence_within_tolerance": round(self.confidence_within, 4),
            "confidence_mae": self.confidence_mae,
            "stance_agreement": round(self.stance_agreement, 4),
            "tolerance": self.tolerance,
            "passed": self.passed(),
            "outcomes": [
                {**asdict(o), "verdict_ok": o.verdict_ok, "confidence_error": round(o.confidence_error, 4)}
                for o in self.outcomes
            ],
        }


def _share(flags: Iterable[bool]) -> float:
    values = list(flags)
    return sum(values) / len(values) if values else 0.0


def load_gold(path: Path = GOLD_PATH) -> list[GoldClaim]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            GoldClaim(
                id=raw["id"],
                question=raw["question"],
                claim=raw["claim"],
                verdict=raw["verdict"],
                confidence=float(raw["confidence"]),
                evidence=tuple(GoldEvidence(**e) for e in raw["evidence"]),
                note=raw.get("note", ""),
            )
        )
    return items


async def evaluate_verification(
    runtime: AgentRuntime,
    items: Sequence[GoldClaim] | None = None,
    *,
    tolerance: float = TOLERANCE,
    now: datetime = EVAL_NOW,
) -> VerificationResult:
    """Fact-check every gold claim with *runtime*'s model and compare to the labels."""
    items = list(items) if items is not None else load_gold()
    run_id = f"eval-verification-{uuid.uuid4()}"
    log = EventLog(runtime, run_id=run_id, agent=AGENT)

    claims: list[ClaimItem] = []
    candidates: dict[str, list[CandidatePassage]] = {}
    sources: dict[str, SourceInfo] = {}
    for item in items:
        passages = []
        for evidence in item.evidence:
            source_id = str(uuid.uuid4())
            domain_type = evidence.domain_type or classify_domain(evidence.url)
            published = _date(evidence.published_at)
            reliability = source_reliability(
                SourceSignals(
                    domain_type=domain_type,
                    published_at=published,
                    has_methodology=domain_type in PAPER_TYPES
                    and describes_methodology(evidence.text),
                ),
                now=now,
                config=runtime.config.scoring,
            )
            sources[source_id] = SourceInfo(
                source_id=source_id,
                url=evidence.url,
                title=evidence.title,
                domain_type=domain_type,
                reliability=reliability,
                published_at=published.isoformat() if published else None,
            )
            passages.append(
                CandidatePassage(
                    passage_id=str(uuid.uuid4()),
                    source_id=source_id,
                    text=evidence.text,
                    url=evidence.url,
                    title=evidence.title,
                )
            )
        claim = ClaimItem(
            id=item.id,
            text=item.claim,
            passage_ids=[passages[0].passage_id],
            evidence=[EvidenceItem(passage_id=p.passage_id, source_id=p.source_id) for p in passages],
        )
        claims.append(claim)
        candidates[claim.id] = passages

    # Every claim is checked on its own question, as in a real run.
    checked: list[ClaimItem] = []
    for item, claim in zip(items, claims, strict=True):
        checked += await check_claims(
            runtime,
            run_id=run_id,
            question=item.question,
            claims=[claim],
            candidates={claim.id: candidates[claim.id]},
            sources=sources,
            log=log,
            llm_calls=[],
        )

    result = VerificationResult(tolerance=tolerance)
    for item, claim in zip(items, checked, strict=True):
        result.outcomes.append(
            ClaimOutcome(
                id=item.id,
                claim=item.claim,
                expected_verdict=item.verdict,
                verdict=claim.verdict,
                expected_confidence=item.confidence,
                confidence=claim.confidence,
                expected_stances=[e.stance for e in item.evidence],
                stances=[e.stance for e in claim.evidence],
            )
        )
    return result


def _date(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
