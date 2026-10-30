"""
LLM-as-judge for answer quality (PRD s11), and the deterministic steps around it.

**Support** (``eval_support/v1``). For each answer sentence, the judge reads
one excerpt per *resolved* citation: the part of that source's evidence most
relevant to the sentence (``tools/excerpts.best_excerpt``, at most
``EVAL_JUDGE_EXCERPT_CHARS``). It returns a stance per excerpt and a verdict
per sentence. A sentence with no resolved citation never reaches the judge:
it is ``unsupported`` by rule. Verdicts are then reconciled with the stances
(:func:`reconcile_verdict`), so a sentence cannot be "supported" by excerpts
the judge itself called neutral.

**Coverage** (``eval_coverage/v1``). Given the expected sub-topics and the
answer's sentences, the judge says which sub-topics are covered and which
sentences are off-topic.

Judge failures never fail a run: the sentences (or the coverage) are left
unjudged, excluded from the metrics, and the error is recorded.

Only the first ``EVAL_MAX_STATEMENTS`` sentences of an answer are judged.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from researchmind_agents.config import EvalConfig
from researchmind_agents.evals.schemas import CoverageJudgement, SupportJudgement
from researchmind_agents.evals.systems import AnswerStatement
from researchmind_agents.llm.base import BaseLLMProvider
from researchmind_agents.nodes.common import llm_call_record
from researchmind_agents.prompts.eval_judge import (
    COVERAGE_PROMPT_VERSION,
    COVERAGE_SYSTEM_PROMPT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    JudgeEvidence,
    JudgeStatement,
    build_coverage_prompt,
    build_support_prompt,
)
from researchmind_agents.tools.excerpts import best_excerpt
from researchmind_agents.tools.urls import domain_of

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT",
    "JudgedCitation",
    "JudgedStatement",
    "SupportResult",
    "CoverageResult",
    "reconcile_verdict",
    "judge_support",
    "judge_coverage",
]

AGENT = "eval_judge"


@dataclass
class JudgedCitation:
    label: str
    url: str = ""
    title: str = ""
    status: str = "resolved"
    reason: str = ""
    # supports / contradicts / neutral; None when not judged (not resolved,
    # or the judge call failed).
    stance: str | None = None
    excerpt: str = ""


@dataclass
class JudgedStatement:
    text: str
    citations: list[JudgedCitation] = field(default_factory=list)
    # supported / partial / unsupported / contradicted; None when unjudged.
    verdict: str | None = None
    # "judge" (the model decided) or "rule" (no resolved citation).
    judged_by: str = "rule"
    reason: str = ""

    @property
    def judged(self) -> bool:
        return self.verdict is not None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgedStatement:
        return cls(
            text=data["text"],
            citations=[JudgedCitation(**c) for c in data.get("citations", [])],
            verdict=data.get("verdict"),
            judged_by=data.get("judged_by", "rule"),
            reason=data.get("reason", ""),
        )


@dataclass
class SupportResult:
    statements: list[JudgedStatement] = field(default_factory=list)
    # Sentences beyond EVAL_MAX_STATEMENTS, not judged.
    truncated: int = 0
    errors: list[str] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CoverageResult:
    topics: list[str] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    statements: int = 0
    off_topic: int = 0
    # The sentences judged off-topic, for checking the judge.
    off_topic_sentences: list[str] = field(default_factory=list)
    # The judge ran and answered.
    judged: bool = False
    error: str | None = None
    llm_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def completeness(self) -> float | None:
        if self.error is not None or not self.topics:
            return None
        if not self.statements:
            return 0.0  # an empty answer covers nothing; no judge needed
        return len(self.covered) / len(self.topics) if self.judged else None

    @property
    def relevance(self) -> float | None:
        if self.error is not None or not self.judged or not self.statements:
            return None
        return 1 - self.off_topic / self.statements

    def as_dict(self) -> dict[str, Any]:
        return {
            "completeness": self.completeness,
            "relevance": self.relevance,
            "topics": self.topics,
            "covered": self.covered,
            "statements": self.statements,
            "off_topic": self.off_topic,
            "off_topic_sentences": self.off_topic_sentences,
            "judged": self.judged,
            "error": self.error,
        }


def reconcile_verdict(verdict: str, stances: Sequence[str]) -> str:
    """
    Make a verdict agree with the stances behind it:

    * no excerpt supports the sentence -> ``contradicted`` if one contradicts
      it, else ``unsupported``;
    * some excerpt supports it but the verdict says ``unsupported`` ->
      ``partial`` (it backs at least part of the sentence);
    * otherwise the judge's verdict stands.
    """
    if "supports" not in stances:
        return "contradicted" if "contradicts" in stances else "unsupported"
    if verdict == "unsupported":
        return "partial"
    return verdict


# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------


async def judge_support(
    llm: BaseLLMProvider,
    *,
    question: str,
    statements: Sequence[AnswerStatement],
    config: EvalConfig,
    run_id: str | None = None,
) -> SupportResult:
    limit = max(0, config.max_statements)
    result = SupportResult(truncated=max(0, len(statements) - limit))

    pending: list[JudgedStatement] = []
    for statement in statements[:limit]:
        judged = JudgedStatement(
            text=statement.text,
            citations=[
                JudgedCitation(
                    label=c.label,
                    url=c.url,
                    title=c.title,
                    status=c.status,
                    reason=c.reason,
                    excerpt=best_excerpt(
                        " ".join(c.evidence), statement.text, max_chars=config.judge_excerpt_chars
                    )
                    if c.status == "resolved" and c.evidence
                    else "",
                )
                for c in statement.citations
            ],
        )
        if any(c.excerpt for c in judged.citations):
            pending.append(judged)
        else:
            judged.verdict = "unsupported"
            judged.reason = (
                "No citation." if not judged.citations else "No cited source could be read."
            )
        result.statements.append(judged)

    size = max(1, config.judge_batch_size)
    for start in range(0, len(pending), size):
        await _judge_batch(
            llm, question, pending[start : start + size], config, run_id=run_id, result=result
        )
    return result


async def _judge_batch(
    llm: BaseLLMProvider,
    question: str,
    batch: list[JudgedStatement],
    config: EvalConfig,
    *,
    run_id: str | None,
    result: SupportResult,
) -> None:
    by_statement: dict[str, JudgedStatement] = {}
    by_evidence: dict[str, tuple[JudgedStatement, JudgedCitation]] = {}
    prompt_statements = []
    for statement in batch:
        label = f"A{len(by_statement) + 1}"
        by_statement[label] = statement
        evidence = []
        for citation in statement.citations:
            if not citation.excerpt:
                continue
            evidence_label = f"E{len(by_evidence) + 1}"
            by_evidence[evidence_label] = (statement, citation)
            evidence.append(
                JudgeEvidence(
                    label=evidence_label,
                    title=citation.title,
                    domain=domain_of(citation.url),
                    excerpt=citation.excerpt,
                )
            )
        prompt_statements.append(JudgeStatement(label=label, text=statement.text, evidence=evidence))

    try:
        response = await llm.generate_structured(
            build_support_prompt(question=question, statements=prompt_statements),
            SupportJudgement,
            system=SYSTEM_PROMPT,
            agent=AGENT,
            run_id=run_id,
            temperature=0.0,
            max_output_tokens=config.max_output_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - leave the batch unjudged
        result.errors.append(f"support judge failed: {type(exc).__name__}: {exc}")
        logger.warning("Support judge failed for run %s: %s", run_id, exc)
        return
    result.llm_calls.append(llm_call_record(response.usage, prompt_version=PROMPT_VERSION))
    if response.parsed is None:
        result.errors.append("support judge returned no parsed output")
        return

    for judgement in response.parsed.judgements:
        target = by_statement.get(judgement.statement.strip().strip("[]").upper())
        if target is None or target.verdict is not None:
            continue
        for item in judgement.evidence:
            entry = by_evidence.get(item.evidence.strip().strip("[]").upper())
            if entry is not None and entry[0] is target:
                entry[1].stance = item.stance
        # An excerpt the judge did not mention counts as neutral.
        for citation in target.citations:
            if citation.excerpt and citation.stance is None:
                citation.stance = "neutral"
        stances = [c.stance for c in target.citations if c.stance is not None]
        target.verdict = reconcile_verdict(judgement.verdict, stances)
        target.judged_by = "judge"
        target.reason = " ".join(judgement.reason.split())

    skipped = [label for label, s in by_statement.items() if s.verdict is None]
    if skipped:
        result.errors.append(f"support judge skipped {len(skipped)} sentence(s)")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


async def judge_coverage(
    llm: BaseLLMProvider,
    *,
    question: str,
    topics: Sequence[str],
    statements: Sequence[str],
    config: EvalConfig,
    run_id: str | None = None,
) -> CoverageResult:
    texts = [t for t in statements if t.strip()][: max(0, config.max_statements)]
    result = CoverageResult(topics=list(topics), statements=len(texts))
    if not topics or not texts:
        return result

    topic_labels = {f"T{i}": topic for i, topic in enumerate(topics, start=1)}
    statement_labels = {f"A{i}": text for i, text in enumerate(texts, start=1)}
    try:
        response = await llm.generate_structured(
            build_coverage_prompt(
                question=question,
                topics=list(topic_labels.items()),
                statements=[(f"A{i}", text) for i, text in enumerate(texts, start=1)],
            ),
            CoverageJudgement,
            system=COVERAGE_SYSTEM_PROMPT,
            agent=AGENT,
            run_id=run_id,
            temperature=0.0,
            max_output_tokens=config.max_output_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - leave the coverage unjudged
        result.error = f"coverage judge failed: {type(exc).__name__}: {exc}"
        logger.warning("Coverage judge failed for run %s: %s", run_id, exc)
        return result
    result.llm_calls.append(llm_call_record(response.usage, prompt_version=COVERAGE_PROMPT_VERSION))
    if response.parsed is None:
        result.error = "coverage judge returned no parsed output"
        return result

    # A topic the judge did not mention counts as not covered.
    covered = {
        label
        for item in response.parsed.topics
        if item.covered and (label := item.topic.strip().strip("[]").upper()) in topic_labels
    }
    result.covered = [topic for label, topic in topic_labels.items() if label in covered]
    off_topic = {
        label
        for raw in response.parsed.off_topic
        if (label := raw.strip().strip("[]").upper()) in statement_labels
    }
    result.off_topic = len(off_topic)
    result.off_topic_sentences = [text for label, text in statement_labels.items() if label in off_topic]
    result.judged = True
    return result
