"""
Metrics for the M7 comparison (PRD s11). Pure functions; no model calls.

**Retrieval** — for systems that retrieve, on questions with gold sources.
Each system's stored sources are ranked by their best passage's similarity to
the question (``systems.rank_sources``). A source is *relevant* when it
matches a gold source (``benchmark.gold_match``).

* Precision@K = relevant sources in the top K / K.
* Recall@K = gold sources matched within the top K / gold sources.
* MRR = mean of 1 / rank of the first relevant source (0 when none).

**Answer quality** — per answer, from the judged sentences, then averaged
over answers (each question weighs the same, however long its answer):

* Faithfulness = sentences fully supported by their own cited evidence /
  sentences judged. An uncited sentence is not supported.
* Citation precision = citations whose excerpt supports their sentence /
  citations. Undefined for an answer with no citations.
* Fabricated citations = citations to sources that do not exist (unknown
  label, not a URL, unresolvable host, HTTP 404/410) / citations.
* Hallucination rate = sentences contradicted by their own cited evidence, or
  citing only sources that do not exist / sentences judged. Stricter than
  "not faithful": the answer misrepresents its evidence.
* Completeness = expected sub-topics covered / expected sub-topics.
* Relevance = sentences not off-topic / sentences.

**Operations** — per run: task completion (no error, at least one sentence,
and the system's model wrote the answer — an extractive fallback quotes its
evidence and would inflate faithfulness), research rounds, seconds, tokens,
and cost of the system's own model calls (never the judge's).

Quality and retrieval means are over completed answers; the table shows how
many. The head-to-head verdict (:func:`compare`) uses only the questions
every system completed, so a system cannot win by failing the hard ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import mean, median
from typing import Any

from researchmind_agents.evals.benchmark import matching_gold
from researchmind_agents.evals.judge import JudgedStatement

__all__ = [
    "ANSWER_METRICS",
    "COVERAGE_METRICS",
    "LOWER_IS_BETTER",
    "HEADLINE_METRICS",
    "retrieval_metric_names",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "retrieval_metrics",
    "answer_metrics",
    "metric_value",
    "summarize_rows",
    "compare",
]

ANSWER_METRICS: tuple[str, ...] = (
    "faithfulness",
    "citation_precision",
    "fabricated_citation_rate",
    "hallucination_rate",
)
COVERAGE_METRICS: tuple[str, ...] = ("completeness", "relevance")
LOWER_IS_BETTER: frozenset[str] = frozenset({"fabricated_citation_rate", "hallucination_rate"})
# The M7 bar: ResearchMind must beat both baselines on these.
HEADLINE_METRICS: tuple[str, ...] = ("faithfulness", "citation_precision")


def retrieval_metric_names(ks: Iterable[int]) -> tuple[str, ...]:
    names: list[str] = []
    for k in ks:
        names += [f"p@{k}", f"r@{k}"]
    return (*names, "mrr")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def precision_at_k(matches: Sequence[str | None], k: int) -> float:
    """*matches* holds, per rank, the gold source matched (or ``None``)."""
    if k <= 0:
        return 0.0
    return sum(1 for m in matches[:k] if m is not None) / k


def recall_at_k(matches: Sequence[str | None], gold_count: int, k: int) -> float:
    if gold_count <= 0:
        return 0.0
    found = {m for m in matches[: max(0, k)] if m is not None}
    return len(found) / gold_count


def reciprocal_rank(matches: Sequence[str | None]) -> float:
    for rank, match in enumerate(matches, start=1):
        if match is not None:
            return 1.0 / rank
    return 0.0


def retrieval_metrics(
    retrieved: Sequence[str], gold_sources: Sequence[str], ks: Sequence[int]
) -> dict[str, Any] | None:
    """P@K, R@K, and reciprocal rank for one ranked list; ``None`` without gold sources."""
    gold = list(dict.fromkeys(gold_sources))
    if not gold:
        return None
    matches = [matching_gold(url, gold) for url in retrieved]
    result: dict[str, Any] = {}
    for k in ks:
        result[f"p@{k}"] = precision_at_k(matches, k)
        result[f"r@{k}"] = recall_at_k(matches, len(gold), k)
    result["mrr"] = reciprocal_rank(matches)
    result["retrieved"] = len(retrieved)
    result["gold"] = len(gold)
    result["gold_found"] = [g for g in gold if g in set(matches)]
    return result


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def _hallucinated(statement: JudgedStatement) -> bool:
    if statement.verdict == "contradicted":
        return True
    return bool(statement.citations) and all(c.status == "missing" for c in statement.citations)


def answer_metrics(statements: Sequence[JudgedStatement]) -> dict[str, Any] | None:
    """Support metrics for one answer; ``None`` when no sentence was judged."""
    judged = [s for s in statements if s.judged]
    if not judged:
        return None
    citations = [c for s in judged for c in s.citations]
    supported = sum(1 for s in judged if s.verdict == "supported")
    partial = sum(1 for s in judged if s.verdict == "partial")
    contradicted = sum(1 for s in judged if s.verdict == "contradicted")
    uncited = sum(1 for s in judged if not s.citations)
    precise = sum(1 for c in citations if c.stance == "supports")
    missing = sum(1 for c in citations if c.status == "missing")
    unreachable = sum(1 for c in citations if c.status == "unreachable")
    return {
        "faithfulness": supported / len(judged),
        "citation_precision": precise / len(citations) if citations else None,
        "fabricated_citation_rate": missing / len(citations) if citations else None,
        "hallucination_rate": sum(1 for s in judged if _hallucinated(s)) / len(judged),
        "partial_rate": partial / len(judged),
        "statements": len(statements),
        "judged": len(judged),
        "supported": supported,
        "partial": partial,
        "contradicted": contradicted,
        "uncited": uncited,
        "citations": len(citations),
        "citations_supporting": precise,
        "citations_missing": missing,
        "citations_unreachable": unreachable,
    }


# ---------------------------------------------------------------------------
# Aggregation over result rows
# ---------------------------------------------------------------------------


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    """One metric from a result row (``None`` when it does not apply)."""
    if metric in ANSWER_METRICS:
        section = row.get("answer")
    elif metric in COVERAGE_METRICS:
        section = row.get("coverage")
    else:
        section = row.get("retrieval")
    value = (section or {}).get(metric)
    return None if value is None else float(value)


def _mean_of(rows: Iterable[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [v for row in rows if (v := metric_value(row, metric)) is not None]
    return {"mean": mean(values) if values else None, "n": len(values)}


def summarize_rows(rows: Sequence[dict[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    """Means for one system's rows (one per question)."""
    completed = [row for row in rows if row.get("completed")]
    seconds = [float(row["ops"]["seconds"]) for row in rows if row.get("ops")]
    summary: dict[str, Any] = {
        "questions": len(rows),
        "completed": len(completed),
        "task_completion": len(completed) / len(rows) if rows else None,
    }
    for metric in (*ANSWER_METRICS, *COVERAGE_METRICS, *retrieval_metric_names(ks)):
        summary[metric] = _mean_of(completed, metric)
    iterations = [float(row["ops"]["iterations"]) for row in completed]
    tokens = [float(row["ops"]["tokens"]) for row in rows if row.get("ops")]
    cost = [float(row["ops"]["cost_usd"]) for row in rows if row.get("ops")]
    summary["iterations"] = mean(iterations) if iterations else None
    summary["latency_p50_s"] = median(seconds) if seconds else None
    summary["latency_mean_s"] = mean(seconds) if seconds else None
    summary["tokens_mean"] = mean(tokens) if tokens else None
    summary["cost_mean_usd"] = mean(cost) if cost else None
    return summary


def compare(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    target: str,
    baselines: Sequence[str],
) -> dict[str, Any] | None:
    """
    Paired comparison of *target* against *baselines* on *metric*, over the
    questions where every one of them completed with a value. ``beats`` is true
    when the target's mean is strictly better than each baseline's.
    """
    if not baselines:
        return None
    systems = [target, *baselines]
    values: dict[str, dict[str, float]] = {}
    for row in rows:
        value = metric_value(row, metric)
        if row.get("completed") and value is not None:
            values.setdefault(row["question_id"], {})[row["system"]] = value
    common = sorted(q for q, by_system in values.items() if all(s in by_system for s in systems))
    if not common:
        return {"metric": metric, "questions": 0, "means": {}, "beats": None}
    means = {s: mean(values[q][s] for q in common) for s in systems}
    lower = metric in LOWER_IS_BETTER
    beats = all(
        (means[target] < means[b]) if lower else (means[target] > means[b]) for b in baselines
    )
    return {"metric": metric, "questions": len(common), "means": means, "beats": beats}
