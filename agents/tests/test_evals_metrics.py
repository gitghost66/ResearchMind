"""
M7: the metrics are documented formulas, so they get exact unit tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from researchmind_agents.evals.judge import JudgedCitation, JudgedStatement, reconcile_verdict
from researchmind_agents.evals.metrics import (
    answer_metrics,
    compare,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
    summarize_rows,
)


def sentence(verdict: str | None, *citations: tuple[str, str | None]) -> JudgedStatement:
    """A judged sentence; each citation is (status, stance)."""
    return JudgedStatement(
        text="A sentence.",
        verdict=verdict,
        judged_by="judge",
        citations=[
            JudgedCitation(label=f"S{i}", status=status, stance=stance)
            for i, (status, stance) in enumerate(citations, start=1)
        ],
    )


class TestRetrieval:
    RETRIEVED = [
        "https://a.example/1",
        "https://x.example/",
        "https://b.example/doc",
        "https://a.example/2",
        "https://y.example/",
    ]
    GOLD = ["a.example", "b.example", "c.example"]

    def test_known_ranking(self) -> None:
        m = retrieval_metrics(self.RETRIEVED, self.GOLD, (2, 5))
        assert m is not None
        assert m["p@2"] == pytest.approx(1 / 2)
        assert m["r@2"] == pytest.approx(1 / 3)
        assert m["p@5"] == pytest.approx(3 / 5)
        # Two pages of a.example match one gold source: recall counts it once.
        assert m["r@5"] == pytest.approx(2 / 3)
        assert m["mrr"] == 1.0
        assert m["gold_found"] == ["a.example", "b.example"]
        assert (m["retrieved"], m["gold"]) == (5, 3)

    def test_precision_divides_by_k_even_with_fewer_results(self) -> None:
        assert precision_at_k(["a.example"], 5) == pytest.approx(0.2)
        assert precision_at_k([], 5) == 0.0
        assert precision_at_k(["a.example"], 0) == 0.0

    def test_recall_needs_gold(self) -> None:
        assert recall_at_k(["g"], 0, 5) == 0.0

    def test_reciprocal_rank_of_the_first_relevant_source(self) -> None:
        assert reciprocal_rank([None, None, "g", "g"]) == pytest.approx(1 / 3)
        assert reciprocal_rank([None, None]) == 0.0

    def test_no_gold_means_no_retrieval_metrics(self) -> None:
        assert retrieval_metrics(self.RETRIEVED, [], (5,)) is None

    def test_nothing_retrieved_scores_zero(self) -> None:
        m = retrieval_metrics([], self.GOLD, (5,))
        assert m is not None and m["p@5"] == m["r@5"] == m["mrr"] == 0.0


class TestAnswerMetrics:
    def test_each_formula(self) -> None:
        m = answer_metrics(
            [
                sentence("supported", ("resolved", "supports"), ("resolved", "neutral")),
                sentence("partial", ("resolved", "supports")),
                sentence("contradicted", ("resolved", "contradicts")),
                sentence("unsupported", ("missing", None)),  # cites only a made-up source
                sentence("unsupported"),  # uncited
            ]
        )
        assert m is not None
        assert m["faithfulness"] == pytest.approx(1 / 5)
        assert m["partial_rate"] == pytest.approx(1 / 5)
        # 5 citations, 2 of them support their sentence.
        assert m["citation_precision"] == pytest.approx(2 / 5)
        assert m["fabricated_citation_rate"] == pytest.approx(1 / 5)
        # The contradicted sentence and the one resting on a made-up source.
        assert m["hallucination_rate"] == pytest.approx(2 / 5)
        assert (m["uncited"], m["citations_missing"], m["judged"]) == (1, 1, 5)

    def test_unjudged_sentences_are_left_out(self) -> None:
        m = answer_metrics(
            [sentence(None, ("resolved", None)), sentence("supported", ("resolved", "supports"))]
        )
        assert m is not None
        assert (m["statements"], m["judged"], m["citations"]) == (2, 1, 1)
        assert m["faithfulness"] == 1.0

    def test_nothing_judged_means_no_metrics(self) -> None:
        assert answer_metrics([sentence(None, ("resolved", None))]) is None
        assert answer_metrics([]) is None

    def test_citation_metrics_are_undefined_without_citations(self) -> None:
        m = answer_metrics([sentence("unsupported")])
        assert m is not None
        assert m["citation_precision"] is None and m["fabricated_citation_rate"] is None
        assert m["faithfulness"] == 0.0 and m["hallucination_rate"] == 0.0

    def test_an_unreadable_source_is_not_fabricated(self) -> None:
        m = answer_metrics([sentence("unsupported", ("unreachable", None))])
        assert m is not None
        assert m["fabricated_citation_rate"] == 0.0 and m["hallucination_rate"] == 0.0
        assert m["citations_unreachable"] == 1


class TestReconcile:
    @pytest.mark.parametrize(
        "verdict, stances, expected",
        [
            ("supported", ["neutral"], "unsupported"),
            ("supported", [], "unsupported"),
            ("supported", ["contradicts", "neutral"], "contradicted"),
            ("partial", ["contradicts"], "contradicted"),
            ("unsupported", ["supports"], "partial"),
            ("partial", ["supports", "neutral"], "partial"),
            ("supported", ["supports", "contradicts"], "supported"),
            ("contradicted", ["supports", "contradicts"], "contradicted"),
        ],
    )
    def test_verdict_agrees_with_stances(self, verdict: str, stances: list[str], expected: str) -> None:
        assert reconcile_verdict(verdict, stances) == expected


def row(
    question: str,
    system: str,
    faithfulness: float | None,
    *,
    completed: bool = True,
    hallucination: float = 0.0,
    seconds: float = 1.0,
    tokens: int = 100,
    mode: str = "quick",
    retrieval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "question_id": question,
        "system": system,
        "mode": mode,
        "completed": completed,
        "answer": None
        if faithfulness is None
        else {"faithfulness": faithfulness, "citation_precision": faithfulness, "hallucination_rate": hallucination},
        "coverage": {"completeness": 0.5, "relevance": 1.0},
        "retrieval": retrieval,
        "ops": {"seconds": seconds, "tokens": tokens, "cost_usd": 0.01, "iterations": 2},
    }


class TestAggregation:
    def test_quality_means_cover_completed_answers_only(self) -> None:
        summary = summarize_rows(
            [row("q1", "rag", 0.8), row("q2", "rag", 0.4), row("q3", "rag", 0.0, completed=False)],
            ks=(5,),
        )
        assert summary["questions"] == 3 and summary["completed"] == 2
        assert summary["task_completion"] == pytest.approx(2 / 3)
        assert summary["faithfulness"] == {"mean": pytest.approx(0.6), "n": 2}
        assert summary["p@5"] == {"mean": None, "n": 0}

    def test_operations_cover_every_run(self) -> None:
        summary = summarize_rows(
            [
                row("q1", "rag", 0.8, seconds=1.0, tokens=100),
                row("q2", "rag", None, completed=False, seconds=9.0, tokens=300),
                row("q3", "rag", 0.5, seconds=2.0, tokens=200),
            ],
            ks=(5,),
        )
        assert summary["latency_p50_s"] == 2.0
        assert summary["latency_mean_s"] == pytest.approx(4.0)
        assert summary["tokens_mean"] == pytest.approx(200.0)
        assert summary["iterations"] == 2.0

    def test_retrieval_means(self) -> None:
        summary = summarize_rows(
            [row("q1", "rag", 1.0, retrieval={"p@5": 0.4, "r@5": 1.0, "mrr": 0.5}),
             row("q2", "rag", 1.0, retrieval={"p@5": 0.2, "r@5": 0.5, "mrr": 1.0})],
            ks=(5,),
        )
        assert summary["p@5"]["mean"] == pytest.approx(0.3)
        assert summary["mrr"]["mean"] == pytest.approx(0.75)

    def test_verdict_uses_only_questions_every_system_completed(self) -> None:
        rows = [
            row("q1", "researchmind", 0.9), row("q1", "rag", 0.5), row("q1", "baseline_llm", 0.2),
            # The RAG baseline failed q2, where ResearchMind did badly: q2 is left out.
            row("q2", "researchmind", 0.1), row("q2", "rag", 0.0, completed=False),
            row("q2", "baseline_llm", 0.9),
        ]
        verdict = compare(rows, "faithfulness", target="researchmind", baselines=["baseline_llm", "rag"])
        assert verdict is not None
        assert verdict["questions"] == 1 and verdict["beats"] is True
        assert verdict["means"] == {"researchmind": 0.9, "baseline_llm": 0.2, "rag": 0.5}

    def test_lower_is_better_for_hallucination(self) -> None:
        rows = [
            row("q1", "researchmind", 0.9, hallucination=0.0),
            row("q1", "rag", 0.5, hallucination=0.1),
        ]
        verdict = compare(rows, "hallucination_rate", target="researchmind", baselines=["rag"])
        assert verdict is not None and verdict["beats"] is True

    def test_a_tie_does_not_beat(self) -> None:
        rows = [row("q1", "researchmind", 0.5), row("q1", "rag", 0.5)]
        verdict = compare(rows, "faithfulness", target="researchmind", baselines=["rag"])
        assert verdict is not None and verdict["beats"] is False

    def test_no_common_question(self) -> None:
        rows = [row("q1", "researchmind", 0.5), row("q2", "rag", 0.5)]
        verdict = compare(rows, "faithfulness", target="researchmind", baselines=["rag"])
        assert verdict == {"metric": "faithfulness", "questions": 0, "means": {}, "beats": None}
        assert compare(rows, "faithfulness", target="researchmind", baselines=[]) is None
