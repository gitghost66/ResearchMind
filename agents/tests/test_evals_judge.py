"""
M7: the LLM-as-judge and the deterministic steps around it.

The judge is scripted here; what is tested is everything the harness does
with its answers: which sentences reach it, how labels map back, how
verdicts are reconciled, and how failures are contained.
"""

from __future__ import annotations

import json
from typing import Any

from researchmind_agents.config import EvalConfig
from researchmind_agents.evals.judge import judge_coverage, judge_support
from researchmind_agents.evals.systems import AnswerStatement, Citation
from researchmind_agents.prompts.eval_judge import COVERAGE_PROMPT_VERSION, PROMPT_VERSION
from researchmind_agents.testing import Responder, ScriptedLLM

OPEN = "<<<UNTRUSTED_DATA"


def judge(**responses: Responder | list[Responder]) -> ScriptedLLM:
    return ScriptedLLM(responses=responses)


def cite(label: str, *, status: str = "resolved", text: str = "Evidence text.") -> Citation:
    return Citation(
        label=label,
        url=f"https://{label.lower()}.example/page",
        title=f"Page {label}",
        status=status,  # type: ignore[arg-type]
        evidence=[text] if status == "resolved" else [],
    )


def say(text: str, *citations: Citation) -> AnswerStatement:
    return AnswerStatement(text=text, citations=list(citations))


def judgements(*items: tuple[str, str, list[tuple[str, str]]]) -> str:
    return json.dumps(
        {
            "judgements": [
                {
                    "statement": label,
                    "verdict": verdict,
                    "evidence": [{"evidence": e, "stance": s} for e, s in evidence],
                    "reason": "Because.",
                }
                for label, verdict, evidence in items
            ]
        }
    )


class TestSupport:
    async def test_sentences_with_no_readable_citation_never_reach_the_judge(self) -> None:
        llm = judge()
        result = await judge_support(
            llm,
            question="Q",
            statements=[
                say("Uncited."),
                say("Made up.", cite("S1", status="missing")),
                say("Blocked.", cite("S2", status="unreachable")),
            ],
            config=EvalConfig(),
        )
        assert llm.calls == []
        assert [s.verdict for s in result.statements] == ["unsupported"] * 3
        assert {s.judged_by for s in result.statements} == {"rule"}
        assert all(c.stance is None for s in result.statements for c in s.citations)

    async def test_stances_and_verdicts_map_back_and_are_reconciled(self) -> None:
        llm = judge(
            SupportJudgement=judgements(
                ("A1", "supported", [("E1", "supports"), ("E2", "neutral")]),
                # Called supported, but its only excerpt is neutral.
                ("A2", "supported", [("E3", "neutral")]),
            )
        )
        result = await judge_support(
            llm,
            question="Q",
            statements=[say("First.", cite("S1"), cite("S2")), say("Second.", cite("S3"))],
            config=EvalConfig(),
            run_id="run-1",
        )
        first, second = result.statements
        assert first.verdict == "supported" and first.judged_by == "judge"
        assert [c.stance for c in first.citations] == ["supports", "neutral"]
        assert second.verdict == "unsupported"
        assert result.errors == []
        assert result.llm_calls[0]["prompt_version"] == PROMPT_VERSION
        assert llm.calls[0]["schema"] == "SupportJudgement"

    async def test_the_judge_reads_the_relevant_excerpt_fenced_as_data(self) -> None:
        filler = "Unrelated background sentence number {}. "
        evidence = (
            "".join(filler.format(i) for i in range(20))
            + "Heat pumps keep a COP of 2.1 at minus fifteen degrees. "
            + "".join(filler.format(i) for i in range(20, 40))
        )
        llm = judge(SupportJudgement=judgements(("A1", "supported", [("E1", "supports")])))
        result = await judge_support(
            llm,
            question="Q",
            statements=[say("Heat pumps keep a COP of 2.1 at minus fifteen.", cite("S1", text=evidence))],
            config=EvalConfig(judge_excerpt_chars=160),
        )
        excerpt = result.statements[0].citations[0].excerpt
        assert "COP of 2.1" in excerpt and len(excerpt) <= 170
        prompt = llm.calls[0]["prompt"]
        assert excerpt in prompt and OPEN in prompt
        assert prompt.index(OPEN) < prompt.index("[A1] Heat pumps")

    async def test_sentences_are_judged_in_batches(self) -> None:
        llm = judge(
            SupportJudgement=lambda prompt: judgements(
                *(("A1", "supported", [("E1", "supports")]),)
                + ((("A2", "supported", [("E2", "supports")]),) if "[A2]" in prompt else ())
            )
        )
        result = await judge_support(
            llm,
            question="Q",
            statements=[say(f"Sentence {i}.", cite(f"S{i}")) for i in range(5)],
            config=EvalConfig(judge_batch_size=2),
        )
        assert len(llm.calls) == 3
        # Labels restart in every batch.
        assert all("[A1]" in call["prompt"] for call in llm.calls)
        assert [s.verdict for s in result.statements] == ["supported"] * 5

    async def test_a_failed_batch_leaves_its_sentences_unjudged(self) -> None:
        llm = judge(
            SupportJudgement=[
                RuntimeError("judge is down"),
                judgements(("A1", "supported", [("E1", "supports")])),
            ]
        )
        result = await judge_support(
            llm,
            question="Q",
            statements=[say("One.", cite("S1")), say("Two.", cite("S2"))],
            config=EvalConfig(judge_batch_size=1),
        )
        assert [s.verdict for s in result.statements] == [None, "supported"]
        assert len(result.errors) == 1 and "judge is down" in result.errors[0]

    async def test_a_sentence_the_judge_skips_stays_unjudged(self) -> None:
        llm = judge(SupportJudgement=judgements(("A1", "supported", [("E1", "supports")])))
        result = await judge_support(
            llm,
            question="Q",
            statements=[say("One.", cite("S1")), say("Two.", cite("S2"))],
            config=EvalConfig(),
        )
        assert [s.verdict for s in result.statements] == ["supported", None]
        assert result.errors == ["support judge skipped 1 sentence(s)"]

    async def test_an_unmentioned_excerpt_counts_as_neutral(self) -> None:
        llm = judge(SupportJudgement=judgements(("[a1]", "partial", [("e1", "supports")])))
        result = await judge_support(
            llm, question="Q", statements=[say("One.", cite("S1"), cite("S2"))], config=EvalConfig()
        )
        [statement] = result.statements
        assert statement.verdict == "partial"
        assert [c.stance for c in statement.citations] == ["supports", "neutral"]

    async def test_a_label_from_another_sentence_is_ignored(self) -> None:
        llm = judge(
            SupportJudgement=judgements(
                ("A1", "supported", [("E2", "supports")]), ("A2", "unsupported", [("E2", "neutral")])
            )
        )
        result = await judge_support(
            llm,
            question="Q",
            statements=[say("One.", cite("S1")), say("Two.", cite("S2"))],
            config=EvalConfig(),
        )
        first, second = result.statements
        # E2 belongs to A2, so A1's only excerpt counts as neutral.
        assert first.citations[0].stance == "neutral" and first.verdict == "unsupported"
        assert second.citations[0].stance == "neutral"

    async def test_only_the_first_max_statements_are_judged(self) -> None:
        result = await judge_support(
            judge(),
            question="Q",
            statements=[say("One."), say("Two."), say("Three.")],
            config=EvalConfig(max_statements=2),
        )
        assert len(result.statements) == 2 and result.truncated == 1

    async def test_round_trip_through_json(self) -> None:
        llm = judge(SupportJudgement=judgements(("A1", "supported", [("E1", "supports")])))
        result = await judge_support(
            llm, question="Q", statements=[say("One.", cite("S1"))], config=EvalConfig()
        )
        from researchmind_agents.evals.judge import JudgedStatement

        data: dict[str, Any] = json.loads(json.dumps(result.statements[0].as_dict()))
        assert JudgedStatement.from_dict(data) == result.statements[0]


def coverage_reply(covered: dict[str, bool], off_topic: list[str]) -> str:
    return json.dumps(
        {"topics": [{"topic": t, "covered": c} for t, c in covered.items()], "off_topic": off_topic}
    )


class TestCoverage:
    async def test_covered_topics_and_off_topic_sentences(self) -> None:
        llm = judge(CoverageJudgement=coverage_reply({"T1": True, "[t2]": False}, ["A2", "A9"]))
        result = await judge_coverage(
            llm,
            question="Q",
            topics=["recall", "memory use"],
            statements=["One.", "Two.", "Three."],
            config=EvalConfig(),
        )
        assert result.covered == ["recall"]
        assert result.completeness == 0.5
        # A9 is not a sentence of this answer.
        assert result.off_topic == 1 and result.off_topic_sentences == ["Two."]
        assert result.relevance == 1 - 1 / 3
        assert result.llm_calls[0]["prompt_version"] == COVERAGE_PROMPT_VERSION
        prompt = llm.calls[0]["prompt"]
        assert "[T2] memory use" in prompt and OPEN in prompt

    def test_off_topic_is_judged_against_the_question_not_the_topics(self) -> None:
        # eval_coverage/v1 let the judge flag sentences on aspects the topic
        # list does not name, and recommendations; the first live run caught it.
        from researchmind_agents.prompts.eval_judge import build_coverage_prompt

        prompt = build_coverage_prompt(question="Q?", topics=[("T1", "a")], statements=[("A1", "b")])
        assert "against the question only, not\n   against the topic list" in prompt
        assert "a recommendation that\n   answers the question is on-topic" in prompt

    async def test_an_unmentioned_topic_is_not_covered(self) -> None:
        llm = judge(CoverageJudgement=coverage_reply({"T1": True}, []))
        result = await judge_coverage(
            llm, question="Q", topics=["a", "b"], statements=["One."], config=EvalConfig()
        )
        assert result.covered == ["a"] and result.completeness == 0.5

    async def test_nothing_to_judge_makes_no_call(self) -> None:
        llm = judge()
        no_topics = await judge_coverage(llm, question="Q", topics=[], statements=["One."], config=EvalConfig())
        no_answer = await judge_coverage(llm, question="Q", topics=["a"], statements=[" "], config=EvalConfig())
        assert llm.calls == []
        # Nothing judged the sentences, so relevance is unknown, not 100%.
        assert no_topics.completeness is None and no_topics.relevance is None
        assert no_answer.completeness == 0.0 and no_answer.relevance is None

    async def test_a_judge_failure_leaves_coverage_undefined(self) -> None:
        llm = judge(CoverageJudgement=RuntimeError("down"))
        result = await judge_coverage(
            llm, question="Q", topics=["a"], statements=["One."], config=EvalConfig()
        )
        assert result.completeness is None and result.relevance is None
        assert result.error is not None and "down" in result.error
