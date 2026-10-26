"""
M3 tests — graph state and schemas.

The state must stay JSON-only so any checkpointer (including LangGraph's
strict msgpack mode) can serialise it, and the list fields written by parallel
branches must carry an append reducer.
"""

from __future__ import annotations

import json
import operator
import typing
from typing import Any

from researchmind_agents.state import (
    SUPPORTED_MODES,
    CitationItem,
    PassageItem,
    PlannedSubQuestion,
    PlanOutput,
    ReportOutput,
    ReportSection,
    ReportStatement,
    ResearchReport,
    ResearchState,
    SourceItem,
    SourceSelection,
    SubQuestionItem,
    new_research_state,
)

REDUCED_FIELDS = ("sources", "passages", "research_results", "events", "llm_calls")


class TestNewResearchState:
    def test_defaults(self) -> None:
        state = new_research_state(run_id="run-1", question="Is RAG better?")

        assert state["run_id"] == "run-1"
        assert state["question"] == "Is RAG better?"
        assert state["mode"] == "quick"
        assert state["user_context"] == {}
        assert state["iteration"] == 0
        assert state["max_iterations"] == 1
        assert state["plan"] == {}
        assert state["sub_questions"] == []
        for name in REDUCED_FIELDS:
            assert state[name] == []  # type: ignore[literal-required]
        assert state["report"] == {}
        assert state["report_markdown"] == ""

    def test_custom_values(self) -> None:
        context = {"audience": "CTO"}
        state = new_research_state(
            run_id="run-2",
            question="Q?",
            mode="quick",
            user_context=context,
            max_iterations=3,
        )

        assert state["user_context"] == {"audience": "CTO"}
        assert state["max_iterations"] == 3

    def test_user_context_is_copied(self) -> None:
        context: dict[str, Any] = {"a": 1}
        state = new_research_state(run_id="r", question="q", user_context=context)
        context["a"] = 2
        assert state["user_context"] == {"a": 1}

    def test_initial_state_is_json_serialisable(self) -> None:
        state = new_research_state(run_id="r", question="q", user_context={"k": [1, 2]})
        assert json.loads(json.dumps(state)) == state


class TestReducers:
    def test_parallel_list_fields_append(self) -> None:
        hints = typing.get_type_hints(ResearchState, include_extras=True)
        for name in REDUCED_FIELDS:
            hint = hints[name]
            assert typing.get_origin(hint) is typing.Annotated, name
            assert operator.add in typing.get_args(hint)[1:], name

    def test_single_writer_fields_have_no_reducer(self) -> None:
        hints = typing.get_type_hints(ResearchState, include_extras=True)
        for name in ("run_id", "question", "plan", "sub_questions", "report", "report_markdown"):
            assert typing.get_origin(hints[name]) is not typing.Annotated, name


class TestSchemasRoundTrip:
    def test_records_round_trip_through_json(self) -> None:
        records = [
            SubQuestionItem(id="sq-1", text="What is RAG?", queries=["rag"]),
            SourceItem(id="s-1", url="https://a.example", title="A", content_hash="h"),
            PassageItem(id="p-1", source_id="s-1", sub_question_id="sq-1", text="t", chunk_index=2),
            CitationItem(label="S1", source_id="s-1", url="https://a.example", passage_ids=["p-1"]),
        ]
        for record in records:
            dumped = record.model_dump(mode="json")
            assert json.loads(json.dumps(dumped)) == dumped
            assert type(record).model_validate(dumped) == record

    def test_report_round_trips(self) -> None:
        statement = ReportStatement(text="RAG is cheaper.", citations=["S1"])
        report = ResearchReport(
            title="T",
            question="Q",
            mode="quick",
            executive_summary=[statement],
            sections=[ReportSection(heading="H", statements=[statement])],
            limitations=["L"],
            sources=[CitationItem(label="S1", source_id="s-1", url="https://a.example")],
            coverage={"sub_questions": 2},
        )
        dumped = report.model_dump(mode="json")
        assert ResearchReport.model_validate(json.loads(json.dumps(dumped))) == report

    def test_sub_question_defaults(self) -> None:
        item = SubQuestionItem(id="x", text="t")
        assert item.queries == []
        assert item.domain == "web"
        assert item.status == "pending"


class TestReportDefaults:
    def test_confidence_is_unscored_and_generation_is_llm(self) -> None:
        report = ResearchReport(title="T", question="Q", mode="quick")
        assert report.overall_confidence is None
        assert report.generation == "llm"
        assert report.dropped_statements == 0
        assert report.sources == []


class TestLLMSchemas:
    def test_llm_output_fields_are_required(self) -> None:
        """Required-only schemas keep the provider's JSON schema unambiguous."""
        for schema in (
            PlannedSubQuestion,
            PlanOutput,
            SourceSelection,
            ReportStatement,
            ReportSection,
            ReportOutput,
        ):
            json_schema = schema.model_json_schema()
            assert set(json_schema.get("required", [])) == set(json_schema["properties"]), schema

    def test_plan_output_parses(self) -> None:
        plan = PlanOutput.model_validate_json(
            '{"objective": "o", "sub_questions": [{"text": "t", "queries": ["q"]}]}'
        )
        assert plan.sub_questions[0].queries == ["q"]


def test_supported_modes_is_quick_only() -> None:
    assert frozenset({"quick"}) == SUPPORTED_MODES
