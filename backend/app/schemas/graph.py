"""
Evidence graph payload for ``GET /api/v1/research/{run_id}/graph`` (PRD s9).

Shaped for a React Flow UI: ``{nodes[], edges[]}``. Every node is
``{id, type, label, meta}`` and each type has its own ``meta`` schema, so a
claim node always carries its ``verdict`` and ``confidence``. Every edge is
``{id, from, to, type, meta}``:

=============  ==========================  ======================================
Edge type      From -> to                  Meaning
=============  ==========================  ======================================
decomposes     question -> sub_question    the plan's top-level sub-questions
               sub_question -> sub_question  a gap-filling follow-up of its parent
               sub_question -> claim       a claim that answers the sub-question
               (question -> claim)         a claim with no sub-question
supports       passage -> claim            the fact checker found it supports
contradicts    passage -> claim            the fact checker found it contradicts
               claim -> claim              a detected contradiction (meta: cause)
cites          passage -> source           the source the passage is from
=============  ==========================  ======================================

Node IDs are ``<type>:<record id>``, so they are unique across types. The
graph validates itself: IDs are unique and every edge joins two nodes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "GraphNodeType",
    "GraphEdgeType",
    "QuestionMeta",
    "SubQuestionMeta",
    "ClaimMeta",
    "SourceMeta",
    "PassageMeta",
    "QuestionNode",
    "SubQuestionNode",
    "ClaimNode",
    "SourceNode",
    "PassageNode",
    "GraphNode",
    "GraphEdge",
    "EvidenceGraph",
]

GraphNodeType = Literal["question", "sub_question", "claim", "source", "passage"]
GraphEdgeType = Literal["decomposes", "supports", "contradicts", "cites"]


class QuestionMeta(BaseModel):
    question: str
    mode: str
    status: str
    # 0-based index of the latest research round.
    iteration: int


class SubQuestionMeta(BaseModel):
    text: str
    domain: str
    status: str
    # Research round that added it (0: the first plan).
    iteration: int = 0
    # Gap-filling follow-ups: the sub-question whose gap it fills.
    parent_id: str | None = None


class ClaimMeta(BaseModel):
    text: str
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    sub_question_id: str | None = None


class SourceMeta(BaseModel):
    url: str
    title: str
    domain_type: str
    reliability_score: float
    published_at: datetime | None = None
    cited: bool = Field(description="Whether the latest report cites this source.")


class PassageMeta(BaseModel):
    source_id: str
    chunk_index: int
    excerpt: str


class _Node(BaseModel):
    id: str
    label: str


class QuestionNode(_Node):
    type: Literal["question"] = "question"
    meta: QuestionMeta


class SubQuestionNode(_Node):
    type: Literal["sub_question"] = "sub_question"
    meta: SubQuestionMeta


class ClaimNode(_Node):
    type: Literal["claim"] = "claim"
    meta: ClaimMeta


class SourceNode(_Node):
    type: Literal["source"] = "source"
    meta: SourceMeta


class PassageNode(_Node):
    type: Literal["passage"] = "passage"
    meta: PassageMeta


GraphNode = Annotated[
    QuestionNode | SubQuestionNode | ClaimNode | SourceNode | PassageNode,
    Field(discriminator="type"),
]


class GraphEdge(BaseModel):
    """``from`` is a Python keyword, so the field is ``from_`` with alias ``from``."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    from_: str = Field(alias="from")
    to: str
    type: GraphEdgeType
    # Contradiction edges: contradiction_id, likely_cause, resolved, explanation.
    meta: dict[str, Any] = Field(default_factory=dict)


class EvidenceGraph(BaseModel):
    run_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    @model_validator(mode="after")
    def _connected(self) -> EvidenceGraph:
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.from_ not in known or edge.to not in known:
                raise ValueError(f"edge {edge.id!r} joins an unknown node")
        edge_ids = [edge.id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("edge ids must be unique")
        return self
