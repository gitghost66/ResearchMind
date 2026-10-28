"""
M6 tests: the evidence graph endpoint, ``GET /api/v1/research/{run_id}/graph``.

M6 acceptance: the graph payload validates against its schema. The payload
is parsed back into :class:`EvidenceGraph` (typed nodes, edges that join
known nodes), and its structure is checked against what the run stored.
"""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any

from sqlalchemy import select

from app.models.claims import Claim, ClaimEvidence, Contradiction, ContradictionClaim
from app.models.research import ResearchRun, SubQuestion
from app.schemas.graph import EvidenceGraph
from app.services.evidence_graph import GraphData, build_evidence_graph

from .conftest import ApiHarness

BASE = "/api/v1/research"


async def get_graph(api: ApiHarness, run_id: str) -> tuple[dict[str, Any], EvidenceGraph]:
    response = await api.client.get(f"{BASE}/{run_id}/graph")
    assert response.status_code == 200, response.text
    body = response.json()
    return body, EvidenceGraph.model_validate(body)


def by_type(body: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [n for n in body["nodes"] if n["type"] == kind]


def edges_of(body: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in body["edges"] if e["type"] == kind]


async def add_contradiction(api: ApiHarness, run_id: str, cause: str) -> tuple[str, list[str]]:
    async with api.session_factory() as session:
        claims = (
            (await session.execute(select(Claim).where(Claim.run_id == uuid.UUID(run_id)).order_by(Claim.created_at)))
            .scalars()
            .all()
        )
        assert len(claims) >= 2
        row = Contradiction(run_id=uuid.UUID(run_id), explanation="They disagree.", likely_cause=cause)
        session.add(row)
        await session.flush()
        for claim in claims[:2]:
            session.add(ContradictionClaim(contradiction_id=row.id, claim_id=claim.id))
        await session.commit()
        return str(row.id), [str(c.id) for c in claims[:2]]


class TestDeepRunGraph:
    async def test_payload_validates_and_mirrors_the_stored_evidence(self, api: ApiHarness) -> None:
        run_id = await api.create(mode="deep")
        await api.wait_for(run_id, "completed")
        contradiction_id, pair = await add_contradiction(api, run_id, "unexplained")

        body, graph = await get_graph(api, run_id)

        assert graph.run_id == run_id
        assert Counter(n["type"] for n in body["nodes"]).keys() == {
            "question", "sub_question", "claim", "passage", "source"
        }
        # JSON uses the PRD's "from" key.
        assert all({"id", "from", "to", "type"} <= set(e) for e in body["edges"])

        async with api.session_factory() as session:
            sub_questions = (await session.execute(select(SubQuestion))).scalars().all()
            claims = (await session.execute(select(Claim))).scalars().all()
            links = (await session.execute(select(ClaimEvidence))).scalars().all()

        [question] = by_type(body, "question")
        assert question["id"] == f"question:{run_id}"
        assert question["meta"]["mode"] == "deep" and question["meta"]["status"] == "completed"
        decomposes = {(e["from"], e["to"]) for e in edges_of(body, "decomposes")}
        for sq in sub_questions:
            assert (question["id"], f"sub_question:{sq.id}") in decomposes

        claim_nodes = {n["id"]: n for n in by_type(body, "claim")}
        assert set(claim_nodes) == {f"claim:{c.id}" for c in claims}
        for claim in claims:
            meta = claim_nodes[f"claim:{claim.id}"]["meta"]
            assert meta["verdict"] == claim.verdict
            assert meta["confidence"] == claim.confidence
            assert (f"sub_question:{claim.sub_question_id}", f"claim:{claim.id}") in decomposes

        evidence = {(f"passage:{x.passage_id}", f"claim:{x.claim_id}", x.stance) for x in links}
        drawn = {(e["from"], e["to"], e["type"]) for e in body["edges"] if e["type"] in ("supports", "contradicts")}
        assert {x for x in evidence if x[2] != "neutral"} <= drawn
        # Every passage cites its source.
        cites = {e["from"] for e in edges_of(body, "cites")}
        assert {n["id"] for n in by_type(body, "passage")} == cites

        [conflict] = [e for e in edges_of(body, "contradicts") if e["from"].startswith("claim:")]
        assert {conflict["from"], conflict["to"]} == {f"claim:{cid}" for cid in pair}
        assert conflict["meta"]["contradiction_id"] == contradiction_id
        assert conflict["meta"]["likely_cause"] == "unexplained"
        assert conflict["meta"]["resolved"] is False

    async def test_resolution_follows_the_scoring_setting(self, make_api: Any) -> None:
        api = await make_api(confidence_resolved_causes="unexplained")
        run_id = await api.create(mode="deep")
        await api.wait_for(run_id, "completed")
        await add_contradiction(api, run_id, "unexplained")
        body, _ = await get_graph(api, run_id)
        [conflict] = [e for e in edges_of(body, "contradicts") if e["from"].startswith("claim:")]
        assert conflict["meta"]["resolved"] is True

    async def test_sources_carry_reliability_and_citation(self, api: ApiHarness) -> None:
        run_id = await api.create(mode="deep")
        await api.wait_for(run_id, "completed")
        body, _ = await get_graph(api, run_id)
        report = (await api.client.get(f"{BASE}/{run_id}/report")).json()
        cited = {s["source_id"] for s in report["content"]["sources"]}
        sources = by_type(body, "source")
        assert sources
        for node in sources:
            assert 0.0 <= node["meta"]["reliability_score"] <= 1.0
            assert node["meta"]["cited"] == (node["id"].removeprefix("source:") in cited)


class TestOtherGraphs:
    async def test_quick_run_graph_has_the_cited_passages(self, api: ApiHarness) -> None:
        run_id = await api.create()
        await api.wait_for(run_id, "completed")
        body, _ = await get_graph(api, run_id)

        assert by_type(body, "claim") == []
        report = (await api.client.get(f"{BASE}/{run_id}/report")).json()
        cited = report["content"]["sources"]
        assert {n["id"] for n in by_type(body, "source")} == {f"source:{s['source_id']}" for s in cited}
        passages = {f"passage:{pid}" for s in cited for pid in s["passage_ids"]}
        assert {n["id"] for n in by_type(body, "passage")} == passages
        assert all(n["meta"]["cited"] for n in by_type(body, "source"))

    async def test_a_running_run_has_a_partial_graph(self, api: ApiHarness) -> None:
        api.gate.clear()
        run_id = await api.create(mode="deep")
        await api.until(api.waiting.is_set)
        body, _ = await get_graph(api, run_id)
        assert [n["type"] for n in body["nodes"]] == ["question"]
        assert body["edges"] == []
        api.gate.set()
        await api.wait_for(run_id, "completed")

    async def test_unknown_run_is_404(self, api: ApiHarness) -> None:
        response = await api.client.get(f"{BASE}/{uuid.uuid4()}/graph")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "run_not_found"

    async def test_links_include_the_graph(self, api: ApiHarness) -> None:
        response = await api.client.post(BASE, json={"question": "RAG versus fine-tuning"})
        body = response.json()
        assert body["links"]["graph"] == f"{BASE}/{body['run_id']}/graph"
        await api.wait_for(body["run_id"])


class TestBuilder:
    def test_follow_ups_hang_off_their_parent(self) -> None:
        run = ResearchRun(
            id=uuid.uuid4(), question="Q?", mode="deep", status="completed", iteration=1, plan={}
        )
        root = SubQuestion(id=uuid.uuid4(), run_id=run.id, text="Root?", domain="web", status="completed")
        child = SubQuestion(id=uuid.uuid4(), run_id=run.id, text="Child?", domain="market", status="completed")
        run.plan = {
            "sub_questions": [
                {"id": str(root.id), "iteration": 0, "parent_id": None},
                {"id": str(child.id), "iteration": 1, "parent_id": str(root.id)},
            ]
        }
        claim = Claim(
            id=uuid.uuid4(), run_id=run.id, sub_question_id=None, text="Floating claim.",
            verdict="weak", confidence=0.3,
        )
        graph = build_evidence_graph(
            GraphData(run=run, sub_questions=[root, child], claims=[claim]), is_resolved=lambda _: True
        )
        edges = {(e.from_, e.to, e.type) for e in graph.edges}
        assert (f"question:{run.id}", f"sub_question:{root.id}", "decomposes") in edges
        assert (f"sub_question:{root.id}", f"sub_question:{child.id}", "decomposes") in edges
        assert (f"question:{run.id}", f"claim:{claim.id}", "decomposes") in edges, "no sub-question"
        [node] = [n for n in graph.nodes if n.id == f"sub_question:{child.id}"]
        assert node.meta.iteration == 1  # type: ignore[union-attr]
        assert node.meta.parent_id == str(root.id)  # type: ignore[union-attr]

    def test_the_schema_rejects_a_dangling_edge(self) -> None:
        import pytest
        from pydantic import ValidationError

        payload = {
            "run_id": "r",
            "nodes": [{"id": "question:r", "type": "question", "label": "Q", "meta": {
                "question": "Q", "mode": "quick", "status": "completed", "iteration": 0}}],
            "edges": [{"id": "e", "from": "question:r", "to": "claim:x", "type": "decomposes"}],
        }
        with pytest.raises(ValidationError, match="unknown node"):
            EvidenceGraph.model_validate(payload)
        payload["edges"] = []
        payload["nodes"].append({"id": "claim:x", "type": "claim", "label": "c", "meta": {"text": "c"}})
        with pytest.raises(ValidationError, match="verdict"):
            EvidenceGraph.model_validate(payload)
