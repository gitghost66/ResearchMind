"""
M6 contradiction eval: the gold set, and the harness against a model that
answers with the labels.

With the labelled answers, every miss is the deterministic part's: a seeded
pair that candidate selection did not put in one topic. The fake embeddings
are a bag of words, so this pins the pipeline, not the live similarity
threshold; the live run (``backend/scripts/eval_contradictions.py``) measures
the real embeddings and model.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from urllib.parse import urlsplit

from researchmind_agents.config import CONTRADICTION_CAUSES, AgentConfig, ContradictionConfig
from researchmind_agents.evals.check_contradictions import (
    TARGET,
    ContradictionResult,
    GoldCase,
    evaluate_contradictions,
    load_gold,
)
from researchmind_agents.reliability import DOMAIN_TYPES
from researchmind_agents.testing import build_fake_runtime

GOLD = load_gold()

# The fake embeddings count shared words, so a paraphrase ("permits" against
# "does not permit") scores lower than a real embedding would: every seeded
# pair scores at least 0.5 here. The live eval reports candidate recall at the
# real threshold.
BAG_OF_WORDS = AgentConfig(contradictions=ContradictionConfig(min_similarity=0.45))


def gold_responder(cases: list[GoldCase], *, cause_of: Callable[[str], str] | None = None) -> Callable[[str], str]:
    """Answer a contradiction prompt with the seeded contradictions among the claims shown."""
    by_text = {claim.text: (case, claim.id) for case in cases for claim in case.claims}

    def respond(prompt: str) -> str:
        labels: dict[tuple[str, str], str] = {}
        case = None
        for match in re.finditer(r"^\[(C\d+)\] \(\w+\) (.*)$", prompt, re.M):
            case, claim_id = by_text[match.group(2)]
            labels[(case.id, claim_id)] = match.group(1)
        found = []
        if case is not None:
            for seeded in case.contradictions:
                keys = [(case.id, cid) for cid in seeded.claims]
                if all(key in labels for key in keys):
                    cause = cause_of(seeded.likely_cause) if cause_of else seeded.likely_cause
                    found.append(
                        {
                            "claims": [labels[key] for key in keys],
                            "likely_cause": cause,
                            "explanation": seeded.note or "Seeded.",
                        }
                    )
        return json.dumps({"contradictions": found})

    return respond


class TestGoldSet:
    def test_fifteen_cases_fourteen_contradictions(self) -> None:
        assert len(GOLD) == 15
        assert len({case.id for case in GOLD}) == 15
        assert sum(len(case.contradictions) for case in GOLD) == 14
        assert sum(1 for case in GOLD if not case.contradictions) == 3

    def test_every_cause_is_seeded(self) -> None:
        causes = Counter(x.likely_cause for case in GOLD for x in case.contradictions)
        assert set(causes) == set(CONTRADICTION_CAUSES)
        for cause in ("conditions", "date", "version", "dataset"):
            assert causes[cause] >= 2, cause

    def test_cases_are_well_formed(self) -> None:
        for case in GOLD:
            ids = [claim.id for claim in case.claims]
            assert len(ids) == len(set(ids)), case.id
            in_conflict = {cid for x in case.contradictions for cid in x.claims}
            assert in_conflict <= set(ids), case.id
            assert all(len(set(x.claims)) >= 2 for x in case.contradictions), case.id
            if case.contradictions:
                assert set(ids) - in_conflict, f"{case.id} needs a claim that conflicts with nothing"
            for claim in case.claims:
                assert claim.verdict in {"supported", "weak", "mixed"}
                for passage in claim.evidence:
                    assert passage.domain_type in DOMAIN_TYPES
                    # Synthetic text lives on example hosts only.
                    host = urlsplit(passage.url).hostname or ""
                    assert host.endswith(".example"), passage.url


class TestHarness:
    async def test_labelled_answers_find_every_seeded_contradiction(self) -> None:
        runtime = build_fake_runtime(BAG_OF_WORDS, responses={"ContradictionOutput": gold_responder(GOLD)})
        result = await evaluate_contradictions(runtime, GOLD)

        assert result.seeded == 14
        assert result.candidate_recall == 1.0, [
            (o.id, o.seeded) for o in result.outcomes if o.candidates_found < len(o.seeded)
        ]
        assert result.recall == 1.0
        assert result.cause_agreement == 1.0
        assert result.precision == 1.0
        assert result.false_positives == 0
        assert result.passed()
        summary = result.summary()
        assert summary["found"] == 14 and summary["passed"] is True
        json.dumps(summary)

    async def test_wrong_causes_lower_cause_agreement_only(self) -> None:
        runtime = build_fake_runtime(
            BAG_OF_WORDS,
            responses={"ContradictionOutput": gold_responder(GOLD, cause_of=lambda _: "unexplained")},
        )
        result = await evaluate_contradictions(runtime, GOLD)
        assert result.recall == 1.0
        assert result.cause_agreement == 1 / 14

    async def test_a_silent_model_misses_everything(self) -> None:
        result = await evaluate_contradictions(build_fake_runtime(BAG_OF_WORDS), GOLD)
        assert result.recall == 0.0
        assert not result.passed()
        assert result.false_positives == 0

    async def test_candidate_recall_reports_pairs_the_threshold_misses(self) -> None:
        runtime = build_fake_runtime(responses={"ContradictionOutput": gold_responder(GOLD)})
        result = await evaluate_contradictions(runtime, GOLD)
        # At the default 0.7, the two paraphrased pairs are not compared.
        assert result.candidate_recall == 12 / 14
        assert result.recall == result.candidate_recall

    def test_target(self) -> None:
        assert TARGET == 0.7
        assert ContradictionResult().recall == 0.0
