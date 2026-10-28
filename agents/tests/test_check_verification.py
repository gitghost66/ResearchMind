"""
M5 verification eval: the gold set, and the scoring's calibration against it.

With a model that returns the labelled stances and verdicts, the fact checker
and the deterministic scoring must reproduce every label: each verdict, and
each confidence within the eval's tolerance. That pins the confidence formula
to the hand labels; the live run (``backend/scripts/eval_verification.py``)
then measures the model itself.
"""

from __future__ import annotations

import json
import re

from researchmind_agents.evals.check_verification import (
    TOLERANCE,
    GoldClaim,
    evaluate_verification,
    load_gold,
)
from researchmind_agents.reliability import DOMAIN_TYPES
from researchmind_agents.state import Verdict
from researchmind_agents.testing import build_fake_runtime

GOLD = load_gold()


def gold_responder(items: list[GoldClaim]):  # type: ignore[no-untyped-def]
    """Answer a fact-check prompt with the labelled stances and verdict."""
    by_text = {item.claim: item for item in items}

    def respond(prompt: str) -> str:
        checks = []
        current = None
        for line in prompt.splitlines():
            claim = re.match(r"^\[(C\d+)\] (.*)$", line)
            evidence = re.match(r"^\s+\[(E\d+)\] ", line)
            if claim:
                item = by_text[claim.group(2)]
                current = {"claim": claim.group(1), "verdict": item.verdict, "evidence": [], "reason": "gold", "_item": item}
                checks.append(current)
            elif evidence and current is not None:
                index = len(current["evidence"])
                stance = current["_item"].evidence[index].stance
                current["evidence"].append({"evidence": evidence.group(1), "stance": stance})
        for check in checks:
            del check["_item"]
        return json.dumps({"checks": checks})

    return respond


class TestGoldSet:
    def test_twenty_claims_covering_every_verdict(self) -> None:
        assert len(GOLD) == 20
        assert len({item.id for item in GOLD}) == 20
        verdicts = [item.verdict for item in GOLD]
        assert set(verdicts) == set(Verdict.__args__)  # type: ignore[attr-defined]
        assert min(verdicts.count(v) for v in set(verdicts)) >= 4

    def test_labels_are_well_formed(self) -> None:
        for item in GOLD:
            assert item.evidence, item.id
            assert 0.0 <= item.confidence <= 1.0
            for evidence in item.evidence:
                assert evidence.stance in {"supports", "contradicts", "neutral"}
                assert evidence.domain_type in DOMAIN_TYPES
                # Synthetic passages live on example hosts only.
                assert ".example" in evidence.url, evidence.url

    def test_labels_agree_with_their_own_stances(self) -> None:
        """No support means unsupported (and confidence 0); support plus contradiction means mixed."""
        for item in GOLD:
            stances = [e.stance for e in item.evidence]
            if "supports" not in stances:
                assert item.verdict == "unsupported" and item.confidence == 0.0, item.id
            elif "contradicts" in stances:
                assert item.verdict == "mixed", item.id
            else:
                assert item.verdict in {"supported", "weak"}, item.id


class TestCalibration:
    async def test_labelled_judgements_reproduce_every_label(self) -> None:
        runtime = build_fake_runtime(responses={"FactCheckOutput": gold_responder(GOLD)})
        result = await evaluate_verification(runtime, GOLD)

        assert result.verdict_agreement == 1.0
        assert result.stance_agreement == 1.0
        misses = [
            (o.id, o.confidence, o.expected_confidence)
            for o in result.outcomes
            if o.confidence_error > TOLERANCE
        ]
        assert misses == []
        assert result.passed()
        assert result.confidence_mae <= 0.08

    async def test_more_independent_support_scores_higher(self) -> None:
        runtime = build_fake_runtime(responses={"FactCheckOutput": gold_responder(GOLD)})
        result = await evaluate_verification(runtime, GOLD)
        score = {o.id: o.confidence for o in result.outcomes}

        # Two reliable sources beat one; a forum anecdote is the weakest support.
        assert score["g01"] > score["g02"] > score["g06"]
        # Contradiction drags a claim below the same claim's support alone.
        assert score["g13"] < score["g02"]
        assert all(score[g] == 0.0 for g in ("g17", "g18", "g19", "g20"))


class TestScoringTheModel:
    async def test_a_model_that_always_says_supported_fails_the_bar(self) -> None:
        def naive(prompt: str) -> str:
            checks = []
            for line in prompt.splitlines():
                claim = re.match(r"^\[(C\d+)\] ", line)
                evidence = re.match(r"^\s+\[(E\d+)\] ", line)
                if claim:
                    checks.append({"claim": claim.group(1), "verdict": "supported", "evidence": [], "reason": "x"})
                elif evidence and checks:
                    checks[-1]["evidence"].append({"evidence": evidence.group(1), "stance": "supports"})
            return json.dumps({"checks": checks})

        runtime = build_fake_runtime(responses={"FactCheckOutput": naive})
        result = await evaluate_verification(runtime, GOLD)

        assert result.verdict_agreement < 0.8
        assert not result.passed()

    async def test_unchecked_claims_count_as_disagreements(self) -> None:
        runtime = build_fake_runtime(responses={"FactCheckOutput": RuntimeError("down")})
        result = await evaluate_verification(runtime, GOLD[:3])

        assert [o.verdict for o in result.outcomes] == ["unverified"] * 3
        assert result.verdict_agreement == 0.0
        summary = result.summary()
        assert summary["claims"] == 3 and summary["passed"] is False
        json.dumps(summary)
