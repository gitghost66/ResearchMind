"""M5/M6 tests — research modes and domains (researchmind_agents.modes)."""

from __future__ import annotations

import pytest

from researchmind_agents.graph import RESEARCHER_NODES, pipeline_steps
from researchmind_agents.modes import (
    ACADEMIC,
    DOMAIN_DESCRIPTIONS,
    MARKET,
    MODES,
    RESEARCH_DOMAINS,
    SUPPORTED_MODES,
    TECHNICAL,
    WEB,
    ModeProfile,
    mode_profile,
    normalize_domain,
)
from researchmind_agents.state import SUPPORTED_MODES as STATE_SUPPORTED_MODES


class TestProfiles:
    def test_supported_modes(self) -> None:
        assert frozenset({"quick", "deep", "academic", "decision"}) == SUPPORTED_MODES
        assert frozenset(MODES) == SUPPORTED_MODES
        # state.py re-exports the same set for older imports.
        assert STATE_SUPPORTED_MODES is SUPPORTED_MODES

    def test_quick_is_web_only_and_unverified(self) -> None:
        quick = mode_profile("quick")
        assert quick.domains == (WEB,)
        assert quick.default_domain == WEB
        assert quick.favoured == ()
        assert quick.verify is False
        assert quick.loop is False
        assert quick.recommendation is False

    def test_deep_excludes_academic_and_verifies(self) -> None:
        deep = mode_profile("deep")
        assert deep.domains == (WEB, TECHNICAL, MARKET)
        assert ACADEMIC not in deep.domains
        assert deep.verify is True
        assert deep.loop is True
        assert deep.recommendation is False

    def test_academic_favours_papers_and_verifies(self) -> None:
        academic = mode_profile("academic")
        assert set(academic.domains) == set(RESEARCH_DOMAINS)
        assert academic.favoured == (ACADEMIC,)
        assert academic.default_domain == WEB
        assert academic.verify is True
        assert academic.loop is True

    def test_decision_favours_market_and_requires_a_recommendation(self) -> None:
        decision = mode_profile("decision")
        # PRD s6.4 "All": the academic researcher is offered, not favoured.
        assert set(decision.domains) == set(RESEARCH_DOMAINS)
        assert decision.favoured == (MARKET,)
        assert decision.default_domain == WEB
        assert decision.verify is True
        assert decision.loop is True
        assert decision.recommendation is True

    def test_prd_latency_targets(self) -> None:
        targets = {name: profile.target_seconds for name, profile in MODES.items()}
        assert targets == {"quick": 45.0, "deep": 300.0, "academic": 360.0, "decision": 300.0}

    def test_a_loop_needs_verification(self) -> None:
        with pytest.raises(ValueError, match="verified claims"):
            ModeProfile(name="bad", domains=(WEB,), loop=True)

    def test_every_mode_starts_with_a_known_domain(self) -> None:
        for profile in MODES.values():
            assert profile.domains
            assert set(profile.domains) <= set(RESEARCH_DOMAINS)
            assert set(profile.favoured) <= set(profile.domains)

    def test_every_domain_is_described(self) -> None:
        assert set(DOMAIN_DESCRIPTIONS) == set(RESEARCH_DOMAINS)
        assert all(text.strip() for text in DOMAIN_DESCRIPTIONS.values())

    @pytest.mark.parametrize("mode", ["nonsense", "", "QUICK"])
    def test_unknown_mode_raises(self, mode: str) -> None:
        with pytest.raises(ValueError, match="not available"):
            mode_profile(mode)


class TestNormalizeDomain:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("technical", TECHNICAL),
            ("  Technical ", TECHNICAL),
            ("MARKET", MARKET),
            ("tech", TECHNICAL),
            ("engineering", TECHNICAL),
            ("cost", MARKET),
            ("market/cost", MARKET),
            ("general", WEB),
            ("web", WEB),
            # Not allowed in deep mode: the mode's default.
            ("academic", WEB),
            ("papers", WEB),
            ("astrology", WEB),
            ("", WEB),
        ],
    )
    def test_deep(self, raw: str, expected: str) -> None:
        assert normalize_domain(raw, mode_profile("deep")) == expected

    @pytest.mark.parametrize("raw", ["academic", "papers", "Scholarly", "research"])
    def test_academic_aliases(self, raw: str) -> None:
        assert normalize_domain(raw, mode_profile("academic")) == ACADEMIC

    @pytest.mark.parametrize("raw", ["technical", "market", "academic", "web"])
    def test_quick_maps_everything_to_web(self, raw: str) -> None:
        assert normalize_domain(raw, mode_profile("quick")) == WEB


class TestPipelineSteps:
    def test_quick(self) -> None:
        assert pipeline_steps("quick") == ("planner", "web_researcher", "synthesizer")

    def test_deep(self) -> None:
        assert pipeline_steps("deep") == (
            "planner",
            "web_researcher",
            "technical_researcher",
            "market_researcher",
            "claim_extractor",
            "fact_checker",
            "contradiction_detector",
            "sufficiency_checker",
            "synthesizer",
        )

    def test_decision_has_every_researcher_and_the_loop(self) -> None:
        assert pipeline_steps("decision") == pipeline_steps("academic")

    def test_academic_adds_the_academic_researcher(self) -> None:
        steps = pipeline_steps("academic")
        assert "academic_researcher" in steps
        assert steps.index("academic_researcher") < steps.index("claim_extractor")

    def test_every_domain_has_a_researcher_node(self) -> None:
        assert set(RESEARCHER_NODES) == set(RESEARCH_DOMAINS)

    def test_unknown_mode_has_no_steps(self) -> None:
        assert pipeline_steps("nonsense") == ()
