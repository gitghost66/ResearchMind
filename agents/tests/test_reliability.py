"""
M5 tests — deterministic source reliability and claim / report confidence
(researchmind_agents.reliability, PRD s8).

Pure functions only: no fakes, no I/O beyond reading the packaged domain list.
Explicit ScoringConfig values keep the arithmetic easy to check by hand.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from researchmind_agents.config import DEFAULT_TYPE_SCORES, DEFAULT_VERDICT_WEIGHTS, ScoringConfig
from researchmind_agents.reliability import (
    DOMAIN_TYPES,
    PAPER_TYPES,
    SCORING_VERSION,
    DomainRules,
    EvidenceSignal,
    SourceSignals,
    citation_bonus,
    claim_confidence,
    classify_domain,
    describes_methodology,
    load_domain_rules,
    overall_confidence,
    recency_factor,
    reconcile_verdict,
    source_reliability,
)

NOW = datetime(2026, 10, 27, tzinfo=UTC)
YEAR = timedelta(days=365.25)
DEFAULTS = ScoringConfig()


def years_ago(years: float) -> datetime:
    return NOW - YEAR * years


# ===========================================================================
# Domain types
# ===========================================================================


class TestClassifyDomain:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            # Listed hosts, and subdomains of listed hosts.
            ("https://nature.com/articles/x", "peer_reviewed"),
            ("https://www.nature.com/articles/x", "peer_reviewed"),
            ("https://en.wikipedia.org/wiki/Heat_pump", "reference"),
            ("https://arxiv.org/abs/2401.01234", "preprint"),
            ("https://github.com/org/repo", "code_repository"),
            ("https://www.reddit.com/r/x", "forum"),
            ("https://medium.com/@a/post", "blog"),
            ("https://sub.blog.google/x", "company_blog"),
            # Longest listed match wins over a shorter parent entry.
            ("https://spectrum.ieee.org/article", "news"),
            ("https://standards.ieee.org/ieee/802", "standards_body"),
            ("https://ieeexplore.ieee.org/document/1", "peer_reviewed"),
            # A listed host beats the TLD rule that would otherwise apply.
            ("https://nist.gov/publications", "standards_body"),
        ],
    )
    def test_listed_hosts(self, url: str, expected: str) -> None:
        assert classify_domain(url) == expected

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://energy.gov/eere", "government"),
            ("https://www.army.mil/x", "government"),
            ("https://www.gov.uk/guidance", "government"),
            ("https://assets.publishing.service.gov.uk/x.pdf", "government"),
            ("https://who.int/news", "government"),
            ("https://cs.stanford.edu/people", "academic"),
            ("https://www.ox.ac.uk/research", "academic"),
        ],
    )
    def test_suffix_rules(self, url: str, expected: str) -> None:
        assert classify_domain(url) == expected

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://docs.example.io/intro", "official_docs"),
            ("https://developer.example.io/api", "official_docs"),
            ("https://blog.example.io/post", "company_blog"),
            ("https://engineering.example.io/post", "company_blog"),
            ("https://forum.example.io/t/1", "forum"),
            ("https://community.example.io/t/1", "forum"),
        ],
    )
    def test_host_prefixes(self, url: str, expected: str) -> None:
        assert classify_domain(url) == expected

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://example.io/docs", "official_docs"),
            ("https://example.io/docs/intro", "official_docs"),
            ("https://example.io/pricing", "official_docs"),
            ("https://example.io/blog/launch", "company_blog"),
            ("https://example.io/questions/42", "forum"),
            # A prefix must be a whole path segment.
            ("https://example.io/docsearch", "unknown"),
            ("https://example.io/", "unknown"),
        ],
    )
    def test_path_prefixes(self, url: str, expected: str) -> None:
        assert classify_domain(url) == expected

    def test_listed_docs_host_yields_to_a_blog_or_forum_path(self) -> None:
        assert classify_domain("https://docs.aws.amazon.com/guide") == "official_docs"
        assert classify_domain("https://docs.aws.amazon.com/blogs/post") == "company_blog"
        assert classify_domain("https://python.org/questions/1") == "forum"
        # Other listed types keep their type whatever the path.
        assert classify_domain("https://github.com/org/repo/blob/main/docs/x.md") == "code_repository"

    def test_case_insensitive(self) -> None:
        assert classify_domain("HTTPS://WWW.Nature.COM/X") == "peer_reviewed"

    @pytest.mark.parametrize("url", ["", "not a url", "https://", "http://[::1", "mailto:a@b.c"])
    def test_garbage_is_unknown(self, url: str) -> None:
        assert classify_domain(url) == "unknown"

    def test_custom_rules(self) -> None:
        rules = DomainRules.from_dict(
            {"version": "t/1", "types": {"news": ["paper.example"]}, "suffix_rules": [[".test", "government"]]}
        )
        assert classify_domain("https://www.paper.example/a", rules) == "news"
        assert classify_domain("https://agency.test/a", rules) == "government"
        assert classify_domain("https://docs.other.example/a", rules) == "unknown"


class TestDomainRules:
    def test_unknown_type_in_types_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown domain type"):
            DomainRules.from_dict({"types": {"tabloid": ["x.example"]}})

    def test_unknown_type_in_a_rule_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown domain type"):
            DomainRules.from_dict({"path_prefixes": [["/x", "tabloid"]]})

    def test_rules_are_tried_longest_first(self) -> None:
        rules = DomainRules.from_dict(
            {"suffix_rules": [[".uk", "news"], [".gov.uk", "government"]]}
        )
        assert rules.suffix_rules[0] == (".gov.uk", "government")
        assert classify_domain("https://x.gov.uk", rules) == "government"
        assert classify_domain("https://x.co.uk", rules) == "news"

    def test_missing_version_is_marked(self) -> None:
        assert DomainRules.from_dict({}).version == "unversioned"


class TestPackagedDomainList:
    def test_loads_and_is_versioned(self) -> None:
        rules = load_domain_rules()
        assert rules.version.startswith("domains/v")
        assert rules.hosts
        assert load_domain_rules() is rules  # read once

    def test_entries_are_bare_hosts_with_known_types(self) -> None:
        from researchmind_agents import reliability

        raw = json.loads(reliability._DOMAINS_PATH.read_text(encoding="utf-8"))
        assert set(raw["types"]) <= set(DOMAIN_TYPES)
        seen: dict[str, str] = {}
        for domain_type, hosts in raw["types"].items():
            for host in hosts:
                assert host == host.lower().strip()
                assert "/" not in host and ":" not in host, host
                assert not host.startswith("www."), host
                assert host not in seen, f"{host} listed as {seen.get(host)} and {domain_type}"
                seen[host] = domain_type

    def test_every_domain_type_has_a_default_score(self) -> None:
        assert set(dict(DEFAULT_TYPE_SCORES)) == set(DOMAIN_TYPES)


# ===========================================================================
# Source reliability
# ===========================================================================


class TestRecencyFactor:
    CONFIG = ScoringConfig(
        recency_full_years=2.0,
        recency_penalty_per_year=0.1,
        recency_floor=0.5,
        unknown_date_factor=0.9,
    )

    def test_fresh_sources_keep_full_score(self) -> None:
        assert recency_factor(years_ago(0.5), now=NOW, config=self.CONFIG) == 1.0
        assert recency_factor(years_ago(2.0), now=NOW, config=self.CONFIG) == 1.0

    def test_unknown_date(self) -> None:
        assert recency_factor(None, now=NOW, config=self.CONFIG) == 0.9

    def test_linear_decay_after_the_full_period(self) -> None:
        assert recency_factor(years_ago(4.0), now=NOW, config=self.CONFIG) == pytest.approx(0.8)
        assert recency_factor(years_ago(5.0), now=NOW, config=self.CONFIG) == pytest.approx(0.7)

    def test_never_below_the_floor(self) -> None:
        assert recency_factor(years_ago(40.0), now=NOW, config=self.CONFIG) == 0.5

    def test_naive_datetime_is_utc(self) -> None:
        naive = years_ago(4.0).replace(tzinfo=None)
        assert recency_factor(naive, now=NOW, config=self.CONFIG) == pytest.approx(0.8)

    def test_future_date_is_not_rewarded(self) -> None:
        assert recency_factor(NOW + YEAR, now=NOW, config=self.CONFIG) == 1.0


class TestCitationBonus:
    CONFIG = ScoringConfig(citation_bonus_max=0.1, citation_saturation=99)

    @pytest.mark.parametrize("count", [None, 0, -5])
    def test_no_citations_no_bonus(self, count: int | None) -> None:
        assert citation_bonus(count, self.CONFIG) == 0.0

    def test_log_scale_up_to_the_cap(self) -> None:
        # log(1 + 9) / log(1 + 99) = 0.5
        assert citation_bonus(9, self.CONFIG) == pytest.approx(0.05)
        assert citation_bonus(99, self.CONFIG) == pytest.approx(0.1)
        assert citation_bonus(10_000, self.CONFIG) == pytest.approx(0.1)

    def test_monotonic(self) -> None:
        values = [citation_bonus(c, self.CONFIG) for c in (1, 3, 10, 50, 99, 500)]
        assert values == sorted(values)

    def test_disabled_when_cap_is_zero(self) -> None:
        assert citation_bonus(500, ScoringConfig(citation_bonus_max=0.0)) == 0.0


class TestDescribesMethodology:
    def test_two_distinct_markers_required(self) -> None:
        assert describes_methodology("We evaluate the approach on a public dataset of 120 homes.")
        assert describes_methodology("A randomized trial with 400 participants.")

    def test_one_marker_is_not_enough(self) -> None:
        assert not describes_methodology("The method is simple.")
        # Variants of the same marker count once.
        assert not describes_methodology("Our methods follow the methodology of earlier work.")
        assert not describes_methodology("")


class TestSourceReliability:
    def score(self, domain_type: str, **signals: object) -> float:
        return source_reliability(
            SourceSignals(domain_type=domain_type, **signals),  # type: ignore[arg-type]
            now=NOW,
            config=DEFAULTS,
        )

    def test_prd_ordering_with_defaults(self) -> None:
        """PRD s8: peer-reviewed / official docs / standards > publications > company blogs > forums/unknown."""
        top = [self.score(t) for t in ("peer_reviewed", "official_docs", "standards_body")]
        news = self.score("news")
        company = self.score("company_blog")
        bottom = [self.score(t) for t in ("forum", "unknown")]
        assert min(top) > news > company > max(bottom)

    def test_type_score_times_recency(self) -> None:
        config = ScoringConfig(type_scores=(("news", 0.8), ("unknown", 0.4)), recency_full_years=1.0, recency_penalty_per_year=0.1)
        fresh = source_reliability(SourceSignals("news", years_ago(0.5)), now=NOW, config=config)
        old = source_reliability(SourceSignals("news", years_ago(3.0)), now=NOW, config=config)
        assert fresh == pytest.approx(0.8)
        assert old == pytest.approx(0.8 * 0.8)

    def test_unknown_type_uses_the_unknown_score(self) -> None:
        assert self.score("not-a-type") == self.score("unknown")

    def test_paper_bonuses_only_for_paper_types(self) -> None:
        bonuses = {"citation_count": 1000, "has_methodology": True}
        for domain_type in PAPER_TYPES:
            assert self.score(domain_type, **bonuses) > self.score(domain_type)
        for domain_type in ("news", "official_docs", "forum", "unknown"):
            assert self.score(domain_type, **bonuses) == self.score(domain_type)

    def test_bonuses_add_up(self) -> None:
        config = ScoringConfig(
            type_scores=(("preprint", 0.6),),
            unknown_date_factor=1.0,
            citation_bonus_max=0.1,
            citation_saturation=99,
            methodology_bonus=0.05,
        )
        signals = SourceSignals("preprint", citation_count=9, has_methodology=True)
        assert source_reliability(signals, now=NOW, config=config) == pytest.approx(0.6 + 0.05 + 0.05)

    def test_clamped_and_rounded(self) -> None:
        maxed = self.score("peer_reviewed", published_at=years_ago(0.1), citation_count=10**6, has_methodology=True)
        assert maxed == 1.0
        value = self.score("unknown")
        assert value == round(value, 4)
        assert 0.0 <= self.score("forum") <= 1.0


# ===========================================================================
# Claims
# ===========================================================================


class TestReconcileVerdict:
    @pytest.mark.parametrize(
        ("verdict", "stances", "expected"),
        [
            ("supported", [], "unsupported"),
            ("supported", ["neutral"], "unsupported"),
            ("supported", ["contradicts", "neutral"], "unsupported"),
            ("supported", ["supports"], "supported"),
            ("weak", ["supports", "neutral"], "weak"),
            ("supported", ["supports", "contradicts"], "mixed"),
            ("unsupported", ["supports", "contradicts"], "mixed"),
            ("mixed", ["supports", "contradicts"], "mixed"),
            # The model saw support but said otherwise: the cautious verdict.
            ("unsupported", ["supports"], "weak"),
            ("mixed", ["supports", "neutral"], "weak"),
        ],
    )
    def test_truth_table(self, verdict: str, stances: list[str], expected: str) -> None:
        assert reconcile_verdict(verdict, stances) == expected

    def test_accepts_any_iterable(self) -> None:
        assert reconcile_verdict("supported", iter(["supports"])) == "supported"


def ev(site: str, reliability: float, stance: str = "supports") -> EvidenceSignal:
    return EvidenceSignal(site=site, reliability=reliability, stance=stance)


class TestClaimConfidence:
    def test_single_source_is_source_weight_times_reliability(self) -> None:
        assert claim_confidence("supported", [ev("a.example", 0.5)], config=DEFAULTS) == pytest.approx(0.8 * 0.5)
        config = ScoringConfig(source_weight=0.5)
        assert claim_confidence("supported", [ev("a.example", 0.9)], config=config) == pytest.approx(0.45)

    def test_one_site_counts_once_with_its_best_source(self) -> None:
        one = claim_confidence("supported", [ev("a.example", 0.5)], config=DEFAULTS)
        twice = claim_confidence("supported", [ev("a.example", 0.5), ev("a.example", 0.5)], config=DEFAULTS)
        assert twice == one
        best = claim_confidence("supported", [ev("a.example", 0.5), ev("a.example", 0.7)], config=DEFAULTS)
        assert best == pytest.approx(0.8 * 0.7)

    def test_independent_sites_compound(self) -> None:
        one = claim_confidence("supported", [ev("a.example", 0.7)], config=DEFAULTS)
        two = claim_confidence("supported", [ev("a.example", 0.7), ev("b.example", 0.7)], config=DEFAULTS)
        three = claim_confidence(
            "supported", [ev("a.example", 0.7), ev("b.example", 0.7), ev("c.example", 0.7)], config=DEFAULTS
        )
        assert one < two < three < 1.0
        # Noisy-OR: 1 - (1 - 0.8 * 0.7)^2
        assert two == pytest.approx(1 - (1 - 0.56) ** 2, abs=1e-4)

    def test_contradiction_lowers_agreement(self) -> None:
        supported = claim_confidence("mixed", [ev("a.example", 0.7)], config=DEFAULTS)
        disputed = claim_confidence("mixed", [ev("a.example", 0.7), ev("b.example", 0.7, "contradicts")], config=DEFAULTS)
        assert disputed == pytest.approx(supported / 2, abs=1e-4)

    def test_neutral_evidence_changes_nothing(self) -> None:
        base = claim_confidence("supported", [ev("a.example", 0.6)], config=DEFAULTS)
        with_neutral = claim_confidence("supported", [ev("a.example", 0.6), ev("b.example", 0.9, "neutral")], config=DEFAULTS)
        assert with_neutral == base

    def test_hand_computed_case(self) -> None:
        """supported a (0.8), b (0.6); contradicted by c (0.4); verdict mixed."""
        config = ScoringConfig(source_weight=0.5)
        evidence = [ev("a.example", 0.8), ev("b.example", 0.6), ev("c.example", 0.4, "contradicts")]
        corroboration = 1 - (1 - 0.4) * (1 - 0.3)  # 0.58
        agreement = 1.4 / (1.4 + 0.4)
        expected = round(0.35 * corroboration * agreement, 4)
        assert claim_confidence("mixed", evidence, config=config) == expected == 0.1579

    def test_verdict_weights(self) -> None:
        evidence = [ev("a.example", 0.5)]
        values = {v: claim_confidence(v, evidence, config=DEFAULTS) for v in dict(DEFAULT_VERDICT_WEIGHTS)}
        assert values["supported"] > values["weak"] > values["mixed"] > 0
        assert values["unsupported"] == values["unverified"] == 0.0
        assert values["weak"] == pytest.approx(0.6 * values["supported"], abs=1e-4)

    def test_no_supporting_evidence_is_zero(self) -> None:
        assert claim_confidence("supported", [], config=DEFAULTS) == 0.0
        assert claim_confidence("supported", [ev("a.example", 0.9, "neutral")], config=DEFAULTS) == 0.0
        assert claim_confidence("supported", [ev("a.example", 0.9, "contradicts")], config=DEFAULTS) == 0.0

    def test_unknown_verdict_is_zero(self) -> None:
        assert claim_confidence("probably", [ev("a.example", 0.9)], config=DEFAULTS) == 0.0

    def test_out_of_range_reliability_is_clamped(self) -> None:
        value = claim_confidence("supported", [ev("a.example", 3.0)], config=DEFAULTS)
        assert value == pytest.approx(0.8)

    def test_result_is_bounded_and_rounded(self) -> None:
        evidence = [ev(f"s{i}.example", 1.0) for i in range(20)]
        value = claim_confidence("supported", evidence, config=DEFAULTS)
        assert 0.0 <= value <= 1.0
        assert value == round(value, 4)


class TestOverallConfidence:
    CONFIG = ScoringConfig(uncovered_penalty=0.5, contradiction_penalty=0.1, contradiction_penalty_cap=0.5)

    def test_none_without_cited_claims(self) -> None:
        assert overall_confidence([], sub_questions=3, covered_sub_questions=3, config=self.CONFIG) is None
        assert overall_confidence([(0.9, 0)], sub_questions=3, covered_sub_questions=3, config=self.CONFIG) is None

    def test_weighted_mean_by_citations(self) -> None:
        value = overall_confidence([(0.8, 2), (0.5, 1)], sub_questions=2, covered_sub_questions=2, config=self.CONFIG)
        assert value == pytest.approx(0.7)

    def test_zero_weight_claims_are_ignored(self) -> None:
        value = overall_confidence([(0.8, 1), (0.1, 0)], sub_questions=1, covered_sub_questions=1, config=self.CONFIG)
        assert value == pytest.approx(0.8)

    def test_uncovered_sub_questions_scale_the_penalty(self) -> None:
        half = overall_confidence([(0.8, 1)], sub_questions=4, covered_sub_questions=2, config=self.CONFIG)
        none = overall_confidence([(0.8, 1)], sub_questions=4, covered_sub_questions=0, config=self.CONFIG)
        assert half == pytest.approx(0.8 * 0.75)
        assert none == pytest.approx(0.8 * 0.5)

    def test_coverage_is_clamped_and_zero_sub_questions_is_no_penalty(self) -> None:
        over = overall_confidence([(0.8, 1)], sub_questions=2, covered_sub_questions=5, config=self.CONFIG)
        empty = overall_confidence([(0.8, 1)], sub_questions=0, covered_sub_questions=0, config=self.CONFIG)
        assert over == empty == pytest.approx(0.8)

    def test_contradiction_penalty_and_cap(self) -> None:
        two = overall_confidence(
            [(0.8, 1)], sub_questions=1, covered_sub_questions=1, unresolved_contradictions=2, config=self.CONFIG
        )
        many = overall_confidence(
            [(0.8, 1)], sub_questions=1, covered_sub_questions=1, unresolved_contradictions=50, config=self.CONFIG
        )
        assert two == pytest.approx(0.8 * 0.8)
        assert many == pytest.approx(0.8 * 0.5)

    def test_combined_hand_computed_case(self) -> None:
        value = overall_confidence(
            [(0.8, 2), (0.5, 1)],
            sub_questions=4,
            covered_sub_questions=2,
            unresolved_contradictions=2,
            config=self.CONFIG,
        )
        assert value == round(0.7 * 0.75 * 0.8, 4) == 0.42

    def test_clamped(self) -> None:
        value = overall_confidence([(1.7, 1)], sub_questions=1, covered_sub_questions=1, config=self.CONFIG)
        assert value == 1.0


def test_scoring_version_is_set() -> None:
    assert SCORING_VERSION.startswith("scoring/v")
