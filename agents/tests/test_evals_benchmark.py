"""
M7: the benchmark suites and gold-source matching.

PRD s11: 30-50 questions across the four modes, with hand-labelled expected
sub-topics and a small set of gold sources.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from researchmind_agents.evals.benchmark import (
    OFFLINE_SUITE,
    gold_match,
    load_questions,
    load_suite,
    matching_gold,
    suite_summary,
)
from researchmind_agents.modes import SUPPORTED_MODES

BENCHMARK = load_suite("default")


class TestBenchmarkSet:
    def test_thirty_to_fifty_questions_across_every_mode(self) -> None:
        assert 30 <= len(BENCHMARK) <= 50
        counts = Counter(q.mode for q in BENCHMARK)
        assert set(counts) == SUPPORTED_MODES
        assert min(counts.values()) >= 7, counts

    def test_every_question_is_labelled(self) -> None:
        assert len({q.id for q in BENCHMARK}) == len(BENCHMARK)
        assert len({q.question for q in BENCHMARK}) == len(BENCHMARK)
        for q in BENCHMARK:
            assert len(q.expected_subtopics) >= 3, q.id
            assert len(q.gold_sources) >= 2, q.id
            assert len(set(q.gold_sources)) == len(q.gold_sources), q.id

    def test_gold_sources_are_hosts_or_pages_without_a_scheme(self) -> None:
        for q in BENCHMARK:
            for gold in q.gold_sources:
                host = gold.split("/", 1)[0]
                assert "://" not in gold and not gold.startswith("www."), (q.id, gold)
                assert "." in host and host == host.lower(), (q.id, gold)
                assert not gold.endswith("/"), (q.id, gold)

    def test_decision_questions_carry_context_and_expect_a_recommendation(self) -> None:
        decisions = [q for q in BENCHMARK if q.mode == "decision"]
        assert decisions
        for q in decisions:
            assert q.user_context, q.id
            assert any("recommendation" in topic for topic in q.expected_subtopics), q.id

    def test_seed_suite_is_unchanged_and_has_no_gold_sources(self) -> None:
        seed = load_suite("seed")
        assert len(seed) == 8
        assert all(q.mode == "quick" and not q.gold_sources for q in seed)

    def test_offline_suite_has_one_question_per_mode(self) -> None:
        offline = load_suite(OFFLINE_SUITE)
        assert sorted(q.mode for q in offline) == sorted(SUPPORTED_MODES)
        assert all(q.gold_sources and q.expected_subtopics for q in offline)

    def test_summary_counts_per_mode(self) -> None:
        assert sum(suite_summary(BENCHMARK).values()) == len(BENCHMARK)


class TestLoading:
    def write(self, tmp_path: Path, *items: dict) -> Path:
        path = tmp_path / "suite.jsonl"
        path.write_text("\n".join(json.dumps(i) for i in items) + "\n\n", encoding="utf-8")
        return path

    def test_a_jsonl_path_is_a_suite(self, tmp_path: Path) -> None:
        path = self.write(
            tmp_path,
            {"id": "x1", "question": "Q?", "mode": "deep", "gold_sources": ["a.example"],
             "user_context": {"team": "two"}},
        )
        [item] = load_suite(str(path))
        assert (item.id, item.mode, item.gold_sources, item.user_context) == (
            "x1", "deep", ("a.example",), {"team": "two"}
        )

    def test_mode_defaults_to_quick(self, tmp_path: Path) -> None:
        [item] = load_questions(self.write(tmp_path, {"id": "x", "question": "Q?"}))
        assert item.mode == "quick" and item.gold_sources == ()

    @pytest.mark.parametrize(
        "items, message",
        [
            ([{"id": "a", "question": "Q"}, {"id": "a", "question": "R"}], "repeated question id"),
            ([{"id": "a", "question": "Q", "mode": "turbo"}], "unknown mode"),
            ([{"id": "a", "question": "  "}], "empty"),
            ([{"question": "Q"}], "id"),
        ],
    )
    def test_bad_files_are_rejected_with_the_line(
        self, tmp_path: Path, items: list[dict], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            load_questions(self.write(tmp_path, *items))

    def test_unknown_suite(self) -> None:
        with pytest.raises(ValueError, match="unknown suite"):
            load_suite("nope")


class TestGoldMatch:
    @pytest.mark.parametrize(
        "url, gold, expected",
        [
            ("https://www.energy.gov/eere/heat-pumps", "energy.gov", True),
            ("https://blog.energy.gov/post", "energy.gov", True),
            ("https://notenergy.gov/post", "energy.gov", False),
            ("https://docs.python.org/3/whatsnew/3.13.html", "docs.python.org/3/whatsnew/3.13", True),
            ("https://docs.python.org/3/whatsnew/3.12.html", "docs.python.org/3/whatsnew/3.13", False),
            ("https://github.com/pgvector/pgvector/blob/master/README.md", "github.com/pgvector/pgvector", True),
            ("https://github.com/other/pgvector", "github.com/pgvector/pgvector", False),
            ("https://arxiv.org/abs/2005.11401v4", "arxiv.org/abs/2005.11401", True),
            ("HTTPS://GitHub.com/PGVector/pgvector", "github.com/pgvector/pgvector", True),
            ("http://energy.gov", "energy.gov", True),
            ("not a url", "energy.gov", False),
            ("https://energy.gov/x", "", False),
        ],
    )
    def test_host_or_subdomain_then_path_prefix(self, url: str, gold: str, expected: bool) -> None:
        assert gold_match(url, gold) is expected

    def test_matching_gold_names_the_first_match(self) -> None:
        gold = ["arxiv.org/abs/2005.11401", "arxiv.org"]
        assert matching_gold("https://arxiv.org/abs/2005.11401", gold) == gold[0]
        assert matching_gold("https://arxiv.org/abs/1603.09320", gold) == "arxiv.org"
        assert matching_gold("https://example.org", gold) is None
