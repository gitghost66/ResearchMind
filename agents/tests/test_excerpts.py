"""M5 tests — picking the part of a passage that bears on a claim (tools/excerpts.py)."""

from __future__ import annotations

import pytest

from researchmind_agents.tools.excerpts import best_excerpt, split_sentences

ELLIPSIS = "…"

PASSAGE = (
    "Heat pumps are widely sold in Europe. "
    "Installers report steady demand in Scandinavia. "
    "In the 2023 field study, cold-climate heat pumps kept a COP of 2.5 at -15 C. "
    "The trial ran in Maine and Alaska over two winters. "
    "Unrelated closing words follow here. "
    "Contact the editors for corrections."
)


class TestSplitSentences:
    def test_splits_on_sentence_ends(self) -> None:
        assert split_sentences("One fact. Two facts! Three? Four.") == [
            "One fact.",
            "Two facts!",
            "Three?",
            "Four.",
        ]

    def test_does_not_split_decimals_or_lowercase_continuations(self) -> None:
        assert split_sentences("The COP is 2.5 at most. e.g. this stays.") == [
            "The COP is 2.5 at most. e.g. this stays."
        ]

    def test_collapses_whitespace(self) -> None:
        assert split_sentences("  A  b.\n\n C d.  ") == ["A b.", "C d."]

    def test_empty(self) -> None:
        assert split_sentences("") == []


class TestBestExcerpt:
    def test_short_text_is_returned_whole(self) -> None:
        assert best_excerpt("  Short   text.  ", "anything", max_chars=100) == "Short text."

    def test_non_positive_budget_returns_whole_text(self) -> None:
        assert best_excerpt(PASSAGE, "COP", max_chars=0) == " ".join(PASSAGE.split())

    def test_centres_on_the_most_relevant_sentence(self) -> None:
        excerpt = best_excerpt(PASSAGE, "Heat pumps keep a COP of 2.5 at -15 C", max_chars=130)
        assert "COP of 2.5 at -15 C" in excerpt
        assert len(excerpt.strip(ELLIPSIS)) <= 130
        assert "Contact the editors" not in excerpt

    def test_widens_after_first_then_before(self) -> None:
        anchor = "In the 2023 field study, cold-climate heat pumps kept a COP of 2.5 at -15 C."
        after = "The trial ran in Maine and Alaska over two winters."
        excerpt = best_excerpt(PASSAGE, "COP 2.5 -15", max_chars=len(anchor) + len(after) + 1)
        assert excerpt == f"{ELLIPSIS}{anchor} {after}{ELLIPSIS}"

    def test_marks_cuts_with_ellipses(self) -> None:
        excerpt = best_excerpt(PASSAGE, "COP 2.5", max_chars=200)
        assert excerpt.startswith(ELLIPSIS)
        assert excerpt.endswith(ELLIPSIS)

    def test_no_leading_ellipsis_at_the_start(self) -> None:
        excerpt = best_excerpt(PASSAGE, "Heat pumps widely sold Europe", max_chars=90)
        assert excerpt.startswith("Heat pumps are widely sold in Europe.")
        assert excerpt.endswith(ELLIPSIS)

    def test_a_shared_number_counts_double(self) -> None:
        text = (
            "Heat pumps are common in homes across the region today. "
            "A separate trial measured 2.5 under load. "
            "Final remarks close the article now."
        )
        # One shared word (1 point) loses to one shared number (2 points).
        excerpt = best_excerpt(text, "pumps 2.5", max_chars=45)
        assert "2.5" in excerpt

    def test_long_single_sentence_is_cut_at_a_word(self) -> None:
        text = "Intro. Word " + "word " * 100 + "end."
        excerpt = best_excerpt(text, "word", max_chars=50)
        assert excerpt.startswith(ELLIPSIS)
        assert excerpt.endswith(ELLIPSIS)
        body = excerpt.strip(ELLIPSIS)
        assert len(body) <= 50
        # Cut at a space: no half word at the end.
        assert set(body.split(" ")) == {"Word", "word"}

    def test_no_overlap_picks_the_first_sentence(self) -> None:
        excerpt = best_excerpt(PASSAGE, "zebra quantum", max_chars=60)
        assert excerpt.startswith("Heat pumps are widely sold in Europe.")

    @pytest.mark.parametrize("budget", [40, 120, 250])
    def test_respects_the_budget(self, budget: int) -> None:
        excerpt = best_excerpt(PASSAGE, "Maine Alaska winters", max_chars=budget)
        assert len(excerpt.replace(ELLIPSIS, "")) <= budget
