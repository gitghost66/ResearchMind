"""M3 tests — text chunking and content hashing."""

from __future__ import annotations

import pytest

from researchmind_agents.tools.chunking import (
    CHARS_PER_TOKEN,
    chunk_text,
    content_hash,
    estimate_tokens,
)


def sentences(count: int, words: int = 12) -> str:
    """Distinct sentences of roughly equal length."""
    return " ".join(
        f"Sentence {i} " + " ".join(f"word{i}x{j}" for j in range(words)) + "."
        for i in range(count)
    )


class TestEstimateTokens:
    def test_empty_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_rounds_up(self) -> None:
        assert estimate_tokens("a") == 1
        assert estimate_tokens("a" * CHARS_PER_TOKEN) == 1
        assert estimate_tokens("a" * (CHARS_PER_TOKEN + 1)) == 2


class TestContentHash:
    def test_ignores_case_and_whitespace(self) -> None:
        assert content_hash("Hello   World\n\nAgain") == content_hash("hello world again")

    def test_differs_on_content(self) -> None:
        assert content_hash("hello world") != content_hash("hello there")

    def test_is_sha256_hex(self) -> None:
        digest = content_hash("x")
        assert len(digest) == 64
        int(digest, 16)


class TestChunkText:
    def test_empty_text(self) -> None:
        assert chunk_text("", chunk_tokens=100) == []
        assert chunk_text("   \n\n  ", chunk_tokens=100) == []

    def test_short_text_is_one_chunk(self) -> None:
        assert chunk_text("One sentence. Another one.", chunk_tokens=100) == [
            "One sentence. Another one."
        ]

    def test_every_chunk_fits(self) -> None:
        chunk_tokens = 60
        chunks = chunk_text(sentences(40), chunk_tokens=chunk_tokens, overlap_tokens=15)

        assert len(chunks) > 1
        assert all(len(chunk) <= chunk_tokens * CHARS_PER_TOKEN for chunk in chunks)

    def test_covers_every_sentence(self) -> None:
        text = sentences(20)
        chunks = chunk_text(text, chunk_tokens=60)
        joined = " ".join(chunks)
        for i in range(20):
            assert f"Sentence {i} " in joined

    def test_overlap_repeats_trailing_sentences(self) -> None:
        chunks = chunk_text(sentences(30, words=4), chunk_tokens=50, overlap_tokens=20)

        assert len(chunks) > 2
        for previous, current in zip(chunks, chunks[1:], strict=False):
            last_sentence = previous.rsplit("Sentence ", 1)[-1]
            assert current.startswith(f"Sentence {last_sentence}")

    def test_no_overlap_means_no_repeats(self) -> None:
        chunks = chunk_text(sentences(30, words=4), chunk_tokens=50, overlap_tokens=0)
        seen: list[str] = []
        for chunk in chunks:
            for part in chunk.split("Sentence ")[1:]:
                assert part not in seen
                seen.append(part)

    def test_max_chunks(self) -> None:
        chunks = chunk_text(sentences(60), chunk_tokens=40, max_chunks=3)
        assert len(chunks) == 3

    def test_max_chunks_keeps_document_order(self) -> None:
        all_chunks = chunk_text(sentences(60), chunk_tokens=40)
        assert chunk_text(sentences(60), chunk_tokens=40, max_chunks=2) == all_chunks[:2]

    def test_over_long_sentence_is_split_on_words(self) -> None:
        sentence = " ".join(f"word{i}" for i in range(200)) + "."
        chunks = chunk_text(sentence, chunk_tokens=20)

        assert len(chunks) > 1
        assert all(len(chunk) <= 20 * CHARS_PER_TOKEN for chunk in chunks)
        # Words are not cut in half.
        assert all(not chunk.startswith("ord") for chunk in chunks)

    def test_single_giant_word_is_hard_split(self) -> None:
        chunks = chunk_text("x" * 1000, chunk_tokens=25)

        assert len(chunks) == 10
        assert all(len(chunk) <= 25 * CHARS_PER_TOKEN for chunk in chunks)
        assert "".join(chunks) == "x" * 1000

    def test_paragraphs_are_split(self) -> None:
        text = "First paragraph here\nSecond paragraph here"
        assert chunk_text(text, chunk_tokens=100) == ["First paragraph here Second paragraph here"]

    def test_deterministic(self) -> None:
        text = sentences(25)
        assert chunk_text(text, chunk_tokens=50, overlap_tokens=10) == chunk_text(
            text, chunk_tokens=50, overlap_tokens=10
        )

    @pytest.mark.parametrize("bad", [0, -5])
    def test_invalid_chunk_tokens(self, bad: int) -> None:
        with pytest.raises(ValueError):
            chunk_text("text", chunk_tokens=bad)
