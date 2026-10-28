"""
Pick the part of a passage that bears on a claim.

The fact checker reads several passages per claim. Showing each one whole
(~800 tokens) would make checking cost more than research, so it reads an
excerpt instead: the sentence that shares the most words (and numbers) with
the claim, widened with its neighbours up to a character budget.

Deterministic and model-free. No LLM calls, no DB access.
"""

from __future__ import annotations

import re

__all__ = ["best_excerpt", "split_sentences"]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOPWORD_TEXT = (
    "the and for with that this from are was were has have had not but its into "
    "than then they them their there which while what when where who whom will "
    "would can could should may might also such these those over under more most "
    "less about between after before because been being does did each other some "
    "any all only"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_ELLIPSIS = "…"


def split_sentences(text: str) -> list[str]:
    squashed = " ".join(text.split())
    return [s for s in _SENTENCE_END.split(squashed) if s]


def _terms(text: str) -> set[str]:
    return {
        word
        for word in _WORD.findall(text.lower())
        if (len(word) >= 3 and word not in _STOPWORDS) or any(ch.isdigit() for ch in word)
    }


def _score(sentence: str, focus: set[str]) -> int:
    terms = _terms(sentence) & focus
    # Numbers are usually what a claim turns on, so they count double.
    return sum(2 if any(ch.isdigit() for ch in term) else 1 for term in terms)


def best_excerpt(text: str, focus: str, *, max_chars: int) -> str:
    """
    The run of sentences in *text* most relevant to *focus*, at most
    *max_chars* long (plus ellipses where text was cut).
    """
    squashed = " ".join(text.split())
    if max_chars <= 0 or len(squashed) <= max_chars:
        return squashed

    sentences = split_sentences(squashed)
    focus_terms = _terms(focus)
    scores = [_score(s, focus_terms) for s in sentences]
    anchor = max(range(len(sentences)), key=lambda i: (scores[i], -i))

    if len(sentences[anchor]) >= max_chars:
        excerpt = _cut(sentences[anchor], max_chars)
        start, end = anchor, anchor
    else:
        start = end = anchor
        length = len(sentences[anchor])
        grew = True
        while grew:
            grew = False
            # After first, then before: the sentence after a claim usually
            # carries its qualification ("... but only below -15 °C").
            if end + 1 < len(sentences) and length + 1 + len(sentences[end + 1]) <= max_chars:
                end += 1
                length += 1 + len(sentences[end])
                grew = True
            if start > 0 and length + 1 + len(sentences[start - 1]) <= max_chars:
                start -= 1
                length += 1 + len(sentences[start])
                grew = True
        excerpt = " ".join(sentences[start : end + 1])

    if start > 0:
        excerpt = _ELLIPSIS + excerpt
    if end < len(sentences) - 1 or excerpt.endswith(_ELLIPSIS):
        excerpt = excerpt if excerpt.endswith(_ELLIPSIS) else excerpt + _ELLIPSIS
    return excerpt


def _cut(sentence: str, max_chars: int) -> str:
    cut = sentence[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_chars // 2 else cut).rstrip(" ,;:") + _ELLIPSIS
