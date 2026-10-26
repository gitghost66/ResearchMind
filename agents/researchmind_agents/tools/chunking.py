"""
Text chunking and content hashing for extracted pages.

Token counts are estimated at ~4 characters per token. That is close enough
for sizing chunks well under the embedding model's input limit, and avoids a
tokenizer dependency (AGENTS.md rule 8).

Chunks follow sentence boundaries where possible; a sentence longer than a
whole chunk is split on word boundaries. Consecutive chunks share roughly
``overlap_tokens`` of trailing sentences, so a fact straddling a boundary is
still retrievable from one chunk.
"""

from __future__ import annotations

import hashlib
import math
import re

__all__ = ["CHARS_PER_TOKEN", "estimate_tokens", "chunk_text", "content_hash"]

CHARS_PER_TOKEN = 4

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_WHITESPACE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Approximate token count of *text*."""
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


def content_hash(text: str) -> str:
    """
    SHA-256 of the normalised text, for spotting the same page under two URLs.

    Case and whitespace are ignored, so trivial re-rendering differences still
    match.
    """
    normalised = _WHITESPACE.sub(" ", text).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def chunk_text(
    text: str,
    *,
    chunk_tokens: int,
    overlap_tokens: int = 0,
    max_chunks: int | None = None,
) -> list[str]:
    """
    Split *text* into chunks of at most ~*chunk_tokens* estimated tokens.

    Returns at most *max_chunks* chunks (all of them when ``None``), in
    document order. Empty or whitespace-only text yields an empty list.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    overlap_tokens = max(0, min(overlap_tokens, chunk_tokens // 2))
    limit = max_chunks if max_chunks is not None else math.inf

    units = _units(text, chunk_tokens)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > chunk_tokens:
            chunks.append(" ".join(current))
            if len(chunks) >= limit:
                return chunks
            current = _overlap_tail(current, overlap_tokens)
            current_tokens = sum(estimate_tokens(u) for u in current)
            # The carried tail plus this unit may still be too big; drop the tail.
            if current_tokens + unit_tokens > chunk_tokens:
                current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit_tokens

    if current and len(chunks) < limit:
        chunks.append(" ".join(current))
    return chunks


def _units(text: str, chunk_tokens: int) -> list[str]:
    """Sentences, each guaranteed to fit in one chunk."""
    max_chars = chunk_tokens * CHARS_PER_TOKEN
    units: list[str] = []
    for paragraph in text.splitlines():
        paragraph = _WHITESPACE.sub(" ", paragraph).strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_BOUNDARY.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                units.extend(_split_words(sentence, max_chars))
    return units


def _split_words(sentence: str, max_chars: int) -> list[str]:
    """Split an over-long sentence on word boundaries (or hard, for one huge word)."""
    pieces: list[str] = []
    current = ""
    for word in sentence.split(" "):
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(units: list[str], overlap_tokens: int) -> list[str]:
    """Trailing units whose combined size stays within *overlap_tokens*."""
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        size = estimate_tokens(unit)
        if total + size > overlap_tokens:
            break
        tail.insert(0, unit)
        total += size
    return tail
