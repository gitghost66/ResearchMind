"""Claim extractor prompt: atomic claims from passages, each tied to its passages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from researchmind_agents.prompts.common import UNTRUSTED_DATA_RULE, wrap_untrusted

__all__ = ["PROMPT_VERSION", "SYSTEM_PROMPT", "PromptPassage", "build_extraction_prompt"]

PROMPT_VERSION = "claim_extractor/v1"

SYSTEM_PROMPT = (
    "You are the claim extraction agent of a research team. You read source "
    "passages and list the atomic factual claims they make that help answer a "
    "sub-question. You never add a fact that is not in the passages. "
    "You reply with JSON only.\n\n" + UNTRUSTED_DATA_RULE
)


@dataclass(frozen=True)
class PromptPassage:
    """A passage as the model sees it: a short label, never a UUID or URL."""

    label: str
    title: str
    domain: str
    text: str


def build_extraction_prompt(
    *,
    question: str,
    sub_question: str,
    passages: Sequence[PromptPassage],
    max_claims: int,
) -> str:
    blocks = "\n\n".join(
        f"[{p.label}] {p.title or p.domain}\nsite: {p.domain}\n{p.text}" for p in passages
    )
    labels = ", ".join(p.label for p in passages)
    return f"""Research question:
{question}

Sub-question:
{sub_question}

Passages (labels: {labels}):
{wrap_untrusted(blocks, label="passages")}

List up to {max_claims} claims from these passages that help answer the sub-question,
most informative first.

Each claim is {{"text": "...", "passages": ["P1", ...]}}:
- "text" is one atomic, self-contained factual statement: a single fact that is
  understandable without the passage. Name the subject; never write "it", "this
  study", or "the author".
- Keep numbers, units, versions, dates, and conditions exactly as the passage
  states them ("at -15 °C", "in 2023", "for tables under 1M rows").
- Attribute opinions, forecasts, and vendor statements to whoever makes them
  ("PostgreSQL's documentation states ...", "Vendor X claims ...").
- "passages" lists every passage that states the claim. Use only these labels:
  {labels}.
- Skip navigation text, advertising, and anything off the sub-question.
- Do not repeat a fact: merge claims that state the same fact and list all of
  their passages.

Return JSON: {{"claims": [{{"text": "...", "passages": ["P1"]}}]}}"""
