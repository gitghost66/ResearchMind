"""
Sufficiency prompt: do the verified claims answer the question, and if not,
what is missing?

Only asked when the deterministic rules found every sub-question covered
(``nodes/sufficiency.py``); the model's job is the judgement the rules cannot
make — whether the covered sub-questions add up to an answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from researchmind_agents.prompts.common import UNTRUSTED_DATA_RULE, wrap_untrusted

__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "PromptCoverage",
    "build_sufficiency_prompt",
]

PROMPT_VERSION = "sufficiency/v1"

SYSTEM_PROMPT = (
    "You are the coverage reviewer of a research team. You judge whether the "
    "verified claims gathered so far answer the research question, and if not, what "
    "is still missing. You do not answer the question yourself. You reply with JSON "
    "only.\n\n" + UNTRUSTED_DATA_RULE
)


@dataclass(frozen=True)
class PromptCoverage:
    """One top-level sub-question and what has been verified for it."""

    position: int
    text: str
    sites: int
    # (claim text, verdict), best first; possibly a sample.
    claims: Sequence[tuple[str, str]]
    total_claims: int


def build_sufficiency_prompt(
    *,
    question: str,
    objective: str,
    coverage: Sequence[PromptCoverage],
    contradictions: Sequence[str],
    max_gaps: int,
) -> str:
    """*contradictions* are short descriptions of unresolved conflicts."""
    blocks = []
    for item in coverage:
        claims = f"{item.total_claims} verified claim" + ("" if item.total_claims == 1 else "s")
        sites = f"{item.sites} site" + ("" if item.sites == 1 else "s")
        lines = [f"{item.position}. {item.text} ({claims} from {sites})"]
        lines += [f"   - ({verdict}) {text}" for text, verdict in item.claims]
        if item.total_claims > len(item.claims):
            lines.append(f"   - ... and {item.total_claims - len(item.claims)} more")
        blocks.append("\n".join(lines))
    if contradictions:
        blocks.append(
            "Unresolved contradictions:\n" + "\n".join(f"   - {text}" for text in contradictions)
        )
    body = "\n".join(blocks)

    return f"""Research question:
{question}

Objective: {objective}

Sub-questions and the claims verified for them:
{wrap_untrusted(body, label="verified claims")}

Decide whether these claims are enough to answer the research question well.

- "sufficient": true when every important part of the question is answered by
  verified claims; false when an important part is missing, or rests only on
  weak or disputed claims that more research could settle.
- "gaps": when not sufficient, up to {max_gaps} short descriptions of what is
  missing, each specific enough to search for (for example "2025 list prices for
  the managed tier"). Empty when sufficient.
- "reason": one short sentence.

Return JSON: {{"sufficient": true, "gaps": [], "reason": "..."}}"""
