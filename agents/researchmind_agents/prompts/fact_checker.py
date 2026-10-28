"""
Fact checker prompt: judge each claim against evidence excerpts.

The model gives a stance per excerpt and a verdict per claim. It never gives a
confidence number: that is computed from the stances and the sources'
reliability (``reliability.claim_confidence``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from researchmind_agents.prompts.common import UNTRUSTED_DATA_RULE, wrap_untrusted

__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "PromptEvidence",
    "PromptClaim",
    "build_fact_check_prompt",
]

PROMPT_VERSION = "fact_checker/v1"

SYSTEM_PROMPT = (
    "You are the fact-checking agent of a research team. For each claim you read "
    "the evidence excerpts and decide, strictly from what they say, whether they "
    "support it. You do not use outside knowledge. You reply with JSON only.\n\n"
    + UNTRUSTED_DATA_RULE
)


@dataclass(frozen=True)
class PromptEvidence:
    label: str
    title: str
    domain: str
    # e.g. "official docs, 2024"; empty when nothing is known.
    details: str
    excerpt: str


@dataclass(frozen=True)
class PromptClaim:
    label: str
    text: str
    evidence: Sequence[PromptEvidence]


def build_fact_check_prompt(*, question: str, claims: Sequence[PromptClaim]) -> str:
    blocks = []
    for claim in claims:
        lines = [f"[{claim.label}] {claim.text}"]
        for item in claim.evidence:
            about = f"{item.title or item.domain} - {item.domain}"
            if item.details:
                about += f" ({item.details})"
            lines.append(f"  [{item.label}] {about}")
            lines.append(f"  {item.excerpt}")
        blocks.append("\n".join(lines))
    body = "\n\n".join(blocks)

    return f"""Research question:
{question}

Claims to check, each followed by its evidence:
{wrap_untrusted(body, label="claims and evidence")}

For every claim, judge each evidence excerpt listed under it, then give a verdict.

Stance of one excerpt toward the claim:
- "supports": the excerpt states the claim, or clearly entails it, under the
  same conditions.
- "contradicts": the excerpt states something incompatible with the claim (a
  different figure, the opposite conclusion, ...).
- "neutral": the excerpt is about something else, or too vague to tell.

Verdict for the claim:
- "supported": at least one excerpt clearly states the claim and none
  contradicts it.
- "weak": support is partial or indirect: hedged, narrower in scope, missing
  the claim's conditions, outdated, or only a vendor's marketing.
- "mixed": some excerpts support the claim and others contradict it.
- "unsupported": no excerpt supports the claim.

Compare numbers, units, dates, versions, and conditions exactly. A claim that
says more than its evidence is "weak", not "supported".

Return JSON: {{"checks": [{{"claim": "C1", "verdict": "...", "evidence": [{{"evidence": "E1", "stance": "..."}}], "reason": "..."}}]}}
Give one check per claim, list every evidence label shown under that claim, and
keep "reason" to one short sentence."""
