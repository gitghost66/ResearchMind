"""
Contradiction detector prompt: find conflicting claims within topic groups,
and say why their sources disagree.

The groups come from code (similar claims, or claims the fact checker found
in conflict), so the model compares a handful of claims at a time. Claims are
labels (``C1``, ``C2``, ...), never IDs. Whether a contradiction is resolved is
decided by code from the cause the model names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from researchmind_agents.prompts.common import UNTRUSTED_DATA_RULE, wrap_untrusted

__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "CAUSE_DESCRIPTIONS",
    "PromptExcerpt",
    "PromptClaim",
    "PromptGroup",
    "build_contradiction_prompt",
]

PROMPT_VERSION = "contradiction_detector/v1"

SYSTEM_PROMPT = (
    "You are the contradiction-detection agent of a research team. You compare "
    "fact-checked claims on the same topic, find claims that cannot all be true as "
    "stated, and explain why their sources disagree. You use only the claims and "
    "excerpts you are given, never outside knowledge. You reply with JSON only.\n\n"
    + UNTRUSTED_DATA_RULE
)

# One per ``state.ContradictionCause``.
CAUSE_DESCRIPTIONS: dict[str, str] = {
    "conditions": "measured or stated under different conditions (workload, scale, "
    "climate, configuration, region, ...)",
    "date": "the sources are from different times, and the newer one reflects a change",
    "version": "about different versions, editions, or releases",
    "dataset": "based on different datasets, benchmarks, or samples",
    "methodology": "different methods of measuring or studying it (for example a vendor "
    "benchmark against an independent test)",
    "definition": "a term is defined or counted differently",
    "unexplained": "nothing in the evidence explains the difference; the sources simply disagree",
}


@dataclass(frozen=True)
class PromptExcerpt:
    title: str
    domain: str
    # e.g. "official docs, 2024-03-01"; empty when nothing is known.
    details: str
    text: str


@dataclass(frozen=True)
class PromptClaim:
    label: str
    text: str
    verdict: str
    excerpts: Sequence[PromptExcerpt]


@dataclass(frozen=True)
class PromptGroup:
    label: str
    claims: Sequence[PromptClaim]


def build_contradiction_prompt(*, question: str, groups: Sequence[PromptGroup]) -> str:
    blocks = []
    for group in groups:
        lines = [f"Topic {group.label}:"]
        for claim in group.claims:
            lines.append(f"[{claim.label}] ({claim.verdict}) {claim.text}")
            for excerpt in claim.excerpts:
                about = f"{excerpt.title or excerpt.domain} - {excerpt.domain}"
                if excerpt.details:
                    about += f" ({excerpt.details})"
                lines.append(f"  source: {about}")
                lines.append(f"  {excerpt.text}")
        blocks.append("\n".join(lines))
    body = "\n\n".join(blocks)
    causes = "\n".join(f'  - "{name}": {what}' for name, what in CAUSE_DESCRIPTIONS.items())

    return f"""Research question:
{question}

Claims grouped by topic, each with the evidence it rests on:
{wrap_untrusted(body, label="claims and evidence")}

Within each topic, find claims that contradict each other: they cannot all be true
as stated (a different figure for the same measure, opposite conclusions,
incompatible facts). Claims about different things, or that only add detail or
narrow the scope, do not contradict. Compare numbers, units, dates, versions, and
conditions exactly.

For each contradiction give:
- "claims": the labels of the conflicting claims, two or more, all from one topic.
- "likely_cause": why the sources disagree, one of:
{causes}
  Choose the cause the excerpts show (a date, version, condition, dataset, method,
  or definition they name). Use "unexplained" when they show none.
- "explanation": one or two sentences saying what differs and naming the detail
  that explains it.

Return JSON: {{"contradictions": [{{"claims": ["C1", "C2"], "likely_cause": "...", "explanation": "..."}}]}}
Return {{"contradictions": []}} when no claims conflict."""
