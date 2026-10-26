"""Synthesizer prompt: write a cited report from retrieved passages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from researchmind_agents.prompts.common import UNTRUSTED_DATA_RULE, wrap_untrusted

__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "PromptSource",
    "build_synthesis_prompt",
    "build_citation_retry_prompt",
]

PROMPT_VERSION = "synthesizer/v1"

SYSTEM_PROMPT = (
    "You are the synthesis agent of a research team. You write a research report "
    "using only the numbered sources you are given. Every factual sentence must be "
    "supported by at least one source and cite it by label. If the sources do not "
    "answer something, say so in the limitations instead of filling the gap from "
    "memory. You reply with JSON only.\n\n" + UNTRUSTED_DATA_RULE
)


@dataclass(frozen=True)
class PromptSource:
    """A source as the model sees it: a short label, never a UUID or URL."""

    label: str
    title: str
    domain: str
    passages: Sequence[str]


def build_synthesis_prompt(
    *,
    question: str,
    objective: str,
    sub_questions: Sequence[tuple[str, bool]],
    sources: Sequence[PromptSource],
) -> str:
    """
    *sub_questions* is a list of ``(text, has_evidence)`` pairs, in plan order.
    """
    plan_lines = "\n".join(
        f"{i}. {text}" + ("" if has_evidence else "  (no sources found)")
        for i, (text, has_evidence) in enumerate(sub_questions, start=1)
    )
    source_blocks = "\n\n".join(_render_source(source) for source in sources)
    labels = ", ".join(source.label for source in sources)

    return f"""Research question:
{question}

Objective: {objective}

Sub-questions:
{plan_lines}

Sources (labels: {labels}):
{wrap_untrusted(source_blocks, label="sources")}

Write the report as JSON with these keys:
- "title": a short, specific title.
- "executive_summary": 2 to 4 statements that directly answer the research question.
- "sections": one section per sub-question that has sources, in plan order. Use a
  short heading. Each section has 2 to 6 statements.
- "limitations": short plain sentences on what the sources could not establish,
  including any sub-question with no sources. No citations here.

Each statement is {{"text": "...", "citations": ["S1", ...]}}:
- "text" is one sentence. Do not put citation markers in the text.
- "citations" lists the labels of the sources that support that sentence. Use
  only these labels: {labels}. Every statement needs at least one.
- State only what the cited passages say. Keep figures, versions, and dates
  exactly as written in the sources. When sources disagree, say so and cite both.
"""


def build_citation_retry_prompt(
    original_prompt: str,
    *,
    unknown_labels: Sequence[str],
    uncited_statements: Sequence[str],
    allowed_labels: Sequence[str],
) -> str:
    """Follow-up prompt after the citation check failed."""
    problems = []
    if unknown_labels:
        problems.append(
            "These citation labels do not exist: " + ", ".join(sorted(set(unknown_labels)))
        )
    if uncited_statements:
        listed = "\n".join(f'- "{text[:200]}"' for text in uncited_statements[:10])
        problems.append(f"These statements have no citation:\n{listed}")

    return (
        f"{original_prompt}\n"
        "--- CITATION CHECK FAILED ---\n"
        + "\n".join(problems)
        + "\n\nRewrite the whole report. Cite only these labels: "
        + ", ".join(allowed_labels)
        + ". Give every statement at least one citation, or remove a statement no "
        "source supports. Reply with corrected JSON only."
    )


def _render_source(source: PromptSource) -> str:
    excerpts = "\n".join(f"- {text}" for text in source.passages)
    return f"[{source.label}] {source.title or source.domain}\nsite: {source.domain}\n{excerpts}"
