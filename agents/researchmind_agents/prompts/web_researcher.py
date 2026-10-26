"""Web researcher prompt: choose which search results are worth reading."""

from __future__ import annotations

from researchmind_agents.prompts.common import UNTRUSTED_DATA_RULE, wrap_untrusted

__all__ = ["PROMPT_VERSION", "SYSTEM_PROMPT", "build_selection_prompt"]

PROMPT_VERSION = "web_researcher/v1"

SYSTEM_PROMPT = (
    "You are the web research agent of a research team. Given a sub-question and "
    "a list of search results, you choose the pages most worth reading in full. "
    "You reply with JSON only.\n\n" + UNTRUSTED_DATA_RULE
)


def build_selection_prompt(
    *,
    sub_question: str,
    results: list[tuple[str, str, str, str]],
    max_selected: int,
) -> str:
    """
    *results* is a list of ``(label, title, domain, snippet)`` tuples.

    Only labels are asked for, so the model never copies URLs.
    """
    listing = "\n\n".join(
        f"[{label}] {title}\nsite: {domain}\n{snippet}" for label, title, domain, snippet in results
    )
    return f"""Sub-question:
{sub_question}

Search results:
{wrap_untrusted(listing, label="search results")}

Choose up to {max_selected} results to read in full, most useful first.

Prefer:
- pages likely to contain specific facts, figures, or first-hand evidence that
  answer the sub-question;
- primary and authoritative sources (official documentation, standards bodies,
  peer-reviewed or well-sourced publications) over aggregators and SEO pages;
- different sites, so the evidence is independent.

Skip results that are off-topic, paywalled-looking, forum noise, or near
duplicates of a better result.

Return JSON: {{"selected": ["R3", "R1", ...]}} using only labels from the list."""
