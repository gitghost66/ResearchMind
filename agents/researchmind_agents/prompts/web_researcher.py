"""
Researcher prompts: choose which search results (or papers) are worth reading.

One prompt serves every researcher; what differs is the guidance on what to
prefer, keyed by research domain (``researchmind_agents.modes``).
"""

from __future__ import annotations

from collections.abc import Sequence

from researchmind_agents.prompts.common import UNTRUSTED_DATA_RULE, wrap_untrusted

__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "SELECTION_GUIDANCE",
    "build_selection_prompt",
    "build_paper_selection_prompt",
]

# v2 (M5): per-domain guidance, and each result's source type.
PROMPT_VERSION = "web_researcher/v2"

SYSTEM_PROMPT = (
    "You are a research agent on a research team. Given a sub-question and a list "
    "of search results, you choose the ones most worth reading in full. "
    "You reply with JSON only.\n\n" + UNTRUSTED_DATA_RULE
)

_GENERAL = """Prefer:
- pages likely to contain specific facts, figures, or first-hand evidence that
  answer the sub-question;
- primary and authoritative sources (official documentation, standards bodies,
  peer-reviewed or well-sourced publications) over aggregators and SEO pages;
- different sites, so the evidence is independent."""

SELECTION_GUIDANCE: dict[str, str] = {
    "web": _GENERAL,
    "technical": """You are the technical researcher: you need precise engineering facts.
Prefer:
- official documentation, specifications, standards, and release notes;
- benchmarks that state their setup (hardware, versions, dataset) and method;
- engineering write-ups from teams that run the system in production, and the
  project's own repository or changelog;
- different sites, so the evidence is independent.
Avoid listicles, vendor marketing without numbers, and undated tutorials.""",
    "market": """You are the market and cost researcher: you need pricing, adoption,
vendor lock-in, and talent facts.
Prefer:
- official pricing pages and published price lists;
- industry and analyst reports, and surveys that state their sample;
- reputable business and technology press; job-market or hiring data;
- different sites, so the evidence is independent.
Avoid affiliate reviews, coupon pages, and undated comparisons.""",
    "academic": """You are the academic researcher: you need scholarly evidence.
Prefer:
- peer-reviewed papers, then preprints from known archives;
- work that describes its method, data, and sample;
- surveys and meta-analyses for an overview; recent work over old;
- different groups of authors, so the evidence is independent.
Avoid press releases and blog summaries of papers when the paper is available.""",
}


def build_selection_prompt(
    *,
    sub_question: str,
    results: Sequence[tuple[str, ...]],
    max_selected: int,
    guidance: str | None = None,
) -> str:
    """
    *results* holds ``(label, title, domain, snippet)`` tuples, optionally with
    a fifth item: the site's source type (``official_docs``, ``forum``, ...).

    Only labels are asked for, so the model never copies URLs.
    """
    entries = []
    for result in results:
        label, title, domain, snippet = result[:4]
        site = f"site: {domain}"
        if len(result) > 4 and result[4] and result[4] != "unknown":
            site += f" ({result[4].replace('_', ' ')})"
        entries.append(f"[{label}] {title}\n{site}\n{snippet}")
    listing = "\n\n".join(entries)
    return f"""Sub-question:
{sub_question}

Search results:
{wrap_untrusted(listing, label="search results")}

Choose up to {max_selected} results to read in full, most useful first.

{guidance or _GENERAL}

Skip results that are off-topic, paywalled-looking, forum noise, or near
duplicates of a better result.

Return JSON: {{"selected": ["R3", "R1", ...]}} using only labels from the list."""


def build_paper_selection_prompt(
    *,
    sub_question: str,
    papers: Sequence[tuple[str, str, str, str]],
    max_selected: int,
) -> str:
    """
    *papers* holds ``(label, title, details, abstract excerpt)`` tuples, where
    details is a short line such as "Nature, 2023, 412 citations".
    """
    listing = "\n\n".join(
        f"[{label}] {title}\n{details}\n{excerpt}" for label, title, details, excerpt in papers
    )
    return f"""Sub-question:
{sub_question}

Papers found:
{wrap_untrusted(listing, label="papers")}

Choose up to {max_selected} papers to read, most useful first.

{SELECTION_GUIDANCE["academic"]}

Skip papers that are off-topic or only mention the subject in passing.

Return JSON: {{"selected": ["R3", "R1", ...]}} using only labels from the list."""
