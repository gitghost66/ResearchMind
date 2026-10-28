"""
Deterministic source reliability and claim / report confidence (PRD s8).

No model is called here. Every score in a report comes from these formulas,
so the same evidence always gets the same numbers. The weights are
``ScoringConfig`` fields (``config.py``), set from ``Settings``; the
domain-type list is ``data/domains.json``.

Source reliability, 0-1 (:func:`source_reliability`)
----------------------------------------------------
::

    score = type_score(domain_type) * recency + citation_bonus + methodology_bonus

* ``domain_type`` — :func:`classify_domain`: the versioned list in
  ``domains.json`` (longest host match), then TLD rules (``.gov``, ``.edu``,
  ...), host prefixes (``docs.``, ``blog.``), and path prefixes (``/docs``,
  ``/pricing``, ``/blog``). Papers from scholarly search are typed by their
  metadata instead: ``peer_reviewed`` with a venue, else ``preprint``.
* ``recency`` — 1.0 up to ``recency_full_years`` old, then minus
  ``recency_penalty_per_year`` per extra year, never below ``recency_floor``;
  ``unknown_date_factor`` when the date is unknown. Age is measured from the
  run's start.
* ``citation_bonus`` (papers) — ``citation_bonus_max * log(1 + c) /
  log(1 + citation_saturation)``, capped at ``citation_bonus_max``.
* ``methodology_bonus`` (papers) — when the text describes data or methods
  (:func:`describes_methodology`).

The result is clamped to [0, 1].

Claim confidence, 0-1 (:func:`claim_confidence`)
------------------------------------------------
::

    confidence = verdict_weight * corroboration * agreement

* Independence: each site (host without ``www.``) counts once, with the
  reliability of its best source. Two pages from one site are one voice.
* ``corroboration`` — noisy-OR over supporting sites:
  ``1 - prod(1 - source_weight * r_i)``. One source establishes at most
  ``source_weight * r`` (0.8 * r by default); each further independent
  source closes part of the remaining gap, so agreement compounds.
* ``agreement`` — supporting reliability / (supporting + contradicting
  reliability), summed over sites.
* ``verdict_weight`` — from ``ScoringConfig.verdict_weights`` (supported 1.0,
  weak 0.6, mixed 0.35, unsupported and unverified 0).
* No supporting evidence gives 0.

The verdict itself must agree with the per-passage stances
(:func:`reconcile_verdict`): no supporting passage means ``unsupported``, and
any contradicting passage means ``mixed``.

Overall report confidence, 0-1 or None (:func:`overall_confidence`)
-------------------------------------------------------------------
::

    base = sum(w_c * confidence_c) / sum(w_c)   over the claims the report cites,
                                                w_c = statements citing claim c
    overall = base * (1 - uncovered_penalty * uncovered_share)
                   * (1 - min(contradiction_penalty_cap,
                              contradiction_penalty * unresolved_contradictions))

``uncovered_share`` is the share of sub-questions with no usable claim (a
gap-filling sub-question's claims count toward the sub-question it fills).
``unresolved_contradictions`` counts the report's contradictions whose likely
cause is not one of ``ScoringConfig.resolved_causes`` (by default conditions,
date, version, dataset, definition: the claims differ in scope, so both can
hold). An ``unexplained`` or ``methodology`` conflict is unresolved.
``None`` when the report cites no claims (quick mode, or nothing verified).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from researchmind_agents.config import ScoringConfig

__all__ = [
    "SCORING_VERSION",
    "DOMAIN_TYPES",
    "PAPER_TYPES",
    "DomainRules",
    "load_domain_rules",
    "classify_domain",
    "SourceSignals",
    "recency_factor",
    "citation_bonus",
    "describes_methodology",
    "source_reliability",
    "EvidenceSignal",
    "reconcile_verdict",
    "claim_confidence",
    "overall_confidence",
]

# Bump when a formula changes, so stored scores can be told apart.
SCORING_VERSION = "scoring/v1"

DOMAIN_TYPES: tuple[str, ...] = (
    "peer_reviewed",
    "standards_body",
    "official_docs",
    "government",
    "academic",
    "preprint",
    "news",
    "reference",
    "code_repository",
    "company_blog",
    "blog",
    "forum",
    "unknown",
)

# Types for which citation and methodology bonuses apply.
PAPER_TYPES: frozenset[str] = frozenset({"peer_reviewed", "preprint", "academic"})

_DOMAINS_PATH = Path(__file__).with_name("data") / "domains.json"


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainRules:
    """The parsed ``domains.json``."""

    version: str
    hosts: Mapping[str, str]
    suffix_rules: tuple[tuple[str, str], ...]
    host_prefixes: tuple[tuple[str, str], ...]
    path_prefixes: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DomainRules:
        hosts: dict[str, str] = {}
        for domain_type, entries in dict(data.get("types") or {}).items():
            if domain_type not in DOMAIN_TYPES:
                raise ValueError(f"unknown domain type {domain_type!r} in domain rules")
            for entry in entries:
                hosts[str(entry).lower().strip()] = domain_type

        def pairs(key: str) -> tuple[tuple[str, str], ...]:
            result = []
            for match, domain_type in data.get(key) or []:
                if domain_type not in DOMAIN_TYPES:
                    raise ValueError(f"unknown domain type {domain_type!r} in {key}")
                result.append((str(match).lower(), str(domain_type)))
            # Longest first, so ".gov.uk" is tried before ".uk"-style rules.
            return tuple(sorted(result, key=lambda pair: -len(pair[0])))

        return cls(
            version=str(data.get("version") or "unversioned"),
            hosts=hosts,
            suffix_rules=pairs("suffix_rules"),
            host_prefixes=pairs("host_prefixes"),
            path_prefixes=pairs("path_prefixes"),
        )


@lru_cache(maxsize=1)
def load_domain_rules() -> DomainRules:
    """The packaged domain list (read once)."""
    return DomainRules.from_dict(json.loads(_DOMAINS_PATH.read_text(encoding="utf-8")))


def _host_and_path(url: str) -> tuple[str, str]:
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return "", ""
    if host.startswith("www."):
        host = host[4:]
    return host, (parts.path or "/").lower()


def _path_type(path: str, rules: DomainRules) -> str | None:
    for prefix, domain_type in rules.path_prefixes:
        if path == prefix or path.startswith(prefix + "/"):
            return domain_type
    return None


def classify_domain(url: str, rules: DomainRules | None = None) -> str:
    """
    The domain type of *url*: listed host, else TLD rule, else host prefix,
    else path prefix, else ``"unknown"``.

    A listed documentation site still yields to a blog or forum path, so
    ``aws.amazon.com/blogs/...`` is not scored as official documentation.
    """
    rules = rules or load_domain_rules()
    host, path = _host_and_path(url)
    if not host:
        return "unknown"

    listed = _listed_type(host, rules)
    if listed is not None:
        if listed == "official_docs":
            by_path = _path_type(path, rules)
            if by_path in {"company_blog", "forum"}:
                return by_path
        return listed

    for suffix, domain_type in rules.suffix_rules:
        # "gov.uk" itself matches ".gov.uk", not only its subdomains.
        if host.endswith(suffix) or host == suffix.lstrip("."):
            return domain_type
    for prefix, domain_type in rules.host_prefixes:
        if host.startswith(prefix):
            return domain_type
    return _path_type(path, rules) or "unknown"


def _listed_type(host: str, rules: DomainRules) -> str | None:
    """Longest listed domain that *host* equals or is a subdomain of."""
    labels = host.split(".")
    for start in range(len(labels) - 1):
        candidate = ".".join(labels[start:])
        if candidate in rules.hosts:
            return rules.hosts[candidate]
    return None


# ---------------------------------------------------------------------------
# Source reliability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSignals:
    """What reliability is computed from."""

    domain_type: str = "unknown"
    published_at: datetime | None = None
    # Papers only.
    citation_count: int | None = None
    has_methodology: bool = False


def recency_factor(
    published_at: datetime | None, *, now: datetime, config: ScoringConfig
) -> float:
    """Multiplier for a source's age (see the module docstring)."""
    if published_at is None:
        return config.unknown_date_factor
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    age_years = max(0.0, (now - published_at).total_seconds() / (365.25 * 86400))
    if age_years <= config.recency_full_years:
        return 1.0
    penalty = config.recency_penalty_per_year * (age_years - config.recency_full_years)
    return max(config.recency_floor, 1.0 - penalty)


def citation_bonus(citation_count: int | None, config: ScoringConfig) -> float:
    if not citation_count or citation_count <= 0 or config.citation_bonus_max <= 0:
        return 0.0
    saturation = max(1, config.citation_saturation)
    share = math.log1p(citation_count) / math.log1p(saturation)
    return config.citation_bonus_max * min(1.0, share)


# Phrases a paper uses when it reports its data or method. Two distinct hits
# are required, so one passing mention does not count.
_METHOD_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bmethod(?:s|ology)?\b",
        r"\bdata ?sets?\b",
        r"\bwe (?:evaluate|measure|propose|conduct|train|collect|analy[sz]e)\b",
        r"\bexperiments?\b",
        r"\bparticipants\b",
        r"\bsample (?:size|of)\b",
        r"\brandomi[sz]ed\b",
        r"\bcohort\b",
        r"\bbaselines?\b",
        r"\bablation\b",
        r"\bstatistically significant\b",
        r"\bconfidence intervals?\b",
        r"\bbenchmarks?\b",
    )
)


def describes_methodology(text: str) -> bool:
    """True when *text* names at least two distinct markers of data or method."""
    hits = sum(1 for pattern in _METHOD_PATTERNS if pattern.search(text))
    return hits >= 2


def source_reliability(
    signals: SourceSignals, *, now: datetime, config: ScoringConfig
) -> float:
    """A source's reliability score in [0, 1] (see the module docstring)."""
    score = config.type_score(signals.domain_type) * recency_factor(
        signals.published_at, now=now, config=config
    )
    if signals.domain_type in PAPER_TYPES:
        score += citation_bonus(signals.citation_count, config)
        if signals.has_methodology:
            score += config.methodology_bonus
    return round(_clamp(score), 4)


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceSignal:
    """One judged passage: its source's site and reliability, and its stance."""

    site: str
    reliability: float
    stance: str


def reconcile_verdict(verdict: str, stances: Iterable[str]) -> str:
    """
    Make the fact checker's verdict agree with its own per-passage stances.

    * no supporting passage -> ``unsupported``;
    * a supporting and a contradicting passage -> ``mixed``;
    * support only -> the model's ``supported``/``weak``, else ``weak``
      (a model that saw support but said "unsupported" or "mixed" is
      contradicting itself, so the claim gets the cautious verdict).
    """
    stances = list(stances)
    supports = stances.count("supports")
    contradicts = stances.count("contradicts")
    if supports == 0:
        return "unsupported"
    if contradicts:
        return "mixed"
    return verdict if verdict in {"supported", "weak"} else "weak"


def _best_per_site(evidence: Iterable[EvidenceSignal], stance: str) -> dict[str, float]:
    best: dict[str, float] = {}
    for item in evidence:
        if item.stance != stance:
            continue
        site = item.site or "?"
        best[site] = max(best.get(site, 0.0), _clamp(item.reliability))
    return best


def claim_confidence(
    verdict: str, evidence: Sequence[EvidenceSignal], *, config: ScoringConfig
) -> float:
    """A claim's confidence in [0, 1] (see the module docstring)."""
    weight = config.verdict_weight(verdict)
    supporting = _best_per_site(evidence, "supports")
    if weight <= 0 or not supporting:
        return 0.0
    contradicting = _best_per_site(evidence, "contradicts")

    corroboration = 1.0 - math.prod(1.0 - config.source_weight * r for r in supporting.values())

    support_total = sum(supporting.values())
    against_total = sum(contradicting.values())
    agreement = support_total / (support_total + against_total) if support_total else 0.0

    return round(_clamp(weight * corroboration * agreement), 4)


def overall_confidence(
    cited: Sequence[tuple[float, int]],
    *,
    sub_questions: int,
    covered_sub_questions: int,
    unresolved_contradictions: int = 0,
    config: ScoringConfig,
) -> float | None:
    """
    The report's confidence (see the module docstring).

    *cited* holds ``(claim_confidence, statements_citing_it)`` for each claim
    the report cites. Returns ``None`` when it is empty.
    """
    weighted = [(c, w) for c, w in cited if w > 0]
    if not weighted:
        return None
    base = sum(c * w for c, w in weighted) / sum(w for _, w in weighted)

    uncovered_share = 0.0
    if sub_questions > 0:
        covered = min(max(covered_sub_questions, 0), sub_questions)
        uncovered_share = 1.0 - covered / sub_questions
    base *= 1.0 - config.uncovered_penalty * uncovered_share

    penalty = min(
        config.contradiction_penalty_cap,
        config.contradiction_penalty * max(0, unresolved_contradictions),
    )
    base *= 1.0 - penalty
    return round(_clamp(base), 4)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
