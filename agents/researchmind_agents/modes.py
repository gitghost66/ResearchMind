"""
Research modes (PRD s6.4) and the research domains each one may use.

A *domain* names the researcher that handles a sub-question: the planner tags
every sub-question with one, and the graph sends it to that researcher. A mode
fixes which domains the planner may choose from, which it should favour,
whether claims are verified (extracted, fact-checked, and checked for
contradictions) before the report is written, whether the gap-filling loop
runs, and whether the report must end in a recommendation.

=========  ==================================  ========  ======  ====  ==============
Mode       Domains                             Favoured  Verify  Loop  Recommendation
=========  ==================================  ========  ======  ====  ==============
quick      web                                 —         no      no    no
deep       web, technical, market              —         yes     yes   no
academic   web, technical, market, academic    academic  yes     yes   no
decision   web, technical, market, academic    market    yes     yes   yes
=========  ==================================  ========  ======  ====  ==============

A looping mode runs up to ``max_iterations`` research rounds (quick mode always
runs one). Decision mode offers the academic researcher, but only a
sub-question the planner tags ``academic`` reaches it.

This module imports nothing from the package, so anything may import it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "WEB",
    "TECHNICAL",
    "MARKET",
    "ACADEMIC",
    "RESEARCH_DOMAINS",
    "DOMAIN_DESCRIPTIONS",
    "ModeProfile",
    "MODES",
    "SUPPORTED_MODES",
    "mode_profile",
    "normalize_domain",
]

WEB = "web"
TECHNICAL = "technical"
MARKET = "market"
ACADEMIC = "academic"

RESEARCH_DOMAINS: tuple[str, ...] = (WEB, TECHNICAL, MARKET, ACADEMIC)

# How the planner is told to choose (prompt text lives with the domains it
# describes, so adding a domain is one edit).
DOMAIN_DESCRIPTIONS: dict[str, str] = {
    WEB: "general web research: overviews, news, comparisons, anything not covered below",
    TECHNICAL: "documentation, specifications, benchmarks, engineering write-ups",
    MARKET: "pricing, cost, adoption, vendors and lock-in, hiring and talent availability",
    ACADEMIC: "peer-reviewed papers and preprints: findings, methodology, datasets",
}

_ALIASES: dict[str, str] = {
    "general": WEB,
    "tech": TECHNICAL,
    "engineering": TECHNICAL,
    "cost": MARKET,
    "market/cost": MARKET,
    "business": MARKET,
    "papers": ACADEMIC,
    "research": ACADEMIC,
    "scholarly": ACADEMIC,
}


@dataclass(frozen=True)
class ModeProfile:
    name: str
    # Domains the planner may assign; the first is the default.
    domains: tuple[str, ...]
    # Domains the planner should favour; at least one sub-question gets the first.
    favoured: tuple[str, ...] = ()
    # Extract claims, fact-check them, and look for contradictions before
    # writing the report.
    verify: bool = False
    # Check sufficiency after each round and research the gaps, up to the
    # run's max_iterations. Requires verify (coverage is counted in claims).
    loop: bool = False
    # The report must include a recommendation section.
    recommendation: bool = False
    # PRD s6.4 latency target (seconds, p50). Reported by the seed eval; the
    # loop's own deadline is the LOOP_DEADLINE_SECONDS setting.
    target_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.loop and not self.verify:
            raise ValueError(f"mode {self.name!r}: the gap-filling loop needs verified claims")

    @property
    def default_domain(self) -> str:
        return self.domains[0]


MODES: dict[str, ModeProfile] = {
    "quick": ModeProfile(name="quick", domains=(WEB,), target_seconds=45.0),
    "deep": ModeProfile(
        name="deep", domains=(WEB, TECHNICAL, MARKET), verify=True, loop=True, target_seconds=300.0
    ),
    "academic": ModeProfile(
        name="academic",
        domains=(WEB, TECHNICAL, MARKET, ACADEMIC),
        favoured=(ACADEMIC,),
        verify=True,
        loop=True,
        target_seconds=360.0,
    ),
    "decision": ModeProfile(
        name="decision",
        domains=(WEB, TECHNICAL, MARKET, ACADEMIC),
        favoured=(MARKET,),
        verify=True,
        loop=True,
        recommendation=True,
        target_seconds=300.0,
    ),
}

SUPPORTED_MODES: frozenset[str] = frozenset(MODES)


def mode_profile(mode: str) -> ModeProfile:
    """The profile for *mode*; raises ``ValueError`` for an unknown mode."""
    try:
        return MODES[mode]
    except KeyError:
        raise ValueError(
            f"Mode {mode!r} is not available yet; supported: {', '.join(sorted(MODES))}"
        ) from None


def normalize_domain(raw: str, profile: ModeProfile) -> str:
    """Map the planner's domain to one the mode allows (else the mode's default)."""
    value = " ".join(str(raw).lower().split())
    value = _ALIASES.get(value, value)
    return value if value in profile.domains else profile.default_domain
