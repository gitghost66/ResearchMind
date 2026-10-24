"""
Token pricing, used to turn token counts into the ``cost_usd`` column of
``llm_usage``.

Prices are USD per 1M tokens. The table below is a convenience default only —
an explicit price on :class:`~researchmind_agents.config.LLMConfig` always wins,
so the deployment's ``.env`` is the real source of truth (AGENTS.md rule 4).

Free-tier usage genuinely costs $0, so an unknown model resolves to zero rather
than to a guess. Cost is therefore never invented; it is either configured,
known, or zero.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelPricing", "DEFAULT_PRICING", "resolve_pricing", "compute_cost_usd"]


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens for one model."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0

    @property
    def is_free(self) -> bool:
        return self.input_per_1m == 0.0 and self.output_per_1m == 0.0


# Longest matching prefix wins, so "gemini-2.5-flash-lite" beats "gemini-2.5-flash".
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "gemini-2.0-flash-lite": ModelPricing(0.075, 0.30),
    "gemini-2.0-flash": ModelPricing(0.10, 0.40),
    "gemini-2.5-flash-lite": ModelPricing(0.10, 0.40),
    "gemini-2.5-flash": ModelPricing(0.30, 2.50),
    "gemini-2.5-pro": ModelPricing(1.25, 10.00),
}

FREE = ModelPricing(0.0, 0.0)


def resolve_pricing(
    model: str,
    *,
    input_per_1m: float = 0.0,
    output_per_1m: float = 0.0,
) -> ModelPricing:
    """
    Resolve pricing for *model*.

    An explicitly configured non-zero price wins. Otherwise the longest matching
    prefix in :data:`DEFAULT_PRICING` is used, falling back to free.
    """
    if input_per_1m or output_per_1m:
        return ModelPricing(input_per_1m, output_per_1m)

    name = model.lower()
    matches = [prefix for prefix in DEFAULT_PRICING if name.startswith(prefix)]
    if not matches:
        return FREE
    return DEFAULT_PRICING[max(matches, key=len)]


def compute_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    pricing: ModelPricing,
) -> float:
    """Cost in USD for a single call. Returns 0.0 for free-tier models."""
    if pricing.is_free:
        return 0.0
    cost = (
        prompt_tokens * pricing.input_per_1m + completion_tokens * pricing.output_per_1m
    ) / 1_000_000
    # Sub-cent calls are the norm; keep enough precision to sum thousands of them.
    return round(cost, 10)
