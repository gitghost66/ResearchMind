"""LLM provider abstraction for ResearchMind agents."""

from researchmind_agents.llm.base import (
    BaseLLMProvider,
    LLMError,
    LLMResponse,
    LLMUsage,
    RawCompletion,
    StructuredOutputError,
    TransientLLMError,
    UsageSink,
)
from researchmind_agents.llm.client import GeminiClient, GeminiProvider
from researchmind_agents.llm.embeddings import (
    DimensionMismatchError,
    EmbeddingResult,
    EmbeddingService,
)
from researchmind_agents.llm.factory import (
    PROVIDERS,
    available_providers,
    build_llm_provider,
)
from researchmind_agents.llm.openai_compatible import OpenAICompatibleProvider
from researchmind_agents.llm.pricing import (
    DEFAULT_PRICING,
    ModelPricing,
    compute_cost_usd,
    resolve_pricing,
)

__all__ = [
    # Base abstraction
    "BaseLLMProvider",
    "LLMResponse",
    "LLMUsage",
    "RawCompletion",
    "UsageSink",
    "LLMError",
    "TransientLLMError",
    "StructuredOutputError",
    # Providers
    "GeminiProvider",
    "GeminiClient",
    "OpenAICompatibleProvider",
    "build_llm_provider",
    "available_providers",
    "PROVIDERS",
    # Embeddings
    "EmbeddingService",
    "EmbeddingResult",
    "DimensionMismatchError",
    # Pricing
    "ModelPricing",
    "DEFAULT_PRICING",
    "resolve_pricing",
    "compute_cost_usd",
]
