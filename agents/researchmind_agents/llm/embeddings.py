"""
Embedding service (Gemini ``text-embedding-004`` by default, 768-dimensional).

Three things matter here beyond calling the API:

* **Real batching** — one request per ``batch_size`` texts, not one per text
  (PRD s10: "embeddings batched").
* **Bounded concurrency** — a free tier dies instantly under an unbounded
  ``gather`` over hundreds of texts.
* **Retries** — a single 429 must not end a research run.

No HTTP server code, no DB sessions — pure agent-layer logic.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from google.genai import types as genai_types

from researchmind_agents.config import EmbeddingConfig
from researchmind_agents.llm.base import retry_async
from researchmind_agents.llm.google_client import shared_genai_client

logger = logging.getLogger(__name__)

__all__ = ["EmbeddingResult", "EmbeddingService", "DimensionMismatchError"]

_TASK_TYPE_RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
_TASK_TYPE_RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


class DimensionMismatchError(ValueError):
    """Raised when a returned vector does not match the configured dimension."""


@dataclass
class EmbeddingResult:
    """Result for a single embedded text."""

    vector: list[float] = field(default_factory=list)
    model: str = ""
    token_count: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.vector)


class EmbeddingService:
    """
    Async wrapper around google-genai's ``embed_content`` API.

    Usage::

        svc = EmbeddingService(EmbeddingConfig(api_key="...", dimension=768))

        result = await svc.embed_query("What is RAG?")
        results = await svc.embed_documents(["doc1...", "doc2..."])

    ``embed_documents`` returns one result per input text, in order. A batch
    that fails after all retries yields results with ``error`` set rather than
    raising, so a partially embedded run can still proceed.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        client: Any | None = None,
    ) -> None:
        self._config = config or EmbeddingConfig()
        self._explicit_client = client
        self._lazy_client: Any | None = None
        self._semaphore = asyncio.Semaphore(max(1, self._config.max_concurrency))

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def _client(self) -> Any:
        """
        The genai client, created on first use.

        Embeddings use their own provider key, which may be absent when the LLM
        runs elsewhere (OpenRouter, a local server). Constructing lazily means a
        runtime without an embedding key still builds, and the missing key is
        reported only if something actually tries to embed.
        """
        if self._explicit_client is not None:
            return self._explicit_client
        if self._lazy_client is None:
            self._lazy_client = shared_genai_client(self._config.api_key)
        return self._lazy_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single query string (``RETRIEVAL_QUERY`` task type)."""
        results = await self._embed_batch([text], task_type=_TASK_TYPE_RETRIEVAL_QUERY)
        return results[0]

    async def embed_document(self, text: str) -> EmbeddingResult:
        """Embed a single document (``RETRIEVAL_DOCUMENT`` task type)."""
        results = await self._embed_batch([text], task_type=_TASK_TYPE_RETRIEVAL_DOCUMENT)
        return results[0]

    async def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Embed many documents, batched and with bounded concurrency.

        Returns one result per input text, in input order.
        """
        if not texts:
            return []

        size = max(1, self._config.batch_size)
        batches = [texts[i : i + size] for i in range(0, len(texts), size)]

        async def run(batch: list[str]) -> list[EmbeddingResult]:
            async with self._semaphore:
                return await self._embed_batch(
                    batch, task_type=_TASK_TYPE_RETRIEVAL_DOCUMENT
                )

        grouped = await asyncio.gather(*(run(batch) for batch in batches))

        results: list[EmbeddingResult] = []
        for group in grouped:
            results.extend(group)

        failed = sum(1 for r in results if not r.ok)
        if failed:
            logger.warning("Embedded %d/%d texts (%d failed)", len(results) - failed, len(results), failed)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _embed_batch(
        self,
        texts: list[str],
        *,
        task_type: str,
    ) -> list[EmbeddingResult]:
        """Embed one batch in a single API call, with retries."""

        def call() -> Any:
            return self._client.models.embed_content(
                model=self._config.model,
                contents=texts,
                config=genai_types.EmbedContentConfig(task_type=task_type),
            )

        async def attempt() -> Any:
            return await asyncio.to_thread(call)

        try:
            response = await retry_async(
                attempt,
                max_attempts=self._config.max_attempts,
                base_delay=self._config.base_delay,
                description=f"embed:{self._config.model}",
            )
        except Exception as exc:  # noqa: BLE001 - a failed batch degrades, not crashes
            logger.error("Embedding batch of %d failed: %s", len(texts), exc)
            return [
                EmbeddingResult(model=self._config.model, error=str(exc)) for _ in texts
            ]

        return self._to_results(response, expected=len(texts))

    def _to_results(self, response: Any, *, expected: int) -> list[EmbeddingResult]:
        """Convert an ``EmbedContentResponse`` into per-text results."""
        embeddings = list(getattr(response, "embeddings", None) or [])
        token_count = _token_count(response)
        # The API reports tokens for the whole batch; spread it so the sum of
        # per-text counts still equals the batch total.
        per_text_tokens = token_count // expected if expected else 0

        results: list[EmbeddingResult] = []
        for index in range(expected):
            if index >= len(embeddings):
                results.append(
                    EmbeddingResult(
                        model=self._config.model,
                        error="provider returned fewer embeddings than inputs",
                    )
                )
                continue

            vector = [float(v) for v in (getattr(embeddings[index], "values", None) or [])]
            if not vector:
                results.append(
                    EmbeddingResult(model=self._config.model, error="empty embedding")
                )
                continue
            if len(vector) != self._config.dimension:
                # A silent dimension change would corrupt every pgvector insert,
                # so flag it loudly instead of storing a mismatched vector.
                message = (
                    f"model {self._config.model} returned dim={len(vector)}, "
                    f"expected {self._config.dimension}"
                )
                logger.error("Embedding dimension mismatch: %s", message)
                results.append(EmbeddingResult(model=self._config.model, error=message))
                continue

            results.append(
                EmbeddingResult(
                    vector=vector,
                    model=self._config.model,
                    token_count=per_text_tokens,
                )
            )
        return results


def _token_count(response: Any) -> int:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return 0
    return int(getattr(metadata, "total_token_count", 0) or 0)
