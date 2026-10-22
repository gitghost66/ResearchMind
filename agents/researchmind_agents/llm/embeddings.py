"""
Embedding service using Google's text-embedding-004 model (768-dimensional).

No HTTP server code, no DB sessions — pure agent-layer logic.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

_TASK_TYPE_RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
_TASK_TYPE_RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


@dataclass
class EmbeddingResult:
    """Result from a single embedding call."""

    vector: list[float]
    model: str
    token_count: int = 0


class EmbeddingService:
    """
    Thin async wrapper around google-genai's embed_content API.

    Usage::

        svc = EmbeddingService(api_key="...", model="text-embedding-004")

        # Embed a single query
        result = await svc.embed_query("What is RAG?")

        # Embed multiple documents (batched)
        results = await svc.embed_documents(["doc1...", "doc2..."])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        dimension: int = 768,
        batch_size: int = 100,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._client = genai.Client(api_key=api_key)

    @property
    def dimension(self) -> int:
        return self._dimension

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single query string (uses RETRIEVAL_QUERY task type)."""
        return await self._embed_one(text, task_type=_TASK_TYPE_RETRIEVAL_QUERY)

    async def embed_document(self, text: str) -> EmbeddingResult:
        """Embed a single document (uses RETRIEVAL_DOCUMENT task type)."""
        return await self._embed_one(text, task_type=_TASK_TYPE_RETRIEVAL_DOCUMENT)

    async def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Embed multiple documents in parallel batches.

        Splits into chunks of *batch_size* and awaits concurrently.
        """
        if not texts:
            return []

        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]

        all_results: list[EmbeddingResult] = []
        for batch in batches:
            tasks = [
                self._embed_one(t, task_type=_TASK_TYPE_RETRIEVAL_DOCUMENT)
                for t in batch
            ]
            results = await asyncio.gather(*tasks)
            all_results.extend(results)

        return all_results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _embed_one(self, text: str, *, task_type: str) -> EmbeddingResult:
        """Embed a single text string."""

        def _sync_embed() -> Any:  # type: ignore[name-defined]
            return self._client.models.embed_content(
                model=self._model,
                contents=text,
                config=genai_types.EmbedContentConfig(task_type=task_type),
            )

        response = await asyncio.to_thread(_sync_embed)

        embedding_obj = response.embeddings[0] if response.embeddings else None
        vector: list[float] = list(embedding_obj.values) if embedding_obj else []

        if not vector:
            logger.warning("Empty embedding returned for text (len=%d)", len(text))

        token_count: int = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_count = getattr(response.usage_metadata, "total_token_count", 0) or 0

        logger.debug(
            "Embedded text len=%d model=%s dim=%d",
            len(text),
            self._model,
            len(vector),
        )

        return EmbeddingResult(
            vector=vector,
            model=self._model,
            token_count=token_count,
        )
