"""
Repositories for Source and Passage with pgvector similarity search.
"""

import json
import math
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.evidence import Passage, Source
from app.repositories.base import BaseRepository


def _cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine distance (1 - cosine_similarity) between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    sim = dot / (norm_a * norm_b)
    return max(0.0, min(2.0, 1.0 - sim))


class SourceRepository(BaseRepository[Source]):
    """Repository handling web/academic sources with URL/content_hash deduplication."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Source, session)

    async def get_by_url_or_hash(
        self,
        run_id: uuid.UUID | str,
        url: str,
        content_hash: str | None = None,
    ) -> Source | None:
        """Find an existing source by URL or content hash within a run."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        stmt = select(Source).where(Source.run_id == uid)
        if content_hash:
            stmt = stmt.where((Source.url == url) | (Source.content_hash == content_hash))
        else:
            stmt = stmt.where(Source.url == url)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def list_by_run(self, run_id: uuid.UUID | str) -> Sequence[Source]:
        """List all sources collected for a run."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        stmt = select(Source).where(Source.run_id == uid).order_by(Source.reliability_score.desc())
        result = await self.session.execute(stmt)
        return result.scalars().all()


class PassageRepository(BaseRepository[Passage]):
    """Repository handling text chunks and vector similarity queries."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Passage, session)

    async def create_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[Passage]:
        """Create multiple passages with embeddings in batch."""
        instances = [Passage(**item) for item in items]
        self.session.add_all(instances)
        await self.session.flush()
        return instances

    async def search_similar(
        self,
        query_embedding: list[float],
        run_id: uuid.UUID | str | None = None,
        sub_question_id: uuid.UUID | str | None = None,
        limit: int = 5,
    ) -> Sequence[tuple[Passage, float]]:
        """
        Search passages by cosine distance using pgvector on PostgreSQL,
        with an in-memory cosine fallback for SQLite unit testing.
        Returns a list of (Passage, distance) tuples sorted by ascending distance.
        """
        bind = self.session.bind
        dialect_name = bind.dialect.name if bind is not None else "postgresql"

        if dialect_name == "postgresql":
            # Native pgvector <=> operator with HNSW index
            stmt = select(
                Passage,
                Passage.embedding.cosine_distance(query_embedding).label("distance"),
            ).options(selectinload(Passage.source))

            if run_id:
                uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
                stmt = stmt.join(Source, Passage.source_id == Source.id).where(Source.run_id == uid)

            if sub_question_id:
                sq_id = (
                    uuid.UUID(str(sub_question_id))
                    if not isinstance(sub_question_id, uuid.UUID)
                    else sub_question_id
                )
                stmt = stmt.where(Passage.sub_question_id == sq_id)

            stmt = stmt.order_by("distance").limit(limit)
            result = await self.session.execute(stmt)
            return [(row[0], float(row[1])) for row in result.all()]

        # SQLite fallback for test environments without native pgvector
        stmt_fallback = select(Passage).options(selectinload(Passage.source))
        if run_id:
            uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
            stmt_fallback = stmt_fallback.join(Source, Passage.source_id == Source.id).where(Source.run_id == uid)
        if sub_question_id:
            sq_id = (
                uuid.UUID(str(sub_question_id))
                if not isinstance(sub_question_id, uuid.UUID)
                else sub_question_id
            )
            stmt_fallback = stmt_fallback.where(Passage.sub_question_id == sq_id)

        rows = (await self.session.execute(stmt_fallback)).scalars().all()
        scored: list[tuple[Passage, float]] = []

        for p in rows:
            emb = p.embedding
            if emb is None:
                continue
            if isinstance(emb, str):
                emb_vec = json.loads(emb) if emb.startswith("[") else []
            elif hasattr(emb, "tolist"):
                emb_vec = emb.tolist()
            else:
                emb_vec = list(emb)

            if emb_vec:
                dist = _cosine_distance(query_embedding, emb_vec)
                scored.append((p, dist))

        scored.sort(key=lambda x: x[1])
        return scored[:limit]
