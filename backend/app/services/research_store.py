"""
SQL implementation of the agents package's ``ResearchStore``.

The graph persists through this object and never sees a session (AGENTS.md
rule 2). Each method opens its own short-lived session and commits, the same
pattern as the usage sink: a write the graph has been told succeeded is
durable, whatever happens to the rest of the run.

Events go through the process-wide :class:`RunEventRecorder`, which numbers
them and wakes SSE streams; pass the app's recorder so they share its locks.

No prompt text or agent logic lives here.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from researchmind_agents.state import (
    PassageItem,
    ResearchReport,
    SourceItem,
    SubQuestionItem,
)
from researchmind_agents.tools.vector_search import (
    VectorSearchFn,
    VectorSearchResult,
    make_vector_search_fn,
)
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.research import SubQuestion
from app.repositories.evidence_repository import PassageRepository, SourceRepository
from app.repositories.research_repository import ReportRepository, ResearchRunRepository
from app.services.run_events import RunEventRecorder

logger = logging.getLogger(__name__)

__all__ = ["SqlResearchStore", "make_session_vector_search_fn"]


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class SqlResearchStore:
    """
    ``ResearchStore`` backed by the repository layer.

    Source get-or-create is serialised by an ``asyncio.Lock`` held by this
    instance, so concurrent graph branches saving the same URL create exactly
    one row. That assumes one store per run inside a single process, which
    holds until M8; multi-worker runs will need a database-level guarantee
    (a unique constraint or advisory lock) instead.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        recorder: RunEventRecorder | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._recorder = recorder or RunEventRecorder(session_factory)
        self._source_lock = asyncio.Lock()

    async def save_plan(
        self,
        *,
        run_id: str,
        plan: dict[str, Any],
        sub_questions: list[SubQuestionItem],
    ) -> None:
        run_uuid = _uuid(run_id)
        async with self._session_factory() as session:
            await ResearchRunRepository(session).update(run_uuid, plan=dict(plan))
            session.add_all(
                SubQuestion(
                    id=_uuid(item.id),
                    run_id=run_uuid,
                    text=item.text,
                    domain=item.domain,
                    status=item.status,
                )
                for item in sub_questions
            )
            await session.commit()

    async def set_sub_question_status(
        self,
        *,
        run_id: str,
        sub_question_id: str,
        status: str,
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(SubQuestion)
                .where(
                    SubQuestion.id == _uuid(sub_question_id),
                    SubQuestion.run_id == _uuid(run_id),
                )
                .values(status=status)
            )
            await session.commit()

    async def save_source(self, *, run_id: str, source: SourceItem) -> tuple[str, bool]:
        run_uuid = _uuid(run_id)
        async with self._source_lock, self._session_factory() as session:
            repo = SourceRepository(session)
            existing = await repo.get_by_url_or_hash(
                run_uuid, source.url, source.content_hash or None
            )
            if existing is not None:
                return str(existing.id), False

            # published_at, domain_type and reliability_score keep their
            # defaults: source scoring arrives in M5 and is never guessed here.
            created = await repo.create(
                id=_uuid(source.id),
                run_id=run_uuid,
                url=source.url,
                title=source.title,
                content_hash=source.content_hash,
            )
            await session.commit()
            return str(created.id), True

    async def save_passages(
        self,
        *,
        run_id: str,
        passages: list[PassageItem],
        embeddings: list[list[float] | None],
        embedding_model: str,
    ) -> None:
        if len(passages) != len(embeddings):
            raise ValueError("passages and embeddings must have the same length")
        if not passages:
            return

        items = [
            {
                "id": _uuid(passage.id),
                "source_id": _uuid(passage.source_id),
                "sub_question_id": (
                    _uuid(passage.sub_question_id) if passage.sub_question_id else None
                ),
                "text": passage.text,
                "chunk_index": passage.chunk_index,
                "embedding": list(vector) if vector else None,
                "embedding_model": embedding_model,
            }
            for passage, vector in zip(passages, embeddings, strict=True)
        ]
        async with self._session_factory() as session:
            await PassageRepository(session).create_batch(items)
            await session.commit()

    async def save_report(self, *, run_id: str, report: ResearchReport, markdown: str) -> str:
        async with self._session_factory() as session:
            row = await ReportRepository(session).create(
                run_id=_uuid(run_id),
                content=report.model_dump(mode="json"),
                markdown=markdown,
                # The column is NOT NULL, and confidence is unscored until M5,
                # so 0.0 is stored here; the content JSON keeps null.
                overall_confidence=report.overall_confidence or 0.0,
                version=1,
            )
            await session.commit()
            return str(row.id)

    async def log_event(
        self,
        *,
        run_id: str,
        agent: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        await self._recorder.record(
            run_id=run_id, agent=agent, event_type=event_type, payload=dict(payload)
        )

    async def list_source_ids(self, *, run_id: str) -> set[str]:
        async with self._session_factory() as session:
            sources = await SourceRepository(session).list_by_run(_uuid(run_id))
            return {str(source.id) for source in sources}


def make_session_vector_search_fn(
    session_factory: async_sessionmaker[AsyncSession],
) -> VectorSearchFn:
    """
    A ``VectorSearchFn`` that opens a fresh session per call.

    The graph searches long after any request session would have closed, and
    its writes happen in their own sessions, so a per-call session is the only
    one guaranteed to see committed passages. Results are built inside the
    session, so nothing lazy-loads after it closes.
    """

    async def search(
        query_vector: list[float],
        *,
        top_k: int = 5,
        run_id: str | None = None,
        sub_question_id: str | None = None,
    ) -> list[VectorSearchResult]:
        async with session_factory() as session:
            inner = make_vector_search_fn(PassageRepository(session))
            return list(
                await inner(
                    query_vector,
                    top_k=top_k,
                    run_id=run_id,
                    sub_question_id=sub_question_id,
                )
            )

    return search
