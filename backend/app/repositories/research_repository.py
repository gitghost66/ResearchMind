"""
Repositories for ResearchRun, SubQuestion, Report, RunEvent, and LLMUsage.
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.research import (
    LLMUsage,
    Report,
    ResearchRun,
    RunEvent,
    SubQuestion,
)
from app.repositories.base import BaseRepository


class ResearchRunRepository(BaseRepository[ResearchRun]):
    """Repository handling research runs."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(ResearchRun, session)

    async def get_with_details(self, run_id: uuid.UUID | str) -> ResearchRun | None:
        """Fetch run with sub-questions, reports, and summary relations loaded."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        stmt = (
            select(ResearchRun)
            .where(ResearchRun.id == uid)
            .options(
                selectinload(ResearchRun.sub_questions),
                selectinload(ResearchRun.reports),
                selectinload(ResearchRun.sources),
                selectinload(ResearchRun.claims),
                selectinload(ResearchRun.contradictions),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        run_id: uuid.UUID | str,
        status: str,
        error: str | None = None,
    ) -> ResearchRun | None:
        """Update run execution status with timestamp tracking."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        update_data: dict[str, Any] = {"status": status}
        if status == "running":
            update_data["started_at"] = datetime.now(UTC)
        elif status in ("completed", "failed", "cancelled"):
            update_data["finished_at"] = datetime.now(UTC)
        if error is not None:
            update_data["error"] = error
        return await self.update(uid, **update_data)


class SubQuestionRepository(BaseRepository[SubQuestion]):
    """Repository handling planner-generated sub-questions."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(SubQuestion, session)

    async def get_by_run(self, run_id: uuid.UUID | str) -> Sequence[SubQuestion]:
        """Fetch all sub-questions for a run."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        stmt = select(SubQuestion).where(SubQuestion.run_id == uid).order_by(SubQuestion.created_at)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def create_batch(
        self,
        run_id: uuid.UUID | str,
        items: list[dict[str, Any]],
    ) -> list[SubQuestion]:
        """Create multiple sub-questions in batch."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        instances = [
            SubQuestion(
                run_id=uid,
                text=item["text"],
                domain=item.get("domain", "web"),
                status=item.get("status", "pending"),
            )
            for item in items
        ]
        self.session.add_all(instances)
        await self.session.flush()
        return instances


class ReportRepository(BaseRepository[Report]):
    """Repository handling synthesized research reports."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Report, session)

    async def get_latest_by_run(self, run_id: uuid.UUID | str) -> Report | None:
        """Get the latest report for a research run."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        stmt = (
            select(Report)
            .where(Report.run_id == uid)
            .order_by(desc(Report.version))
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


class RunEventRepository(BaseRepository[RunEvent]):
    """Repository handling agent event log."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(RunEvent, session)

    async def log_event(
        self,
        run_id: uuid.UUID | str,
        agent: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        """
        Append an event to the run log with the next ``seq``.

        Two writers racing for the same run would pick the same ``seq`` and
        one would fail the unique index. Concurrent writers go through
        ``app.services.run_events.RunEventRecorder``, which serialises them.
        """
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        event = RunEvent(
            run_id=uid,
            seq=await self.next_seq(uid),
            agent=agent,
            type=event_type,
            payload=payload or {},
        )
        self.session.add(event)
        await self.session.flush()
        return event

    async def next_seq(self, run_id: uuid.UUID | str) -> int:
        """The ``seq`` the run's next event gets."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        stmt = select(func.coalesce(func.max(RunEvent.seq), 0)).where(RunEvent.run_id == uid)
        return int((await self.session.execute(stmt)).scalar_one()) + 1

    async def list_by_run(
        self,
        run_id: uuid.UUID | str,
        *,
        after_seq: int = 0,
        limit: int | None = None,
    ) -> Sequence[RunEvent]:
        """A run's events in order, optionally only those after *after_seq*."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        stmt = (
            select(RunEvent)
            .where(RunEvent.run_id == uid, RunEvent.seq > after_seq)
            .order_by(RunEvent.seq)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()


class LLMUsageRepository(BaseRepository[LLMUsage]):
    """Repository handling LLM token and cost tracking."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(LLMUsage, session)

    async def record_usage(
        self,
        run_id: uuid.UUID | str,
        agent: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        latency_ms: int,
    ) -> LLMUsage:
        """Record usage metrics for an agent LLM call."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        usage = LLMUsage(
            run_id=uid,
            agent=agent,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self.session.add(usage)
        await self.session.flush()
        return usage

    async def get_summary_by_run(self, run_id: uuid.UUID | str) -> dict[str, Any]:
        """Aggregate total tokens and cost for a run, with per-agent breakdown."""
        uid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        totals_stmt = select(
            func.coalesce(func.sum(LLMUsage.tokens_in), 0).label("tokens_in"),
            func.coalesce(func.sum(LLMUsage.tokens_out), 0).label("tokens_out"),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0.0).label("total_cost"),
        ).where(LLMUsage.run_id == uid)

        totals_res = (await self.session.execute(totals_stmt)).one()

        breakdown_stmt = select(
            LLMUsage.agent,
            func.coalesce(func.sum(LLMUsage.tokens_in), 0).label("tokens_in"),
            func.coalesce(func.sum(LLMUsage.tokens_out), 0).label("tokens_out"),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0.0).label("cost_usd"),
        ).where(LLMUsage.run_id == uid).group_by(LLMUsage.agent)

        breakdown_rows = (await self.session.execute(breakdown_stmt)).all()

        per_agent = {
            row.agent: {
                "tokens_in": int(row.tokens_in),
                "tokens_out": int(row.tokens_out),
                "cost_usd": float(row.cost_usd),
            }
            for row in breakdown_rows
        }

        return {
            "total_tokens_in": int(totals_res.tokens_in),
            "total_tokens_out": int(totals_res.tokens_out),
            "total_cost_usd": float(totals_res.total_cost),
            "per_agent": per_agent,
        }
