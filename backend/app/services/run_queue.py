"""
The run queue, kept in ``research_runs`` itself (M8).

The API only inserts ``queued`` runs. Workers (``app.workers.research_worker``)
take them with :meth:`RunQueue.claim_next`, which hands out, in order:

1. runs whose worker died: ``running`` with a lapsed lease. They are resumed
   from their last checkpoint, or failed once they reach ``MAX_RUN_ATTEMPTS``
   (a run that keeps killing its worker must not loop forever);
2. ``queued`` runs, oldest first.

A claim is one conditional ``UPDATE`` (``... WHERE status = 'queued'``, or
``... AND lease_expires_at <= now``) plus its ``run_status`` event, so two
workers racing for a run cannot both get it: the second update matches no row.
This works the same on PostgreSQL and SQLite, with no row locks.

A claimed run carries a lease the worker renews (:meth:`renew`). Renewal
reports the runs this worker no longer owns (cancelled, deleted, or taken over
after a stall) so it stops working on them. A worker shutting down cleanly
:meth:`release`\\ s its runs: they become claimable at once, and the handover
does not count as an attempt.

Lease times come from each process's clock; keep hosts in sync (NTP). The
lease is long enough (``RUN_LEASE_SECONDS``, default 60) to absorb small skew.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.research import ResearchRun
from app.services.run_events import ACTIVE_STATUSES, INTERRUPTED_ERROR, RunEventRecorder

logger = logging.getLogger(__name__)

__all__ = ["Claim", "RunQueue", "gave_up_error"]

# Candidates read per claim attempt; a worker losing a race tries the next.
_CANDIDATES = 5


def gave_up_error(attempts: int) -> str:
    return f"{INTERRUPTED_ERROR} Its worker stopped {attempts} times, so it was not resumed again."


@dataclass(frozen=True)
class Claim:
    """A run a worker now owns."""

    run_id: str
    # Starts since the run was last queued: 1 for a fresh run.
    attempt: int
    # The run has started before: continue from its checkpoint, count its
    # earlier spend, and close the steps the interruption left open.
    resumed: bool


class RunQueue:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        recorder: RunEventRecorder,
        *,
        lease_seconds: float,
        max_attempts: int,
    ) -> None:
        self._session_factory = session_factory
        self._recorder = recorder
        self.lease_seconds = max(1.0, lease_seconds)
        self.max_attempts = max(1, max_attempts)

    def _lease_until(self) -> datetime:
        return datetime.now(UTC) + timedelta(seconds=self.lease_seconds)

    # -- reads ----------------------------------------------------------

    async def counts(self) -> dict[str, int]:
        """Runs queued and running, across every worker."""
        async with self._session_factory() as session:
            rows = await session.execute(
                select(ResearchRun.status, func.count())
                .where(ResearchRun.status.in_(list(ACTIVE_STATUSES)))
                .group_by(ResearchRun.status)
            )
            found = {status: int(count) for status, count in rows.all()}
        return {status: found.get(status, 0) for status in ("queued", "running")}

    # -- claiming -------------------------------------------------------

    async def claim_next(self, worker_id: str) -> Claim | None:
        """The next run for *worker_id*: an interrupted one first, then the oldest queued."""
        await self._fail_unleased()
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            lapsed = (
                await session.execute(
                    select(ResearchRun.id, ResearchRun.attempts)
                    .where(
                        ResearchRun.status == "running",
                        ResearchRun.lease_expires_at.is_not(None),
                        ResearchRun.lease_expires_at <= now,
                    )
                    .order_by(ResearchRun.lease_expires_at, ResearchRun.id)
                    .limit(_CANDIDATES)
                )
            ).all()
        for run_uuid, attempts in lapsed:
            run_id = str(run_uuid)
            lapsed_now = [ResearchRun.lease_expires_at <= now]
            if attempts >= self.max_attempts:
                await self._recorder.transition(
                    run_id,
                    "failed",
                    error=gave_up_error(attempts),
                    from_statuses={"running"},
                    where=lapsed_now,
                )
                logger.warning(
                    "Run %s was interrupted %d times; failing it", run_id, attempts,
                    extra={"run_id": run_id},
                )
                continue
            claimed = await self._recorder.transition(
                run_id,
                "running",
                from_statuses={"running"},
                values={
                    "worker_id": worker_id,
                    "lease_expires_at": self._lease_until(),
                    "attempts": ResearchRun.attempts + 1,
                },
                where=lapsed_now,
                details={"resumed": True, "attempt": attempts + 1},
            )
            if claimed:
                logger.info(
                    "Resuming run %s (attempt %d)", run_id, attempts + 1, extra={"run_id": run_id}
                )
                return Claim(run_id=run_id, attempt=attempts + 1, resumed=True)

        async with self._session_factory() as session:
            queued = (
                await session.execute(
                    select(ResearchRun.id, ResearchRun.attempts, ResearchRun.started_at)
                    .where(ResearchRun.status == "queued")
                    .order_by(ResearchRun.created_at, ResearchRun.id)
                    .limit(_CANDIDATES)
                )
            ).all()
        for run_uuid, attempts, started_at in queued:
            claim = await self.claim_run(
                str(run_uuid), worker_id, attempts=attempts, resumed=started_at is not None
            )
            if claim is not None:
                return claim
        return None

    async def _fail_unleased(self) -> None:
        """
        Fail ``running`` runs that carry no lease. Every claim sets one, so
        these predate leases (migration 0004): runs a process left behind
        before M8. No worker can own or resume them; failed, they can still
        be resumed through the API.
        """
        async with self._session_factory() as session:
            stranded = (
                await session.execute(
                    select(ResearchRun.id).where(
                        ResearchRun.status == "running", ResearchRun.lease_expires_at.is_(None)
                    )
                )
            ).scalars().all()
        for run_uuid in stranded:
            await self._recorder.transition(
                str(run_uuid),
                "failed",
                error=INTERRUPTED_ERROR,
                from_statuses={"running"},
                where=[ResearchRun.lease_expires_at.is_(None)],
            )
            logger.warning("Failed run %s, left running before leases", run_uuid)

    async def claim_run(
        self,
        run_id: str,
        worker_id: str,
        *,
        attempts: int = 0,
        resumed: bool = False,
    ) -> Claim | None:
        """Claim one ``queued`` run; ``None`` if another worker (or a cancel) got there first."""
        details = {"resumed": True, "attempt": attempts + 1} if resumed else None
        claimed = await self._recorder.transition(
            run_id,
            "running",
            from_statuses={"queued"},
            values={
                "worker_id": worker_id,
                "lease_expires_at": self._lease_until(),
                "attempts": ResearchRun.attempts + 1,
            },
            details=details,
        )
        if not claimed:
            return None
        return Claim(run_id=run_id, attempt=attempts + 1, resumed=resumed)

    # -- leases ---------------------------------------------------------

    async def renew(self, worker_id: str, run_ids: Iterable[str]) -> set[str]:
        """Extend *worker_id*'s leases; returns the runs it no longer owns."""
        lost: set[str] = set()
        ids = list(run_ids)
        if not ids:
            return lost
        until = self._lease_until()
        async with self._session_factory() as session:
            for run_id in ids:
                result = await session.execute(
                    update(ResearchRun)
                    .where(
                        ResearchRun.id == uuid.UUID(run_id),
                        ResearchRun.worker_id == worker_id,
                        ResearchRun.status == "running",
                    )
                    .values(lease_expires_at=until)
                    .execution_options(synchronize_session=False)
                )
                if not getattr(result, "rowcount", 0):
                    lost.add(run_id)
            await session.commit()
        return lost

    async def release(self, worker_id: str, run_id: str) -> bool:
        """
        Hand a running run back (clean shutdown): claimable at once, and the
        interrupted attempt is not counted.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                update(ResearchRun)
                .where(
                    ResearchRun.id == uuid.UUID(run_id),
                    ResearchRun.worker_id == worker_id,
                    ResearchRun.status == "running",
                )
                .values(
                    worker_id=None,
                    lease_expires_at=datetime.now(UTC),
                    attempts=case(
                        (ResearchRun.attempts > 0, ResearchRun.attempts - 1), else_=0
                    ),
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()
        released = bool(getattr(result, "rowcount", 0))
        if released:
            self._recorder.notifier.notify_status(run_id, "queued")  # wake local workers
        return released

    @contextlib.asynccontextmanager
    async def keep_lease(self, run_id: str, worker_id: str) -> AsyncIterator[None]:
        """Renew *run_id*'s lease in the background while the block runs (scripts)."""

        async def renew_forever() -> None:
            while True:
                await asyncio.sleep(self.lease_seconds / 3)
                try:
                    await self.renew(worker_id, [run_id])
                except Exception as exc:  # noqa: BLE001 - try again next time
                    logger.warning("Could not renew the lease on run %s: %s", run_id, exc)

        task = asyncio.create_task(renew_forever())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # -- resume ---------------------------------------------------------

    async def requeue_failed(self, run_id: str) -> bool:
        """Queue a failed run again; a worker resumes it from its last checkpoint."""
        return await self._recorder.transition(
            run_id,
            "queued",
            from_statuses={"failed"},
            values={
                "error": None,
                "finished_at": None,
                "worker_id": None,
                "lease_expires_at": None,
                "attempts": 0,
            },
            details={"resumed": True},
        )
