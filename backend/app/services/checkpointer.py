"""
A LangGraph checkpoint saver on the app's own database (M8).

Runs resume from these checkpoints when their worker dies
(``app.workers.research_worker``) or when a failed run is resumed through the
API. The layout mirrors LangGraph's ``InMemorySaver``, row for row, in three
tables (``app.models.checkpoints``): checkpoints, channel blobs, and the writes
of tasks that finished inside an unfinished step.

It uses the app's SQLAlchemy engine, so it needs no driver or pool of its own,
works through Supabase's transaction pooler, and runs on the SQLite test
harness. Only the async API is implemented; the graph is only run with
``ainvoke``.

Every call is one short transaction. A step's checkpoint is written before
the next step starts (``run_research`` uses ``durability="sync"``), and a
task's writes as soon as it finishes, so an interrupted run loses at most the
tasks that were still running.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from sqlalchemy import and_, delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.checkpoints import GraphCheckpoint, GraphCheckpointBlob, GraphCheckpointWrite

__all__ = ["SqlCheckpointSaver"]

# A channel that a step cleared is stored as this type, with no value.
_EMPTY = "empty"


class SqlCheckpointSaver(BaseCheckpointSaver[str]):
    """``BaseCheckpointSaver`` over SQLAlchemy; one thread per research run."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self._session_factory = session_factory

    # -- reads ----------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config["configurable"]
        query = select(GraphCheckpoint).where(
            GraphCheckpoint.thread_id == str(configurable["thread_id"]),
            GraphCheckpoint.checkpoint_ns == configurable.get("checkpoint_ns", ""),
        )
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            query = query.where(GraphCheckpoint.checkpoint_id == checkpoint_id)
        else:
            # Checkpoint IDs are time-ordered (uuid6), so the largest is the latest.
            query = query.order_by(GraphCheckpoint.checkpoint_id.desc()).limit(1)
        async with self._session_factory() as session:
            row = (await session.execute(query)).scalar_one_or_none()
            return None if row is None else await self._tuple(session, row)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002 - LangGraph's name
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        query = select(GraphCheckpoint)
        if config is not None:
            configurable = config["configurable"]
            query = query.where(GraphCheckpoint.thread_id == str(configurable["thread_id"]))
            if configurable.get("checkpoint_ns") is not None:
                query = query.where(GraphCheckpoint.checkpoint_ns == configurable["checkpoint_ns"])
            if checkpoint_id := get_checkpoint_id(config):
                query = query.where(GraphCheckpoint.checkpoint_id == checkpoint_id)
        if before is not None and (before_id := get_checkpoint_id(before)):
            query = query.where(GraphCheckpoint.checkpoint_id < before_id)
        query = query.order_by(
            GraphCheckpoint.thread_id, GraphCheckpoint.checkpoint_ns, GraphCheckpoint.checkpoint_id.desc()
        )

        remaining = limit
        async with self._session_factory() as session:
            rows = (await session.execute(query)).scalars().all()
            for row in rows:
                if remaining is not None and remaining <= 0:
                    return
                if filter:
                    metadata = self.serde.loads_typed((row.metadata_type, row.metadata_blob))
                    if not all(metadata.get(key) == value for key, value in filter.items()):
                        continue
                if remaining is not None:
                    remaining -= 1
                yield await self._tuple(session, row)

    async def _tuple(self, session: AsyncSession, row: GraphCheckpoint) -> CheckpointTuple:
        checkpoint: Checkpoint = self.serde.loads_typed((row.checkpoint_type, row.checkpoint))
        config: RunnableConfig = {
            "configurable": {
                "thread_id": row.thread_id,
                "checkpoint_ns": row.checkpoint_ns,
                "checkpoint_id": row.checkpoint_id,
            }
        }
        parent: RunnableConfig | None = None
        if row.parent_checkpoint_id:
            parent = {
                "configurable": {
                    "thread_id": row.thread_id,
                    "checkpoint_ns": row.checkpoint_ns,
                    "checkpoint_id": row.parent_checkpoint_id,
                }
            }
        return CheckpointTuple(
            config=config,
            checkpoint={
                **checkpoint,
                "channel_values": await self._load_blobs(
                    session, row.thread_id, row.checkpoint_ns, checkpoint["channel_versions"]
                ),
            },
            metadata=self.serde.loads_typed((row.metadata_type, row.metadata_blob)),
            parent_config=parent,
            pending_writes=await self._load_writes(
                session, row.thread_id, row.checkpoint_ns, row.checkpoint_id
            ),
        )

    async def _load_blobs(
        self,
        session: AsyncSession,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        if not versions:
            return {}
        rows = await session.execute(
            select(GraphCheckpointBlob).where(
                GraphCheckpointBlob.thread_id == thread_id,
                GraphCheckpointBlob.checkpoint_ns == checkpoint_ns,
                or_(
                    *(
                        and_(
                            GraphCheckpointBlob.channel == channel,
                            GraphCheckpointBlob.version == str(version),
                        )
                        for channel, version in versions.items()
                    )
                ),
            )
        )
        return {
            blob.channel: self.serde.loads_typed((blob.type, blob.blob))
            for blob in rows.scalars()
            if blob.type != _EMPTY
        }

    async def _load_writes(
        self,
        session: AsyncSession,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[PendingWrite]:
        rows = (
            await session.execute(
                select(GraphCheckpointWrite).where(
                    GraphCheckpointWrite.thread_id == thread_id,
                    GraphCheckpointWrite.checkpoint_ns == checkpoint_ns,
                    GraphCheckpointWrite.checkpoint_id == checkpoint_id,
                )
            )
        ).scalars().all()
        ordered = sorted(rows, key=lambda w: writes_sort_key(w.task_path, w.task_id, w.idx))
        return [(w.task_id, w.channel, self.serde.loads_typed((w.type, w.blob))) for w in ordered]

    # -- writes ---------------------------------------------------------

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        stored = checkpoint.copy()
        values: dict[str, Any] = stored.pop("channel_values")  # type: ignore[misc]
        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(stored)
        metadata_type, metadata_blob = self.serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )

        async with self._session_factory() as session:
            for channel, version in new_versions.items():
                value_type, value_blob = (
                    self.serde.dumps_typed(values[channel]) if channel in values else (_EMPTY, b"")
                )
                await session.merge(
                    GraphCheckpointBlob(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        channel=channel,
                        version=str(version),
                        type=value_type,
                        blob=value_blob,
                    )
                )
            await session.merge(
                GraphCheckpoint(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint["id"],
                    parent_checkpoint_id=configurable.get("checkpoint_id"),
                    checkpoint_type=checkpoint_type,
                    checkpoint=checkpoint_blob,
                    metadata_type=metadata_type,
                    metadata_blob=metadata_blob,
                )
            )
            await session.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = str(configurable["checkpoint_id"])

        async with self._session_factory() as session:
            stored = set(
                (
                    await session.execute(
                        select(GraphCheckpointWrite.idx).where(
                            GraphCheckpointWrite.thread_id == thread_id,
                            GraphCheckpointWrite.checkpoint_ns == checkpoint_ns,
                            GraphCheckpointWrite.checkpoint_id == checkpoint_id,
                            GraphCheckpointWrite.task_id == task_id,
                        )
                    )
                ).scalars()
            )
            for position, (channel, value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel, position)
                # A task's ordinary writes are kept from its first save;
                # special ones (errors, interrupts) are replaced.
                if idx >= 0 and idx in stored:
                    continue
                value_type, value_blob = self.serde.dumps_typed(value)
                await session.merge(
                    GraphCheckpointWrite(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        task_id=task_id,
                        idx=idx,
                        channel=channel,
                        type=value_type,
                        blob=value_blob,
                        task_path=task_path,
                    )
                )
            await session.commit()

    async def adelete_thread(self, thread_id: str) -> None:
        """Remove every checkpoint of *thread_id* (a run's ID)."""
        async with self._session_factory() as session:
            for model in (GraphCheckpointWrite, GraphCheckpointBlob, GraphCheckpoint):
                await session.execute(delete(model).where(model.thread_id == str(thread_id)))
            await session.commit()

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Same scheme as ``InMemorySaver``: a zero-padded counter plus a random tail."""
        if current is None:
            number = 0
        elif isinstance(current, int):
            number = current
        else:
            number = int(current.split(".")[0])
        return f"{number + 1:032}.{random.random():016}"
