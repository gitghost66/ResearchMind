"""
LangGraph checkpoints (migration 0004): what a run needs to resume.

Three tables, the same split as LangGraph's own savers:

* ``graph_checkpoints`` — one row per step: the checkpoint without its channel
  values, its metadata, and the step before it;
* ``graph_checkpoint_blobs`` — channel values, one row per (channel, version),
  so a channel a step did not change is not stored again;
* ``graph_checkpoint_writes`` — what each finished task of an unfinished step
  wrote, so a resume re-runs only the tasks that had not finished.

``thread_id`` is the run ID, as a string: the saver is generic, and runs
remove their checkpoints themselves (when they complete, are cancelled, or
are deleted). Values are LangGraph-serialised bytes; nothing else reads them.
"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class GraphCheckpoint(Base):
    __tablename__ = "graph_checkpoints"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(255), primary_key=True, default="")
    checkpoint_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checkpoint_type: Mapped[str] = mapped_column(String(32), nullable=False)
    checkpoint: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    metadata_type: Mapped[str] = mapped_column(String(32), nullable=False)
    metadata_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class GraphCheckpointBlob(Base):
    __tablename__ = "graph_checkpoint_blobs"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(255), primary_key=True, default="")
    channel: Mapped[str] = mapped_column(String(255), primary_key=True)
    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class GraphCheckpointWrite(Base):
    __tablename__ = "graph_checkpoint_writes"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(255), primary_key=True, default="")
    checkpoint_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    task_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
