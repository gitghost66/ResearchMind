"""
Base generic async SQLAlchemy repository.
"""

import uuid
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Generic async repository providing standard CRUD operations."""

    def __init__(self, model_cls: type[ModelT], session: AsyncSession) -> None:
        self.model_cls = model_cls
        self.session = session

    async def get_by_id(self, entity_id: uuid.UUID | str) -> ModelT | None:
        """Fetch a single record by primary key."""
        uid = uuid.UUID(str(entity_id)) if not isinstance(entity_id, uuid.UUID) else entity_id
        stmt = select(self.model_cls).where(self.model_cls.id == uid)  # type: ignore[attr-defined]
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self,
        skip: int = 0,
        limit: int = 100,
        **filters: Any,
    ) -> Sequence[ModelT]:
        """List records with optional pagination and filtering."""
        stmt = select(self.model_cls)
        for key, value in filters.items():
            if hasattr(self.model_cls, key):
                stmt = stmt.where(getattr(self.model_cls, key) == value)
        stmt = stmt.offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count(self, **filters: Any) -> int:
        """Count records matching optional filters."""
        stmt = select(func.count()).select_from(self.model_cls)
        for key, value in filters.items():
            if hasattr(self.model_cls, key):
                stmt = stmt.where(getattr(self.model_cls, key) == value)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def create(self, **kwargs: Any) -> ModelT:
        """Create and persist a new record."""
        instance = self.model_cls(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def update(self, entity_id: uuid.UUID | str, **kwargs: Any) -> ModelT | None:
        """Update an existing record."""
        uid = uuid.UUID(str(entity_id)) if not isinstance(entity_id, uuid.UUID) else entity_id
        stmt = (
            update(self.model_cls)
            .where(self.model_cls.id == uid)  # type: ignore[attr-defined]
            .values(**kwargs)
            .execution_options(synchronize_session="fetch")
        )
        await self.session.execute(stmt)
        await self.session.flush()
        return await self.get_by_id(uid)

    async def delete(self, entity_id: uuid.UUID | str) -> bool:
        """Delete a record by primary key."""
        uid = uuid.UUID(str(entity_id)) if not isinstance(entity_id, uuid.UUID) else entity_id
        stmt = delete(self.model_cls).where(self.model_cls.id == uid)  # type: ignore[attr-defined]
        result = await self.session.execute(stmt)
        await self.session.flush()
        self.session.expire_all()
        rowcount = getattr(result, "rowcount", 0)
        return bool(rowcount and rowcount > 0)
