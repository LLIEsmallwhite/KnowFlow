"""
知识库 CRUD Service

Provides async PostgreSQL operations for KnowledgeBase model.
"""

import logging
from typing import List, Optional
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class KnowledgeBaseCRUD:
    """Async CRUD for knowledge_bases table."""

    async def create(
        self,
        db: AsyncSession,
        name: str,
        description: Optional[str] = None,
        kb_type: str = "document",
        created_by: Optional[str] = None,
        department: str = "_",
        security_level: int = 1,
    ) -> KnowledgeBase:
        kb = KnowledgeBase(
            name=name,
            description=description,
            kb_type=kb_type,
            created_by=created_by,
            department=department,
            security_level=security_level,
        )
        db.add(kb)
        await db.flush()
        await db.refresh(kb)
        logger.info("KB created: id=%s, name=%s", kb.id, name)
        return kb

    async def get(self, db: AsyncSession, kb_id: str) -> Optional[KnowledgeBase]:
        result = await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.is_active == True,
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        created_by: Optional[str] = None,
    ) -> List[KnowledgeBase]:
        stmt = select(KnowledgeBase).where(KnowledgeBase.is_active == True)
        if created_by:
            stmt = stmt.where(KnowledgeBase.created_by == created_by)
        stmt = stmt.order_by(KnowledgeBase.updated_at.desc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def update(
        self,
        db: AsyncSession,
        kb_id: str,
        **kwargs,
    ) -> Optional[KnowledgeBase]:
        allowed = {"name", "description", "kb_type", "chunking_config",
                   "retrieval_config", "is_active"}
        values = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not values:
            return await self.get(db, kb_id)
        await db.execute(
            update(KnowledgeBase).where(KnowledgeBase.id == kb_id).values(**values)
        )
        await db.flush()
        return await self.get(db, kb_id)

    async def delete(self, db: AsyncSession, kb_id: str) -> bool:
        """Soft-delete a knowledge base."""
        result = await db.execute(
            update(KnowledgeBase)
            .where(KnowledgeBase.id == kb_id)
            .values(is_active=False)
        )
        return result.rowcount > 0

    async def update_counts(
        self,
        db: AsyncSession,
        kb_id: str,
        doc_delta: int = 0,
        chunk_delta: int = 0,
    ) -> None:
        """Atomically update document_count and chunk_count."""
        await db.execute(
            update(KnowledgeBase)
            .where(KnowledgeBase.id == kb_id)
            .values(
                document_count=KnowledgeBase.document_count + doc_delta,
                chunk_count=KnowledgeBase.chunk_count + chunk_delta,
            )
        )
        await db.flush()

    async def get_total_count(self, db: AsyncSession) -> int:
        result = await db.execute(
            select(func.count()).select_from(KnowledgeBase).where(
                KnowledgeBase.is_active == True
            )
        )
        return result.scalar() or 0


# Singleton
kb_crud = KnowledgeBaseCRUD()
