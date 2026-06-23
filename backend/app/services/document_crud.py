"""
Document CRUD Service

Provides async PostgreSQL operations for Document model.
"""

import logging
from typing import List, Optional
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document

logger = logging.getLogger(__name__)


class DocumentCRUD:
    """Async CRUD for documents table."""

    async def create(
        self,
        db: AsyncSession,
        knowledge_base_id: str,
        title: str,
        file_name: str,
        file_type: str,
        file_path: str,
        file_size: Optional[int] = None,
        file_hash: Optional[str] = None,
        source: str = "upload",
    ) -> Document:
        doc = Document(
            knowledge_base_id=knowledge_base_id,
            title=title,
            file_name=file_name,
            file_type=file_type,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            source=source,
            status="pending",
        )
        db.add(doc)
        await db.flush()
        await db.refresh(doc)
        logger.info("Doc created: id=%s, title=%s", doc.id, title)
        return doc

    async def get(self, db: AsyncSession, doc_id: str) -> Optional[Document]:
        result = await db.execute(
            select(Document).where(
                Document.id == doc_id,
                Document.is_active == True,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_kb(
        self,
        db: AsyncSession,
        kb_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Document]:
        stmt = (
            select(Document)
            .where(
                Document.knowledge_base_id == kb_id,
                Document.is_active == True,
            )
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def update_status(
        self,
        db: AsyncSession,
        doc_id: str,
        status: str,
        error_message: Optional[str] = None,
        progress: Optional[float] = None,
        chunk_count: Optional[int] = None,
        is_indexed: Optional[bool] = None,
    ) -> Optional[Document]:
        values = {"status": status}
        if error_message is not None:
            values["error_message"] = error_message
        if progress is not None:
            values["progress"] = progress
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        if is_indexed is not None:
            values["is_indexed"] = is_indexed

        await db.execute(
            update(Document).where(Document.id == doc_id).values(**values)
        )
        await db.flush()
        return await self.get(db, doc_id)

    async def delete(self, db: AsyncSession, doc_id: str) -> bool:
        result = await db.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(is_active=False)
        )
        return result.rowcount > 0

    async def count_by_kb(self, db: AsyncSession, kb_id: str) -> int:
        result = await db.execute(
            select(func.count())
            .select_from(Document)
            .where(
                Document.knowledge_base_id == kb_id,
                Document.is_active == True,
            )
        )
        return result.scalar() or 0


# Singleton
doc_crud = DocumentCRUD()
