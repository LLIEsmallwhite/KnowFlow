"""
Chunk CRUD Service

Provides async PostgreSQL operations for Chunk model.
"""

import logging
from typing import List, Optional, Dict
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk

logger = logging.getLogger(__name__)


class ChunkCRUD:
    """Async CRUD for chunks table."""

    async def bulk_create(
        self,
        db: AsyncSession,
        chunks: List[Dict],
    ) -> List[Chunk]:
        """Bulk insert chunks. Each dict must have content, chunk_index,
        chunk_type, document_id, knowledge_base_id.

        Optional: content_hash, parent_chunk_id, metadata, start_at, end_at.
        """
        orm_chunks = []
        for c in chunks:
            orm_chunks.append(Chunk(
                document_id=c["document_id"],
                knowledge_base_id=c["knowledge_base_id"],
                content=c["content"],
                chunk_index=c.get("chunk_index", 0),
                chunk_type=c.get("chunk_type", "text"),
                content_hash=c.get("content_hash"),
                parent_chunk_id=c.get("parent_chunk_id"),
                extra_metadata=c.get("metadata"),
                start_at=c.get("start_at"),
                end_at=c.get("end_at"),
                is_indexed=False,
            ))
        db.add_all(orm_chunks)
        await db.flush()
        # Refresh all to get generated IDs
        for chunk in orm_chunks:
            await db.refresh(chunk)
        logger.info("Bulk created %d chunks", len(orm_chunks))
        return orm_chunks

    async def get_by_kb(
        self,
        db: AsyncSession,
        kb_id: str,
        skip: int = 0,
        limit: int = 10000,
        chunk_types: Optional[List[str]] = None,
    ) -> List[Chunk]:
        stmt = select(Chunk).where(
            Chunk.knowledge_base_id == kb_id,
            Chunk.is_enabled == True,
        )
        if chunk_types:
            stmt = stmt.where(Chunk.chunk_type.in_(chunk_types))
        stmt = stmt.order_by(Chunk.chunk_index).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_doc(
        self,
        db: AsyncSession,
        doc_id: str,
        skip: int = 0,
        limit: int = 10000,
    ) -> List[Chunk]:
        stmt = (
            select(Chunk)
            .where(
                Chunk.document_id == doc_id,
                Chunk.is_enabled == True,
            )
            .order_by(Chunk.chunk_index)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_indexable(
        self,
        db: AsyncSession,
        kb_id: str,
        chunk_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Return chunks as dicts suitable for BM25/Milvus indexing."""
        if chunk_types is None:
            chunk_types = ["child", "text"]
        chunks = await self.get_by_kb(db, kb_id, chunk_types=chunk_types)
        return [
            {
                "id": c.id,
                "chunk_id": c.id,
                "content": c.content,
                "chunk_index": c.chunk_index,
                "chunk_type": c.chunk_type,
                "document_id": c.document_id,
                "metadata": c.extra_metadata or {},
            }
            for c in chunks
        ]

    async def delete_by_kb(self, db: AsyncSession, kb_id: str) -> int:
        result = await db.execute(
            delete(Chunk).where(Chunk.knowledge_base_id == kb_id)
        )
        await db.flush()
        logger.info("Deleted %d chunks from KB '%s'", result.rowcount, kb_id)
        return result.rowcount

    async def delete_by_doc(self, db: AsyncSession, doc_id: str) -> int:
        result = await db.execute(
            delete(Chunk).where(Chunk.document_id == doc_id)
        )
        await db.flush()
        logger.info("Deleted %d chunks from doc '%s'", result.rowcount, doc_id)
        return result.rowcount

    async def mark_indexed(
        self,
        db: AsyncSession,
        chunk_ids: List[str],
    ) -> None:
        from sqlalchemy import update as sql_update
        await db.execute(
            sql_update(Chunk)
            .where(Chunk.id.in_(chunk_ids))
            .values(is_indexed=True)
        )
        await db.flush()


# Singleton
chunk_crud = ChunkCRUD()
