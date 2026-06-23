"""
Celery Async Tasks for Document Processing

Provides background processing for document uploads:
- Parse document
- Chunk text
- Generate embeddings
- Index into Milvus
- Build BM25 index
- Update DB status
"""

import os
import uuid
import logging
from celery import Celery

from app.core.config import settings

logger = logging.getLogger(__name__)

# Celery app instance
celery_app = Celery(
    "knowflow",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def process_document_task(
    self,
    doc_id: str,
    kb_id: str,
    file_path: str,
    file_type: str,
):
    """
    Async document processing pipeline:
    Parse → Chunk → Embed → Index → Update DB

    Args:
        doc_id: Document ID in PostgreSQL
        kb_id: Knowledge Base ID
        file_path: Local temp file path or MinIO object path
        file_type: File extension (pdf, docx, md, etc.)
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    sync_url = settings.DATABASE_URL
    engine = create_engine(sync_url)

    try:
        # Update status to processing
        with Session(engine) as db:
            from app.services.document_crud import DocumentCRUD
            doc_crud = DocumentCRUD()
            doc_crud.sync_update_status(db, doc_id, status="processing")

        # Parse and chunk
        from app.services.doc_service import doc_service
        parsed_doc, chunks = doc_service.process_document_pipeline(
            file_path=file_path,
            file_type=file_type,
        )

        # Build chunk records
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dicts.append({
                "document_id": doc_id,
                "knowledge_base_id": kb_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "content_hash": chunk.metadata.get("content_hash") if chunk.metadata else None,
                "metadata": chunk.metadata or {},
            })

        # Persist chunks
        with Session(engine) as db:
            from app.services.chunk_crud import ChunkCRUD
            chunk_crud = ChunkCRUD()
            orm_chunks = chunk_crud.sync_bulk_create(db, chunk_dicts)

        # Embed + index into Milvus
        idx_chunks = [c for c in orm_chunks if c.chunk_type in ("child", "text")]
        if idx_chunks:
            from app.retrieval.dense_retriever import DenseRetriever
            from app.retrieval.milvus_client import MilvusClient

            contents = [c.content for c in idx_chunks]
            dense = DenseRetriever()
            embeddings = dense.embed_documents(contents)

            milvus = MilvusClient()
            milvus.connect()
            milvus.create_collection()
            milvus._ensure_partition(kb_id)
            milvus.insert_vectors(
                chunk_ids=[c.id for c in idx_chunks],
                embeddings=embeddings,
                contents=contents,
                kb_id=kb_id,
            )

        # Update BM25 index
        with Session(engine) as db:
            from app.services.chunk_crud import ChunkCRUD
            chunk_crud = ChunkCRUD()
            indexable = chunk_crud.sync_get_indexable(db, kb_id)
            from app.retrieval.bm25_retriever import BM25Retriever, build_bm25_index_from_db
            bm25 = BM25Retriever()
            build_bm25_index_from_db(kb_id, indexable, bm25)

        # Update status to completed
        with Session(engine) as db:
            from app.services.document_crud import DocumentCRUD
            from app.services.kb_crud import KnowledgeBaseCRUD
            doc_crud = DocumentCRUD()
            doc_crud.sync_update_status(
                db, doc_id, status="completed",
                chunk_count=len(chunks), is_indexed=True,
            )
            kb_crud = KnowledgeBaseCRUD()
            kb_crud.sync_update_counts(db, kb_id, doc_delta=1, chunk_delta=len(chunks))

        # Cleanup temp file
        try:
            os.unlink(file_path)
        except Exception:
            pass

        return {"status": "completed", "doc_id": doc_id, "chunk_count": len(chunks)}

    except Exception as exc:
        logger.error("Document processing failed: %s", exc, exc_info=True)
        # Update status to failed
        try:
            with Session(engine) as db:
                from app.services.document_crud import DocumentCRUD
                doc_crud = DocumentCRUD()
                doc_crud.sync_update_status(
                    db, doc_id, status="failed",
                    error_message=str(exc)[:500],
                )
        except Exception:
            pass

        raise self.retry(exc=exc)


@celery_app.task
def rebuild_index_task(kb_id: str):
    """Rebuild BM25 index for a knowledge base from DB chunks."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.services.chunk_crud import ChunkCRUD
    from app.retrieval.bm25_retriever import BM25Retriever, build_bm25_index_from_db

    sync_url = settings.DATABASE_URL
    engine = create_engine(sync_url)

    with Session(engine) as db:
        chunk_crud = ChunkCRUD()
        indexable = chunk_crud.sync_get_indexable(db, kb_id)
        bm25 = BM25Retriever()
        build_bm25_index_from_db(kb_id, indexable, bm25)

    return {"status": "completed", "kb_id": kb_id, "chunk_count": len(indexable)}
