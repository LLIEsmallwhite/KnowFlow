"""
Celery Async Tasks for Document Processing

Background: parse -> chunk -> embed -> Milvus + BM25 -> update DB.
"""

import os
import logging
from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "knowflow",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json", accept_content=["json"], result_serializer="json",
    timezone="Asia/Shanghai", enable_utc=True,
    task_track_started=True, task_acks_late=True, worker_prefetch_multiplier=1,
)


def _get_sync_session() -> Session:
    """Create a sync DB session for Celery worker."""
    engine = create_engine(settings.DATABASE_URL, pool_size=5, max_overflow=10)
    return Session(engine)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def process_document_task(self, doc_id: str, kb_id: str, file_path: str, file_type: str):
    """Process uploaded document asynchronously."""
    session = _get_sync_session()
    try:
        # 1. Update status -> processing
        from app.models.document import Document
        from sqlalchemy import update as sql_update
        session.execute(sql_update(Document).where(Document.id == doc_id).values(status="processing"))
        session.commit()

        # 2. Parse + chunk
        from app.services.doc_service import doc_service
        parsed_doc, chunks = doc_service.process_document_pipeline(file_path, file_type)

        # 3. Persist chunks to DB
        from app.models.chunk import Chunk
        from app.models.knowledge_base import KnowledgeBase
        kb = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        kb_security = kb.security_level if kb else 1
        kb_dept = kb.department if kb else ""

        chunk_dicts = []
        for chunk in chunks:
            chunk_dicts.append({
                "document_id": doc_id, "knowledge_base_id": kb_id,
                "content": chunk.content, "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "content_hash": chunk.metadata.get("content_hash") if chunk.metadata else None,
                "metadata": chunk.metadata or {},
                "security_level": kb_security, "department": kb_dept,
            })

        orm_chunks = []
        for c in chunk_dicts:
            orm_chunks.append(Chunk(
                document_id=c["document_id"], knowledge_base_id=c["knowledge_base_id"],
                content=c["content"], chunk_index=c.get("chunk_index", 0),
                chunk_type=c.get("chunk_type", "text"),
                content_hash=c.get("content_hash"),
                extra_metadata=c.get("metadata"),
                security_level=c.get("security_level", 1),
                department=c.get("department", ""),
            ))
        session.add_all(orm_chunks)
        session.flush()
        for oc in orm_chunks:
            session.refresh(oc)

        # 4. Embed + Milvus
        idx_chunks = [{"id": c.id, "content": c.content}
                      for c in orm_chunks
                      if c.chunk_type in ("child", "text") and c.content and c.content.strip()]
        if idx_chunks:
            from app.retrieval.dense_retriever import DenseRetriever
            from app.retrieval.milvus_client import MilvusClient
            dense = DenseRetriever()
            milvus = MilvusClient()
            milvus.connect()
            milvus._ensure_partition(kb_id)

            # Batch embeddings in groups of 10 (DashScope limit)
            batch_size = 10
            for i in range(0, len(idx_chunks), batch_size):
                batch = idx_chunks[i:i + batch_size]
                contents = [c["content"] for c in batch]
                embeddings = dense.embed_documents(contents)
                if embeddings and len(embeddings) == len(batch):
                    milvus.insert_vectors(
                        chunk_ids=[c["id"] for c in batch],
                        embeddings=embeddings, contents=contents, kb_id=kb_id,
                        security_levels=[kb_security] * len(batch),
                        departments=[kb_dept] * len(batch),
                    )

        # 5. BM25
        from app.retrieval import shared_bm25
        from app.retrieval.bm25_retriever import build_bm25_index_from_db
        from app.services.chunk_crud import ChunkCRUD
        ccrud = ChunkCRUD()
        indexable = ccrud.sync_get_indexable(session, kb_id)
        build_bm25_index_from_db(kb_id, indexable, shared_bm25)

        # 6. Update status -> completed
        session.execute(sql_update(Document).where(Document.id == doc_id).values(
            status="completed", chunk_count=len(chunks), is_indexed=True))
        session.execute(sql_update(KnowledgeBase).where(KnowledgeBase.id == kb_id).values(
            document_count=KnowledgeBase.document_count + 1,
            chunk_count=KnowledgeBase.chunk_count + len(chunks)))
        session.commit()

        # Cleanup temp
        try:
            os.unlink(file_path)
        except Exception:
            pass

        return {"status": "completed", "doc_id": doc_id, "chunk_count": len(chunks)}

    except Exception as exc:
        logger.error("Document processing failed: %s", exc)
        try:
            session.execute(sql_update(Document).where(Document.id == doc_id).values(
                status="failed", error_message=str(exc)[:500]))
            session.commit()
        except Exception:
            pass
        raise self.retry(exc=exc)
    finally:
        session.close()


@celery_app.task
def rebuild_index_task(kb_id: str):
    """Rebuild BM25 index from DB."""
    session = _get_sync_session()
    try:
        from app.retrieval import shared_bm25
        from app.retrieval.bm25_retriever import build_bm25_index_from_db
        from app.services.chunk_crud import ChunkCRUD
        ccrud = ChunkCRUD()
        indexable = ccrud.sync_get_indexable(session, kb_id)
        build_bm25_index_from_db(kb_id, indexable, shared_bm25)
        return {"status": "completed", "kb_id": kb_id, "chunk_count": len(indexable)}
    finally:
        session.close()
