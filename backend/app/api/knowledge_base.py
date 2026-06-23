"""
Knowledge Base Management API

Provides CRUD for knowledge bases, document upload, and search test.
Uses PostgreSQL for persistence via async SQLAlchemy sessions.
"""

import uuid
import os
import tempfile
import hashlib
import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import settings
from app.services.doc_service import doc_service
from app.services.kb_crud import kb_crud
from app.services.document_crud import doc_crud
from app.services.chunk_crud import chunk_crud
from app.graph.rag_pipeline import invoke_rag_pipeline
from app.retrieval.bm25_retriever import build_bm25_index_from_db
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.milvus_client import MilvusClient
from app.retrieval import shared_bm25 as _bm25

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["Knowledge Base"])

# ─── Global retrievers ───
_dense: Optional[DenseRetriever] = None
_milvus: Optional[MilvusClient] = None


def _get_dense():
    global _dense
    if _dense is None:
        _dense = DenseRetriever()
    return _dense


def _get_milvus():
    global _milvus
    if _milvus is None:
        _milvus = MilvusClient()
        _milvus.connect()
    return _milvus


# ─── Models ───

class KBCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    kb_type: str = Field("document", pattern="^(document|faq|wiki)$")
    chunk_size: int = Field(512, ge=100, le=4000)
    chunk_overlap: int = Field(80, ge=0, le=500)


class KBUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class KBResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    kb_type: str = "document"
    document_count: int = 0
    chunk_count: int = 0
    created_at: Optional[str] = None


class DocumentResponse(BaseModel):
    id: str
    title: str
    file_name: str
    file_type: str
    status: str
    chunk_count: int
    created_at: Optional[str] = None


class KBSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    kb_ids: Optional[List[str]] = None
    top_k: int = Field(10, ge=1, le=100)


# ─── KB CRUD ───

@router.get("", response_model=List[KBResponse])
async def list_knowledge_bases(
    db: AsyncSession = Depends(get_db),
):
    """List all active knowledge bases."""
    kbs = await kb_crud.list(db)
    return [
        KBResponse(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            kb_type=kb.kb_type,
            document_count=kb.document_count,
            chunk_count=kb.chunk_count,
            created_at=kb.created_at.isoformat() if kb.created_at else None,
        )
        for kb in kbs
    ]


@router.post("", response_model=KBResponse, status_code=201)
async def create_knowledge_base(
    req: KBCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new knowledge base."""
    kb = await kb_crud.create(
        db,
        name=req.name,
        description=req.description,
        kb_type=req.kb_type,
    )
    logger.info("KB created: id=%s, name=%s", kb.id, req.name)
    return KBResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        kb_type=kb.kb_type,
        created_at=kb.created_at.isoformat() if kb.created_at else None,
    )


@router.get("/{kb_id}", response_model=KBResponse)
async def get_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get knowledge base details."""
    kb = await kb_crud.get(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return KBResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        kb_type=kb.kb_type,
        document_count=kb.document_count,
        chunk_count=kb.chunk_count,
        created_at=kb.created_at.isoformat() if kb.created_at else None,
    )


@router.patch("/{kb_id}", response_model=KBResponse)
async def update_knowledge_base(
    kb_id: str,
    req: KBUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update knowledge base info."""
    kb = await kb_crud.get(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    kb = await kb_crud.update(
        db, kb_id,
        name=req.name,
        description=req.description,
    )
    return KBResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        kb_type=kb.kb_type,
        document_count=kb.document_count,
        chunk_count=kb.chunk_count,
        created_at=kb.created_at.isoformat() if kb.created_at else None,
    )


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a knowledge base (cascade: documents, chunks, indices)."""
    kb = await kb_crud.get(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # Delete chunks from DB
    await chunk_crud.delete_by_kb(db, kb_id)
    # Soft-delete KB
    await kb_crud.delete(db, kb_id)
    # Clean up BM25 index
    _bm25.remove_index(kb_id)
    # Clean up Milvus partition
    try:
        _get_milvus().drop_partition(kb_id)
    except Exception as e:
        logger.warning("Milvus partition cleanup failed: %s", e)
    return {"status": "deleted", "kb_id": kb_id}


# ─── Document Management ───

@router.post("/{kb_id}/documents")
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document to a knowledge base.

    Pipeline: save temp → parse → chunk → embed → Milvus + BM25 → DB persistence
    """
    kb = await kb_crud.get(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # Validate file type
    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    if not doc_service.is_supported(ext):
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    # Save to temp file
    content_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content_bytes)
        tmp_path = tmp.name

    doc_title = title or file.filename or "untitled"
    file_size = len(content_bytes)
    file_hash = hashlib.sha256(content_bytes).hexdigest()
    logger.info("Upload: kb=%s, file=%s, size=%d, type=%s", kb_id, doc_title, file_size, ext)

    # Create Document record in DB
    doc = await doc_crud.create(
        db,
        knowledge_base_id=kb_id,
        title=doc_title,
        file_name=file.filename or "untitled",
        file_type=ext,
        file_path=tmp_path,  # For now, store local path (will be MinIO later)
        file_size=file_size,
        file_hash=file_hash,
    )
    await doc_crud.update_status(db, doc.id, status="processing")

    try:
        # Parse and chunk
        parsed_doc, chunks = doc_service.process_document_pipeline(
            file_path=tmp_path,
            file_type=ext,
        )

        # Build chunk records for DB
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dicts.append({
                "document_id": doc.id,
                "knowledge_base_id": kb_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "content_hash": chunk.metadata.get("content_hash") if chunk.metadata else None,
                "metadata": chunk.metadata or {},
                "parent_chunk_id": getattr(chunk, "parent_chunk_id", None),
                "start_at": getattr(chunk, "start_at", None),
                "end_at": getattr(chunk, "end_at", None),
            })

        # Persist chunks to DB
        orm_chunks = await chunk_crud.bulk_create(db, chunk_dicts)

        # Index into Milvus (child and text chunks only)
        try:
            milvus = _get_milvus()
            milvus._ensure_partition(kb_id)

            idx_chunks = [
                {"id": c.id, "content": c.content}
                for c in orm_chunks
                if c.chunk_type in ("child", "text") and c.content and c.content.strip()
            ]
            if idx_chunks:
                contents = [c["content"] for c in idx_chunks]
                embeddings = _get_dense().embed_documents(contents)
                if embeddings and len(embeddings) == len(idx_chunks):
                    milvus.insert_vectors(
                        chunk_ids=[c["id"] for c in idx_chunks],
                        embeddings=embeddings,
                        contents=contents,
                        kb_id=kb_id,
                    )
                    # Mark as indexed
                await chunk_crud.mark_indexed(db, [c["id"] for c in idx_chunks])
        except Exception as e:
            logger.warning("Milvus indexing failed (non-fatal): %s", e)

        # Update BM25 index from DB
        indexable = await chunk_crud.get_indexable(db, kb_id)
        logger.info("Building BM25 for KB %s: %d indexable chunks", kb_id, len(indexable))
        if indexable:
            logger.info("Sample chunk[0]: id=%s, type=%s, preview=%s",
                        indexable[0].get("id", "?"),
                        indexable[0].get("chunk_type", "?"),
                        indexable[0].get("content", "")[:100])
        build_bm25_index_from_db(kb_id, indexable, _bm25)
        logger.info("BM25 built: KB=%s, index_size=%d, all_indexed=%s",
                    kb_id, _bm25.get_index_size(kb_id), _bm25.get_indexed_kbs())

        # Update document status
        await doc_crud.update_status(
            db, doc.id,
            status="completed",
            chunk_count=len(chunks),
            is_indexed=True,
        )

        # Update KB counts
        await kb_crud.update_counts(db, kb_id, doc_delta=1, chunk_delta=len(chunks))

        logger.info("Document processed: %s -> %d chunks", doc_title, len(chunks))
        return {
            "status": "completed",
            "kb_id": kb_id,
            "doc_id": doc.id,
            "filename": file.filename,
            "title": doc_title,
            "chunk_count": len(chunks),
            "content_length": len(parsed_doc.content),
        }

    except Exception as e:
        logger.error("Document processing failed: %s", e, exc_info=True)
        await doc_crud.update_status(db, doc.id, status="failed", error_message=str(e)[:500])
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)[:200]}")

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.get("/{kb_id}/documents", response_model=List[DocumentResponse])
async def list_documents(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List all documents in a knowledge base."""
    kb = await kb_crud.get(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    docs = await doc_crud.list_by_kb(db, kb_id)
    return [
        DocumentResponse(
            id=d.id,
            title=d.title,
            file_name=d.file_name,
            file_type=d.file_type,
            status=d.status,
            chunk_count=d.chunk_count,
            created_at=d.created_at.isoformat() if d.created_at else None,
        )
        for d in docs
    ]


@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_document(
    kb_id: str,
    doc_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and its chunks from a knowledge base."""
    doc = await doc_crud.get(db, doc_id)
    if not doc or doc.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")

    chunk_count = await chunk_crud.delete_by_doc(db, doc_id)
    await doc_crud.delete(db, doc_id)
    await kb_crud.update_counts(db, kb_id, doc_delta=-1, chunk_delta=-chunk_count)

    # Rebuild BM25 index
    indexable = await chunk_crud.get_indexable(db, kb_id)
    logger.info("Rebuilding BM25 after document delete for KB %s: %d chunks", kb_id, len(indexable))
    build_bm25_index_from_db(kb_id, indexable, _bm25)

    logger.info("Document deleted: %s, chunks removed: %d", doc_id, chunk_count)
    return {"status": "deleted", "doc_id": doc_id, "chunks_removed": chunk_count}


# ─── Debug ───

@router.get("/debug/bm25")
async def debug_bm25():
    """Debug endpoint: inspect BM25 index state."""
    return {
        "indexed_kbs": _bm25.get_indexed_kbs(),
        "index_sizes": {kb: _bm25.get_index_size(kb) for kb in _bm25.get_indexed_kbs()},
    }


@router.post("/debug/revectorize")
async def revectorize_kb(kb_id: str, db: AsyncSession = Depends(get_db)):
    """Re-embed all chunks in a KB and insert into Milvus."""
    from app.services.chunk_crud import ChunkCRUD
    ccrud = ChunkCRUD()
    all_chunks = await ccrud.get_by_kb(db, kb_id)

    idx_chunks = [
        {"id": c.id, "content": c.content}
        for c in all_chunks
        if c.chunk_type in ("child", "text") and c.content and c.content.strip()
    ]
    if not idx_chunks:
        return {"status": "no chunks to vectorize"}

    milvus = _get_milvus()
    milvus._ensure_partition(kb_id)

    contents = [c["content"] for c in idx_chunks]
    batch_size = 20
    total = 0
    for i in range(0, len(contents), batch_size):
        batch_contents = contents[i:i + batch_size]
        batch_ids = [idx_chunks[j]["id"] for j in range(i, min(i + batch_size, len(idx_chunks)))]
        embeddings = _get_dense().embed_documents(batch_contents)
        if embeddings and len(embeddings) == len(batch_contents):
            milvus.insert_vectors(
                chunk_ids=batch_ids,
                embeddings=embeddings,
                contents=batch_contents,
                kb_id=kb_id,
            )
            total += len(batch_contents)

    await ccrud.mark_indexed(db, [c["id"] for c in idx_chunks])
    return {"status": "revectorized", "total": total, "kb_id": kb_id}


@router.post("/debug/rebuild-bm25")
async def rebuild_all_bm25(db: AsyncSession = Depends(get_db)):
    """Rebuild BM25 indexes for all KBs from DB chunks."""
    kbs = await kb_crud.list(db)
    rebuilt = {}
    for kb in kbs:
        indexable = await chunk_crud.get_indexable(db, kb.id)
        build_bm25_index_from_db(kb.id, indexable, _bm25)
        rebuilt[kb.id] = len(indexable)
    return {"rebuilt": rebuilt, "indexed_kbs": _bm25.get_indexed_kbs()}


# ─── Search ───

@router.post("/search")
async def search_knowledge_bases(
    req: KBSearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Test retrieval without LLM generation."""
    from app.retrieval.hybrid_search import HybridSearchOrchestrator
    from app.retrieval.dynamic_rrf import DynamicRRF
    from app.retrieval.dedup import MultiLevelDeduplicator

    if not req.query.strip():
        return {"query": req.query, "results": [], "message": "查询为空"}

    import time
    start = time.time()

    orchestrator = HybridSearchOrchestrator(dense_retriever=_get_dense(), bm25_retriever=_bm25)
    hr = orchestrator.search(
        query=req.query,
        kb_ids=req.kb_ids,
        vector_top_k=50,
        keyword_top_k=50,
    )

    weights = {"vector": 0.7, "keyword": 0.3}
    results = []

    if hr.vector_results or hr.keyword_results:
        rrf = DynamicRRF()
        fused = rrf.fuse(req.query, hr.vector_results, hr.keyword_results)
        wc_result = rrf.weight_calc.compute(req.query, hr.vector_results, hr.keyword_results)
        weights = {"vector": wc_result.vector, "keyword": wc_result.keyword}

        dedup = MultiLevelDeduplicator()
        fused, stats = dedup.deduplicate(fused)

        for i, r in enumerate(fused[:req.top_k]):
            results.append({
                "rank": i + 1,
                "chunk_id": r.chunk_id[:16] if hasattr(r, 'chunk_id') else "",
                "content_preview": (r.content[:300] + "...") if hasattr(r, 'content') and r.content else "",
                "rrf_score": getattr(r, 'rrf_score', 0),
                "source": getattr(r, 'source', ''),
                "vector_rank": getattr(r, 'vector_rank', -1),
                "keyword_rank": getattr(r, 'keyword_rank', -1),
            })

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "query": req.query,
        "results": results,
        "rrf_weights": weights,
        "vector_hits": len(hr.vector_results),
        "keyword_hits": len(hr.keyword_results),
        "search_time_ms": elapsed_ms,
    }
