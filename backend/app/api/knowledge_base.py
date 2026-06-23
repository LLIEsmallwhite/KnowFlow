"""
知识库管理 API

提供知识库的 CRUD、文档上传、检索测试等接口。
开发阶段使用内存存储，生产环境切换为 PostgreSQL。
"""

import uuid
import os
import tempfile
import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.services.doc_service import doc_service
from app.graph.rag_pipeline import invoke_rag_pipeline
from app.retrieval.bm25_retriever import BM25Retriever, build_bm25_index_from_db
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.milvus_client import MilvusClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["Knowledge Base"])

# ─── 开发阶段内存存储（生产切换为 SQLAlchemy + PostgreSQL） ───
_kb_store: dict = {}       # kb_id → KBResponse
_doc_store: dict = {}      # doc_id → dict
_chunk_store: dict = {}    # kb_id → list of chunk dicts

# 全局检索器实例
_bm25 = BM25Retriever()
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
        _milvus.create_collection()
    return _milvus


# ─── 模型 ───

class KBCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    kb_type: str = Field("document", pattern="^(document|faq|wiki)$")
    chunk_size: int = Field(512, ge=100, le=4000)
    chunk_overlap: int = Field(80, ge=0, le=500)


class KBResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    kb_type: str = "document"
    document_count: int = 0
    chunk_count: int = 0
    created_at: Optional[str] = None


class KBSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    kb_ids: Optional[List[str]] = None
    top_k: int = Field(10, ge=1, le=100)


# ─── API 端点 ───

@router.get("", response_model=List[KBResponse])
async def list_knowledge_bases():
    """获取所有知识库"""
    return list(_kb_store.values())


@router.post("", response_model=KBResponse, status_code=201)
async def create_knowledge_base(req: KBCreateRequest):
    """创建知识库"""
    kb_id = str(uuid.uuid4())
    kb = KBResponse(
        id=kb_id,
        name=req.name,
        description=req.description,
        kb_type=req.kb_type,
        created_at=datetime.now().isoformat(),
    )
    _kb_store[kb_id] = kb
    _chunk_store[kb_id] = []
    logger.info(f"KB created: id={kb_id}, name={req.name}")
    return kb


@router.get("/{kb_id}", response_model=KBResponse)
async def get_knowledge_base(kb_id: str):
    """获取知识库详情"""
    kb = _kb_store.get(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb


@router.delete("/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    """删除知识库（级联删除文档和索引）"""
    if kb_id not in _kb_store:
        raise HTTPException(status_code=404, detail="知识库不存在")
    del _kb_store[kb_id]
    _chunk_store.pop(kb_id, None)
    _bm25.remove_index(kb_id)
    try:
        _get_milvus().drop_partition(kb_id)
    except Exception as e:
        logger.warning(f"Milvus partition cleanup failed: {e}")
    return {"status": "deleted", "kb_id": kb_id}


@router.post("/{kb_id}/documents")
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    """
    上传文档到知识库

    完整流程：
    1. 保存上传文件到临时目录
    2. 文档解析为纯文本
    3. 自适应分块
    4. 向量化写入 Milvus
    5. 构建 BM25 索引
    """
    if kb_id not in _kb_store:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 检查文件类型
    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    if not doc_service.is_supported(ext):
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        content_bytes = await file.read()
        tmp.write(content_bytes)
        tmp_path = tmp.name

    doc_title = title or file.filename or "untitled"
    logger.info(f"Upload: kb={kb_id}, file={doc_title}, size={len(content_bytes)} bytes, type={ext}")

    try:
        # 解析文档
        doc, chunks = doc_service.process_document_pipeline(
            file_path=tmp_path,
            file_type=ext,
        )

        # 生成 Chunk ID
        chunk_records = []
        for i, chunk in enumerate(chunks):
            cid = str(uuid.uuid4())
            chunk.chunk_id = cid  # 给 AdaptiveChunker 的 Chunk 加 id
            chunk_id_val = getattr(chunk, 'chunk_id', cid) or cid
            record = {
                "id": chunk_id_val,
                "chunk_id": chunk_id_val,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "document_id": kb_id,  # 简化：doc 不在 DB 时用 kb_id
                "doc_id": kb_id,
                "metadata": chunk.metadata or {},
            }
            chunk_records.append(record)

        # 写入 Milvus（仅索引 child 和普通 text chunk）
        try:
            milvus = _get_milvus()
            milvus.ensure_partition(kb_id)

            idx_chunks = [c for c in chunk_records if c["chunk_type"] in ("child", "text")]
            if idx_chunks:
                contents = [c["content"] for c in idx_chunks]
                embeddings = _get_dense().embed_documents(contents)
                milvus.insert_vectors(
                    chunk_ids=[c["chunk_id"] for c in idx_chunks],
                    embeddings=embeddings,
                    contents=contents,
                    kb_id=kb_id,
                )
        except Exception as e:
            logger.warning(f"Milvus indexing failed (non-fatal): {e}")

        # 更新 BM25 索引
        _chunk_store[kb_id] = chunk_records
        build_bm25_index_from_db(kb_id, chunk_records, _bm25)

        # 更新统计
        _kb_store[kb_id].document_count += 1
        _kb_store[kb_id].chunk_count += len(chunks)

        logger.info(f"Document processed: {doc_title} → {len(chunks)} chunks")

        return {
            "status": "completed",
            "kb_id": kb_id,
            "filename": file.filename,
            "title": doc_title,
            "chunk_count": len(chunks),
            "content_length": len(doc.content),
        }

    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)[:200]}")

    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.post("/search")
async def search_knowledge_bases(req: KBSearchRequest):
    """
    检索测试

    仅执行检索（不调用 LLM 生成），返回检索结果和 RRF 权重用于调试。
    """
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
        weights = {
            "vector": rrf.weight_calc.compute(req.query, hr.vector_results, hr.keyword_results).vector,
            "keyword": rrf.weight_calc.compute(req.query, hr.vector_results, hr.keyword_results).keyword,
        }

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
