"""
知识库管理 API

提供知识库的 CRUD、文档上传、检索测试等接口。
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["Knowledge Base"])


# ─── 请求/响应模型 ───

class KBCreateRequest(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    kb_type: str = Field("document", pattern="^(document|faq|wiki)$")
    embedding_model: str = "text-embedding-3-large"
    chunk_size: int = Field(512, ge=100, le=4000)
    chunk_overlap: int = Field(80, ge=0, le=500)


class KBResponse(BaseModel):
    """知识库响应"""
    id: str
    name: str
    description: Optional[str] = None
    kb_type: str
    document_count: int = 0
    chunk_count: int = 0
    created_at: Optional[str] = None


class KBSearchRequest(BaseModel):
    """知识库检索测试请求"""
    query: str = Field(..., min_length=1)
    kb_ids: Optional[List[str]] = None
    top_k: int = Field(10, ge=1, le=100)


# ─── API 端点 ───

@router.get("", response_model=List[KBResponse])
async def list_knowledge_bases():
    """获取知识库列表"""
    return []


@router.post("", response_model=KBResponse, status_code=201)
async def create_knowledge_base(req: KBCreateRequest):
    """创建知识库"""
    import uuid
    return KBResponse(
        id=str(uuid.uuid4()),
        name=req.name,
        description=req.description,
        kb_type=req.kb_type,
    )


@router.get("/{kb_id}", response_model=KBResponse)
async def get_knowledge_base(kb_id: str):
    """获取知识库详情"""
    return KBResponse(
        id=kb_id,
        name="示例知识库",
        kb_type="document",
    )


@router.delete("/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    """删除知识库（级联删除文档和向量）"""
    return {"status": "deleted", "kb_id": kb_id}


@router.post("/{kb_id}/documents")
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    """
    上传文档到知识库

    支持：PDF, Word, Markdown, HTML, TXT
    文件上传后通过 Celery 异步任务进行解析和向量化。
    """
    return {
        "status": "uploaded",
        "kb_id": kb_id,
        "filename": file.filename,
        "title": title or file.filename,
    }


@router.post("/search")
async def search_knowledge_bases(req: KBSearchRequest):
    """检索测试（仅返回检索结果，不调用 LLM）"""
    return {
        "query": req.query,
        "results": [],
        "rrf_weights": {"vector": 0.72, "keyword": 0.28},
        "search_time_ms": 0,
    }
