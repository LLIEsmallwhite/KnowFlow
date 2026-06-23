"""
对话 API 路由

提供核心的 RAG 问答接口：
- POST /chat: 普通问答（全量返回）
- POST /chat/stream: SSE 流式问答
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


# ─── 请求/响应模型 ───

class ChatRequest(BaseModel):
    """问答请求"""
    query: str = Field(..., description="用户查询文本", min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None, description="会话 ID（新会话不传）")
    kb_ids: List[str] = Field(default_factory=list, description="知识库 ID 列表")
    stream: bool = Field(False, description="是否流式返回")
    enable_web_search: bool = Field(False, description="是否启用联网搜索")
    enable_memory: bool = Field(True, description="是否启用记忆压缩")
    images: Optional[List[str]] = Field(None, description="图片 Base64 / URL 列表")
    top_k: int = Field(10, description="返回的上下文数量", ge=1, le=50)
    temperature: float = Field(0.1, description="LLM 温度", ge=0.0, le=2.0)


class KnowledgeRef(BaseModel):
    """知识引用"""
    chunk_id: str
    content_preview: str = Field(..., max_length=200)
    score: float
    document_title: str = ""
    chunk_index: int = 0


class ChatResponse(BaseModel):
    """问答响应"""
    answer: str
    session_id: str
    knowledge_refs: List[KnowledgeRef] = Field(default_factory=list)
    token_usage: dict = Field(default_factory=dict)
    search_info: dict = Field(default_factory=dict)


# ─── API 端点 ───

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    知识库问答接口

    完整流程：
    Query Understanding → Hybrid Search → Dynamic RRF →
    Rerank → Context Merge → Memory Compress → LLM Generate

    Args:
        request: 问答请求参数

    Returns:
        ChatResponse: 回答 + 引用 + Token 统计
    """
    logger.info(f"Chat request: query='{request.query[:80]}...', kb_ids={request.kb_ids}")

    # ── 骨架实现：返回占位响应 ──
    # 后续步骤会将 LangGraph rag_pipeline 集成到此端点
    return ChatResponse(
        answer=f"收到您的问题：「{request.query}」\n\n"
               f"（KnowFlow RAG Pipeline 正在建设中，将在后续版本提供完整检索与生成能力）",
        session_id=request.session_id or "new_session_placeholder",
        knowledge_refs=[],
        token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        search_info={
            "rrf_weights": {"vector": 0.7, "keyword": 0.3},
            "vector_hits": 0,
            "keyword_hits": 0,
            "reranked_count": 0,
        },
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE 流式问答接口

    使用 Server-Sent Events 逐步返回 LLM 生成的 Token。
    前端可实时展示回答内容。

    响应格式: text/event-stream
    """
    logger.info(f"Chat stream request: query='{request.query[:80]}...'")

    async def event_generator():
        import json
        # 模拟流式输出（后续接入 LangGraph rag_pipeline.stream()）
        demo_answer = (
            f"正在为您处理：「{request.query}」...\n\n"
            f"将搜索知识库: {request.kb_ids or '全部'}\n"
            f"使用动态 RRF 融合检索 + Cross-Encoder Rerank\n\n"
            f"（完整流式输出将在后续版本实现）"
        )
        for char in demo_answer:
            yield f"data: {json.dumps({'token': char})}\n\n"
            import asyncio
            await asyncio.sleep(0.02)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
