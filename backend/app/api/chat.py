"""
对话 API 路由

提供核心的 RAG 问答接口：
- POST /chat: 普通问答（全量返回）
- POST /chat/stream: SSE 流式问答
"""

import json
import asyncio
import logging
from typing import Optional, List
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graph.rag_pipeline import invoke_rag_pipeline  # 非流式
from app.core.config import settings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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


# ─── LLM 实例 ───
def _get_stream_llm():
    """获取流式 LLM 实例"""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        streaming=True,
    )


# ─── API 端点 ───

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    知识库问答接口（非流式）

    完整流程：
    Query Understanding → Hybrid Search → Dynamic RRF →
    Rerank → Context Merge → Memory Compress → LLM Generate
    """
    logger.info(f"Chat: query='{request.query[:80]}...', kb_ids={request.kb_ids}")

    result = invoke_rag_pipeline(
        query=request.query,
        kb_ids=request.kb_ids,
        session_id=request.session_id or "",
    )

    return ChatResponse(
        answer=result["answer"],
        session_id=request.session_id or "default",
        knowledge_refs=[
            KnowledgeRef(
                chunk_id=r.get("chunk_id", ""),
                content_preview=r.get("content_preview", "")[:200],
                score=r.get("score", 0.0),
            )
            for r in result.get("knowledge_refs", [])
        ],
        token_usage=result.get("token_usage", {}),
        search_info=result.get("search_info", {}),
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE 流式问答接口

    使用 Server-Sent Events 逐步返回 LLM 生成的 Token。
    先执行检索→RRF→Rerank→合并，然后用流式 LLM 生成回答。
    """
    logger.info(f"Chat stream: query='{request.query[:80]}...'")

    async def event_generator():
        query = request.query.strip()
        kb_ids = request.kb_ids

        # Step 1: 执行检索（非流式部分）
        search_info = {"rrf_weights": {}, "vector_hits": 0, "keyword_hits": 0}
        context_text = ""

        try:
            # 先跑检索管线（非 LLM 部分）
            from app.graph.rag_pipeline import _get_hybrid_search, _get_dynamic_rrf, _get_dedup, _get_reranker, _get_merger
            from app.memory.consolidator import MemoryConsolidator

            rewritten = query
            try:
                llm = _get_stream_llm()
                resp = llm.invoke([
                    SystemMessage(content="将用户查询改写为更完整的检索查询，只输出改写结果，不加解释。"),
                    HumanMessage(content=query),
                ])
                if resp.content:
                    rewritten = resp.content.strip()
            except Exception:
                pass

            # 混合检索
            orchestrator = _get_hybrid_search()
            hr = orchestrator.search(query=rewritten, kb_ids=kb_ids if kb_ids else None)
            search_info["vector_hits"] = len(hr.vector_results)
            search_info["keyword_hits"] = len(hr.keyword_results)

            # RRF 融合
            if hr.vector_results or hr.keyword_results:
                rrf = _get_dynamic_rrf()
                fused = rrf.fuse(rewritten, hr.vector_results, hr.keyword_results)
                dedup = _get_dedup()
                fused, _ = dedup.deduplicate(fused)
                search_info["rrf_weights"] = {
                    "vector": rrf.weight_calc.compute(rewritten, hr.vector_results, hr.keyword_results).vector,
                    "keyword": rrf.weight_calc.compute(rewritten, hr.vector_results, hr.keyword_results).keyword,
                }

                # Rerank
                if len(fused) > 3:
                    try:
                        reranker = _get_reranker()
                        fused = reranker.rerank(query=rewritten, candidates=fused, top_k=settings.RERANK_TOP_K)
                    except Exception:
                        fused = fused[:settings.RERANK_TOP_K]

                # 合并上下文
                merger = _get_merger()
                merged = merger.merge(fused, top_k=settings.RERANK_TOP_K)
                context_text = merger.format_for_llm(merged)

                yield f"data: {json.dumps({'type': 'search_info', 'data': search_info}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'search_info', 'data': search_info}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.warning(f"Search phase error (continuing without context): {e}")
            yield f"data: {json.dumps({'type': 'warning', 'data': f'检索阶段出错，将直接回答: {str(e)[:100]}'})}\n\n"

        # Step 2: LLM 流式生成
        try:
            stream_llm = _get_stream_llm()
            messages = []

            if context_text:
                system_prompt = (
                    "你是一个专业的企业知识库问答助手。请根据参考文档回答用户问题。\n"
                    "规则：优先基于文档，标注来源编号 [1][2] 等。文档无相关信息请明确说明。\n"
                )
                user_prompt = f"参考文档：\n\n{context_text}\n\n---\n用户问题：{query}\n\n请回答："
            else:
                system_prompt = "你是一个有帮助的企业知识库助手。请友好简洁地回答用户。"
                user_prompt = query

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            full_answer = ""
            async for chunk in stream_llm.astream(messages):
                if chunk.content:
                    full_answer += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.content}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)

            yield f"data: {json.dumps({'type': 'done', 'data': {'full_answer': full_answer, 'search_info': search_info}}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)[:300]})}\n\n"

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
