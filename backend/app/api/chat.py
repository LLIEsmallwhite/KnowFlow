"""
Chat API

Provides RAG Q&A endpoints with DB persistence:
- POST /chat: synchronous Q&A
- POST /chat/stream: SSE streaming Q&A

Sessions and messages are persisted to PostgreSQL.
"""

import json
import asyncio
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.database import get_db
from app.core.config import settings
from app.graph.rag_pipeline import invoke_rag_pipeline
from app.services.session_crud import session_crud
from app.services.message_crud import message_crud

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["开始聊天"])

# Default user ID (until auth is implemented)
DEFAULT_USER_ID = "default-user"


# ─── Request/Response Models ───

class ChatRequest(BaseModel):
    """Q&A request."""
    query: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None)
    kb_ids: List[str] = Field(default_factory=list)
    stream: bool = Field(False)
    enable_web_search: bool = Field(False)
    enable_memory: bool = Field(True)
    images: Optional[List[str]] = Field(None)
    top_k: int = Field(10, ge=1, le=50)
    temperature: float = Field(0.1, ge=0.0, le=2.0)


class KnowledgeRef(BaseModel):
    chunk_id: str
    content_preview: str = Field(..., max_length=200)
    score: float
    document_title: str = ""
    chunk_index: int = 0


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    knowledge_refs: List[KnowledgeRef] = Field(default_factory=list)
    token_usage: dict = Field(default_factory=dict)
    search_info: dict = Field(default_factory=dict)


# ─── LLM Instance ───

def _get_stream_llm():
    callbacks = []
    if settings.LANGFUSE_ENABLED:
        from app.observability.langfuse_client import langfuse_manager
        cb = langfuse_manager.get_langchain_callback()
        if cb:
            callbacks.append(cb)
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        streaming=True,
        callbacks=callbacks if callbacks else None,
    )


# ─── Helpers ───

async def _ensure_session(
    db: AsyncSession,
    session_id: Optional[str],
    query: str,
) -> str:
    """Get or create a session. Returns session_id."""
    if session_id:
        session = await session_crud.get(db, session_id)
        if session:
            return session_id
    # Create new session with query as title
    title = query[:50] + ("..." if len(query) > 50 else "")
    session = await session_crud.create(db, user_id=DEFAULT_USER_ID, title=title)
    return session.id


# ─── Endpoints ───

@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Synchronous RAG Q&A with DB persistence."""
    logger.info("Chat: query='%s...', kb_ids=%s", request.query[:80], request.kb_ids)

    # Ensure session exists
    session_id = await _ensure_session(db, request.session_id, request.query)

    # Save user message
    await message_crud.create(db, session_id=session_id, role="user",
                              content=request.query)

    # Run RAG pipeline
    result = invoke_rag_pipeline(
        query=request.query,
        kb_ids=request.kb_ids,
        session_id=session_id,
    )

    # Save assistant message
    token_usage = result.get("token_usage", {}) or {}
    total_tokens = sum(token_usage.values())
    await message_crud.create(
        db, session_id=session_id, role="assistant",
        content=result["answer"],
        knowledge_references=result.get("knowledge_refs", []),
        token_usage=token_usage,
    )
    await session_crud.increment_message_count(
        db, session_id, delta=2, token_delta=total_tokens,
    )

    return ChatResponse(
        answer=result["answer"],
        session_id=session_id,
        knowledge_refs=[
            KnowledgeRef(
                chunk_id=r.get("chunk_id", ""),
                content_preview=r.get("content_preview", "")[:200],
                score=r.get("score", 0.0),
            )
            for r in result.get("knowledge_refs", [])
        ],
        token_usage=token_usage,
        search_info=result.get("search_info", {}),
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """SSE streaming RAG Q&A with DB persistence."""
    logger.info("Chat stream: query='%s...'", request.query[:80])

    # Ensure session BEFORE generator (needs a running event loop)
    session_id = request.session_id
    if not session_id:
        title = request.query[:50] + ("..." if len(request.query) > 50 else "")
        session = await session_crud.create(db, user_id=DEFAULT_USER_ID, title=title)
        session_id = session.id
    else:
        existing = await session_crud.get(db, session_id)
        if not existing:
            title = request.query[:50] + ("..." if len(request.query) > 50 else "")
            session = await session_crud.create(db, user_id=DEFAULT_USER_ID, title=title)
            session_id = session.id

    # Save user message
    await message_crud.create(db, session_id=session_id, role="user",
                              content=request.query)

    async def event_generator():
        query = request.query.strip()
        kb_ids = request.kb_ids

        from datetime import datetime
        now_str = datetime.now().strftime("%Y年%m月%d日 %A")
        time_hint = f"\n[系统时间: {now_str}]"

        logger.info("Chat stream: query='%s', kb_ids=%s", query[:80], kb_ids)

        # Langfuse tracing
        from app.observability.langfuse_client import langfuse_manager

        search_info = {"rrf_weights": {}, "vector_hits": 0, "keyword_hits": 0, "web_results": 0}
        context_text = ""
        knowledge_refs = []

        # Web search disabled (blocked in China, use LLM's knowledge instead)
        # if request.enable_web_search: ...

        try:
            from app.graph.rag_pipeline import (
                _get_hybrid_search, _get_dynamic_rrf, _get_dedup,
                _get_reranker, _get_merger,
            )
            from app.retrieval import shared_bm25

            logger.info("BM25 indexed KBs before search: %s",
                        shared_bm25.get_indexed_kbs())

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

            # Build permission pre-filter for RBAC
            from app.core.permissions import build_permission_filter
            from app.core.dependencies import get_current_user
            perm_filter = build_permission_filter(None)  # Will be user-aware when auth wired
            logger.info("Permission filter: %s", perm_filter.milvus_expr)

            orchestrator = _get_hybrid_search()
            hr = orchestrator.search(
                query=rewritten, kb_ids=kb_ids if kb_ids else None,
                permission_expr=perm_filter.milvus_expr,
            )
            search_info["vector_hits"] = len(hr.vector_results)
            search_info["keyword_hits"] = len(hr.keyword_results)

            # Trace retrieval stats
            with langfuse_manager.trace("hybrid-search", session_id=session_id,
                                         metadata={"query": query, "rewritten": rewritten,
                                                   "vector_hits": search_info["vector_hits"],
                                                   "keyword_hits": search_info["keyword_hits"]}):
                pass  # Auto-logged via context manager

            if hr.vector_results or hr.keyword_results:
                rrf = _get_dynamic_rrf()
                fused = rrf.fuse(rewritten, hr.vector_results, hr.keyword_results)
                dedup = _get_dedup()
                fused, _ = dedup.deduplicate(fused)
                wc_result = rrf.weight_calc.compute(rewritten, hr.vector_results, hr.keyword_results)
                search_info["rrf_weights"] = {
                    "vector": wc_result.vector,
                    "keyword": wc_result.keyword,
                }

                if len(fused) > 3:
                    try:
                        reranker = _get_reranker()
                        fused = reranker.rerank(query=rewritten, candidates=fused,
                                                top_k=settings.RERANK_TOP_K)
                    except Exception:
                        fused = fused[:settings.RERANK_TOP_K]

                merger = _get_merger()
                merged = merger.merge(fused, top_k=settings.RERANK_TOP_K)
                context_text = merger.format_for_llm(merged)

                # Build valid chunk_id set from DB (single source of truth)
                from app.models.chunk import Chunk as ChunkModel
                from sqlalchemy import select as sql_select
                valid_ids = set()
                chunk_doc_map = {}
                # Get all indexable chunks with doc names from DB
                all_db_chunks = await db.execute(
                    sql_select(ChunkModel.id, ChunkModel.document_id, ChunkModel.content)
                    .where(ChunkModel.knowledge_base_id.in_(kb_ids) if kb_ids else True)
                )
                doc_cache = {}
                for row in all_db_chunks:
                    cid, doc_id, content = row[0], row[1], row[2]
                    valid_ids.add(cid)
                    if doc_id not in doc_cache:
                        from app.models.document import Document as DocModel
                        dresult = await db.execute(
                            sql_select(DocModel.title).where(DocModel.id == doc_id)
                        )
                        doc_cache[doc_id] = dresult.scalar_one_or_none() or ""
                    chunk_doc_map[cid] = doc_cache[doc_id]
                    chunk_doc_map[(content or "")[:80]] = doc_cache[doc_id]

                seen_content = set()
                for ctx in merged:
                    for cid in ctx.chunk_ids[:3]:
                        # Skip chunks not in DB (stale Milvus data)
                        if cid not in valid_ids and cid not in chunk_doc_map:
                            continue
                        content_sig = ctx.content[:100]
                        if content_sig in seen_content:
                            continue
                        seen_content.add(content_sig)

                        display_name = chunk_doc_map.get(cid, "")
                        if not display_name:
                            display_name = chunk_doc_map.get(ctx.content[:80], "")

                        knowledge_refs.append({
                            "chunk_id": cid,
                            "content_preview": ctx.content[:200],
                            "score": ctx.relevance_score,
                            "doc_title": display_name,
                            "doc_filename": display_name,
                            "full_content": ctx.content,
                        })

                yield f"data: {json.dumps({'type': 'search_info', 'data': search_info}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'knowledge_refs', 'data': knowledge_refs}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'search_info', 'data': search_info}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.warning("Search phase error (continuing without context): %s", e)
            yield f"data: {json.dumps({'type': 'warning', 'data': f'检索阶段出错，将直接回答: {str(e)[:100]}'})}\n\n"

        # Step 2: Stream LLM generation
        full_answer = ""
        try:
            stream_llm = _get_stream_llm()

            if context_text:
                system_prompt = (
                    "你是一个专业的企业知识库问答助手。请**仅根据**参考资料回答。\n"
                    "规则：严格基于参考资料，优先知识库文档，其次网络结果。"
                )
                user_prompt = f"参考资料：\n\n{context_text}\n\n---\n用户问题：{query}{time_hint}\n\n请回答："
            elif request.enable_web_search:
                system_prompt = (
                    "你是一个企业知识库问答助手。用户已启用联网搜索。\n"
                    "请用你自己的知识回答用户问题（如日期、常识、新闻等）。"
                )
                user_prompt = f"{query}{time_hint}"
            else:
                system_prompt = (
                    "你是一个企业知识库问答助手。当前没有检索到相关资料。\n"
                    "拒绝回答事实性问题，建议上传文档。闲聊和常识可直接回复。"
                )
                user_prompt = f"{query}{time_hint}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            async for chunk in stream_llm.astream(messages):
                if chunk.content:
                    full_answer += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.content}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.005)  # Yield control

            yield f"data: {json.dumps({'type': 'done', 'data': {'full_answer': full_answer, 'search_info': search_info}}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error("LLM stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)[:300]})}\n\n"

        # Save assistant message after streaming completes
        try:
            async with db.bind.connect() as conn:
                from app.services.message_crud import MessageCRUD
                msg_crud = MessageCRUD()
                from sqlalchemy.ext.asyncio import AsyncSession as AS
                s = AS(conn)
                await session_crud.increment_message_count(s, session_id, delta=1)
                await msg_crud.create(s, session_id=session_id,
                                      role="assistant", content=full_answer,
                                      knowledge_references=knowledge_refs)
                await s.commit()
        except Exception as e:
            logger.error("Failed to persist assistant message: %s", e)

        # Trace completion
        try:
            with langfuse_manager.trace("rag-chat", session_id=session_id,
                                         metadata={"query": query,
                                                   "vector_hits": search_info.get("vector_hits", 0),
                                                   "keyword_hits": search_info.get("keyword_hits", 0),
                                                   "ref_count": len(knowledge_refs)},
                                         input_data={"answer": full_answer[:500]}):
                pass
        except Exception:
            pass

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
