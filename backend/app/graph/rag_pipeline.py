"""
RAG Pipeline — LangGraph StateGraph 实现

完整的 RAG 问答流水线，包含 7 个处理节点：

  QueryUnderstand → HybridSearch → DynamicRRF → Rerank
       → ContextMerge → [MemoryCompress] → Generate

流程由 LangGraph 的条件边控制：
- QueryUnderstand 判断是否需要检索
- ContextMerge 判断是否需要记忆压缩
- Generate 判断是否成功（失败 → 回退）

每个节点都调用真实的检索/RRF/Rerank/LLM 模块。
"""

import json
import logging
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.graph.states import RAGState
from app.retrieval.hybrid_search import HybridSearchOrchestrator
from app.retrieval.dynamic_rrf import DynamicRRF, RRFChunk
from app.retrieval.dedup import MultiLevelDeduplicator
from app.retrieval.reranker import create_reranker, NoOpReranker
from app.retrieval.context_merge import ContextMerger
from app.retrieval import shared_bm25
from app.memory.consolidator import MemoryConsolidator
from app.memory.token_estimator import TokenEstimator

logger = logging.getLogger(__name__)

# ─── 辅助函数 ───

def _extract_token_usage(resp) -> dict:
    """安全提取 token 用量（兼容 LangChain UsageMetadata 对象和 dict）"""
    try:
        um = resp.usage_metadata
        if um is None:
            return {}
        if isinstance(um, dict):
            return {
                "prompt_tokens": um.get("input_tokens", 0),
                "completion_tokens": um.get("output_tokens", 0),
            }
        return {
            "prompt_tokens": getattr(um, "input_tokens", 0) or 0,
            "completion_tokens": getattr(um, "output_tokens", 0) or 0,
        }
    except Exception:
        return {}

# ─── 全局复用的实例 ───
_llm: Optional[ChatOpenAI] = None
_hybrid_search: Optional[HybridSearchOrchestrator] = None
_dynamic_rrf: Optional[DynamicRRF] = None
_dedup: Optional[MultiLevelDeduplicator] = None
_reranker: Optional[NoOpReranker] = None
_merger: Optional[ContextMerger] = None
_consolidator: Optional[MemoryConsolidator] = None
_estimator: Optional[TokenEstimator] = None


def _get_llm() -> ChatOpenAI:
    """获取 LLM 实例（DeepSeek / Qwen 等 OpenAI 兼容接口）"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            streaming=True,
        )
    return _llm


def _get_hybrid_search() -> HybridSearchOrchestrator:
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearchOrchestrator(bm25_retriever=shared_bm25)
    return _hybrid_search


def _get_dynamic_rrf() -> DynamicRRF:
    global _dynamic_rrf
    if _dynamic_rrf is None:
        _dynamic_rrf = DynamicRRF(k=settings.RRF_K)
    return _dynamic_rrf


def _get_dedup() -> MultiLevelDeduplicator:
    global _dedup
    if _dedup is None:
        _dedup = MultiLevelDeduplicator()
    return _dedup


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = create_reranker(settings.RERANK_PROVIDER)
    return _reranker


def _get_merger() -> ContextMerger:
    global _merger
    if _merger is None:
        _merger = ContextMerger()
    return _merger


def _get_consolidator() -> MemoryConsolidator:
    global _consolidator
    if _consolidator is None:
        _consolidator = MemoryConsolidator()
    return _consolidator


def _get_estimator() -> TokenEstimator:
    global _estimator
    if _estimator is None:
        _estimator = TokenEstimator()
    return _estimator


# ═══════════════════════════════════════════════════════════
# 节点实现
# ═══════════════════════════════════════════════════════════

def query_understand_node(state: RAGState) -> RAGState:
    """节点 1: Query Understanding — 使用 LLM 改写查询"""
    query = state.get("query", "").strip()

    if not query:
        state["rewritten_query"] = query
        state["intent"] = "chitchat"
        state["needs_retrieval"] = False
        state["pipeline_stage"] = "query_understand"
        return state

    # ── 使用 LLM 改写查询（消除歧义、补全上下文） ──
    try:
        llm = _get_llm()
        history = state.get("history_messages", [])
        history_text = ""
        if history:
            recent = history[-4:]  # 最近 2 轮
            for m in recent:
                history_text += f"{m['role']}: {m.get('content', '')[:200]}\n"

        rewrite_prompt = (
            "你是一个查询优化助手。根据对话历史，将用户的查询改写为更完整、准确的检索查询。\n"
            "规则：\n"
            "1. 如果查询中有人称代词（它/他/这个/那个），根据历史补全为具体名词\n"
            "2. 如果查询很短（≤5字），根据历史推断用户意图并扩展\n"
            "3. 如果查询本身已很完整，直接返回原查询\n"
            "4. 只输出改写后的查询，不要加任何解释\n"
        )
        if history_text:
            rewrite_prompt += f"\n对话历史：\n{history_text}"

        resp = llm.invoke([
            SystemMessage(content=rewrite_prompt),
            HumanMessage(content=query),
        ])
        rewritten = resp.content.strip() if resp.content else query
        state["rewritten_query"] = rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
        state["rewritten_query"] = query

    # 意图分类
    chitchat_keywords = ["你好", "谢谢", "再见", "hello", "hi", "thanks", "你是谁"]
    if query.lower() in chitchat_keywords or len(query) <= 2:
        state["intent"] = "chitchat"
        state["needs_retrieval"] = False
    else:
        state["intent"] = "knowledge_qa"
        state["needs_retrieval"] = True

    state["pipeline_stage"] = "query_understand"
    logger.info(f"[Pipeline] QueryUnderstand: '{query[:60]}' → '{state['rewritten_query'][:60]}' intent={state['intent']}")
    return state


def hybrid_search_node(state: RAGState) -> RAGState:
    """节点 2: Hybrid Search — 并行向量 + BM25 检索"""
    query = state.get("rewritten_query", "")
    kb_ids = state.get("kb_ids", [])

    try:
        orchestrator = _get_hybrid_search()
        result = orchestrator.search(
            query=query,
            kb_ids=kb_ids if kb_ids else None,
            vector_top_k=settings.VECTOR_SEARCH_TOP_K,
            keyword_top_k=settings.KEYWORD_SEARCH_TOP_K,
            vector_threshold=settings.VECTOR_THRESHOLD,
            keyword_threshold=settings.KEYWORD_THRESHOLD,
        )
        state["vector_results"] = result.vector_results
        state["keyword_results"] = result.keyword_results
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        state["vector_results"] = []
        state["keyword_results"] = []

    state["pipeline_stage"] = "hybrid_search"
    logger.info(f"[Pipeline] HybridSearch: v={len(state['vector_results'])}, k={len(state['keyword_results'])}")
    return state


def dynamic_rrf_fusion_node(state: RAGState) -> RAGState:
    """节点 3: Dynamic RRF Fusion — 四因子自适应权重融合"""
    vector_results = state.get("vector_results", [])
    keyword_results = state.get("keyword_results", [])
    query = state.get("rewritten_query", "")

    if not vector_results and not keyword_results:
        state["fused_results"] = []
        state["rrf_weights"] = {"vector": 0.7, "keyword": 0.3}
        state["pipeline_stage"] = "rrf_fusion"
        return state

    try:
        rrf = _get_dynamic_rrf()
        fused = rrf.fuse(query, vector_results, keyword_results)

        # 去重
        dedup = _get_dedup()
        clean_results, stats = dedup.deduplicate(fused)

        state["fused_results"] = clean_results
        wc_result = rrf.weight_calc.compute(query, vector_results, keyword_results)
        state["rrf_weights"] = {
            "vector": wc_result.vector,
            "keyword": wc_result.keyword,
        }
    except Exception as e:
        logger.error(f"RRF fusion failed: {e}")
        state["fused_results"] = []
        state["rrf_weights"] = {"vector": 0.7, "keyword": 0.3}

    state["pipeline_stage"] = "rrf_fusion"
    logger.info(f"[Pipeline] RRF: fused={len(state['fused_results'])} chunks, weights={state['rrf_weights']}")
    return state


def rerank_node(state: RAGState) -> RAGState:
    """节点 4: Cross-Encoder Rerank — 精排"""
    fused = state.get("fused_results", [])
    query = state.get("rewritten_query", "")

    if len(fused) <= 3:
        state["reranked_results"] = fused
        state["pipeline_stage"] = "rerank"
        return state

    try:
        reranker = _get_reranker()
        reranked = reranker.rerank(
            query=query,
            candidates=fused,
            top_k=settings.RERANK_TOP_K,
            threshold=settings.RERANK_THRESHOLD,
        )
        state["reranked_results"] = reranked
    except Exception as e:
        logger.warning(f"Rerank failed, using fused results: {e}")
        state["reranked_results"] = fused[:settings.RERANK_TOP_K]

    state["pipeline_stage"] = "rerank"
    logger.info(f"[Pipeline] Rerank: {len(fused)} → {len(state['reranked_results'])}")
    return state


def context_merge_node(state: RAGState) -> RAGState:
    """节点 5: Context Merge — 合并相邻片段，格式化 LLM 上下文"""
    reranked = state.get("reranked_results", [])

    if not reranked:
        state["merged_contexts"] = []
        state["pipeline_stage"] = "context_merge"
        return state

    try:
        merger = _get_merger()
        merged = merger.merge(reranked, top_k=settings.RERANK_TOP_K)
        state["merged_contexts"] = merged
        state["formatted_context"] = merger.format_for_llm(merged)
    except Exception as e:
        logger.error(f"Context merge failed: {e}")
        state["merged_contexts"] = []
        state["formatted_context"] = ""

    state["pipeline_stage"] = "context_merge"
    logger.info(f"[Pipeline] Merge: {len(reranked)} chunks → {len(state['merged_contexts'])} blocks")
    return state


def memory_compress_node(state: RAGState) -> RAGState:
    """节点 6: Memory Compress — LLM 驱动对话历史摘要"""
    history = state.get("history_messages", [])

    if len(history) <= 3:
        state["compressed_messages"] = history
        state["memory_consolidated"] = False
        state["pipeline_stage"] = "memory_compress"
        return state

    try:
        consolidator = _get_consolidator()
        if consolidator.should_consolidate(history):
            compressed = consolidator.consolidate(history)
            state["compressed_messages"] = compressed
            state["memory_consolidated"] = True
        else:
            state["compressed_messages"] = history
            state["memory_consolidated"] = False
    except Exception as e:
        logger.warning(f"Memory compression failed: {e}")
        state["compressed_messages"] = history
        state["memory_consolidated"] = False

    state["pipeline_stage"] = "memory_compress"
    return state


def generate_node(state: RAGState) -> RAGState:
    """节点 7: LLM Generate — 结合上下文生成最终回答"""
    query = state.get("query", "")
    intent = state.get("intent", "knowledge_qa")
    formatted_ctx = state.get("formatted_context", "")
    history = state.get("compressed_messages", state.get("history_messages", []))

    # ── 闲聊模式：不需要检索 ──
    if intent == "chitchat" or not state.get("needs_retrieval", True):
        try:
            llm = _get_llm()
            messages = [SystemMessage(content="你是一个有帮助的企业知识库助手。请友好简洁地回答用户。")]
            for m in history[-6:]:
                role = "user" if m["role"] == "user" else "assistant"
                messages.append(HumanMessage(content=m["content"]) if role == "user"
                                else SystemMessage(content=m["content"]))
            messages.append(HumanMessage(content=query))
            resp = llm.invoke(messages)
            state["final_answer"] = resp.content if resp.content else "抱歉，请再试一次。"
            state["token_usage"] = _extract_token_usage(resp)
        except Exception as e:
            logger.error(f"LLM generate failed: {e}")
            state["final_answer"] = f"抱歉，生成回答时出错：{str(e)[:200]}"
            state["error"] = str(e)

        state["knowledge_refs"] = []
        state["pipeline_stage"] = "generate"
        return state

    # ── RAG 问答模式 ──
    if not formatted_ctx:
        # 没有检索到相关内容
        try:
            llm = _get_llm()
            resp = llm.invoke([
                SystemMessage(content="你是一个企业知识库助手。当前没有找到相关文档，请告知用户并建议换个问法。"),
                HumanMessage(content=query),
            ])
            state["final_answer"] = resp.content or "抱歉，未找到相关文档。请尝试用不同的关键词描述您的问题。"
        except Exception as e:
            state["final_answer"] = "抱歉，未找到相关文档，且生成回答时出错。"
            state["error"] = str(e)
        state["knowledge_refs"] = []
        state["token_usage"] = {}
        state["pipeline_stage"] = "generate"
        return state

    # 构建 RAG Prompt
    system_prompt = (
        "你是一个专业的企业知识库问答助手。请根据提供的参考文档回答用户问题。\n\n"
        "规则：\n"
        "1. 优先基于参考文档回答，如果文档中没有相关信息，请明确说明\n"
        "2. 回答要准确、简洁、有条理\n"
        "3. 引用文档内容时标注来源编号，如 [1]\n"
        "4. 如果文档信息不足以回答，请告知用户并给出建议\n"
    )

    try:
        llm = _get_llm()
        messages = [SystemMessage(content=system_prompt)]

        # 加入压缩后的历史
        for m in history[-6:]:
            content = m.get("content", "")[:500]
            if m["role"] == "user":
                messages.append(HumanMessage(content=content))
            elif m["role"] == "assistant":
                messages.append(SystemMessage(content=content))

        # 当前查询 + 上下文
        user_prompt = f"参考文档：\n\n{formatted_ctx}\n\n---\n用户问题：{query}\n\n请回答："
        messages.append(HumanMessage(content=user_prompt))

        resp = llm.invoke(messages)
        state["final_answer"] = resp.content if resp.content else "抱歉，生成回答失败。"
        state["token_usage"] = _extract_token_usage(resp)
    except Exception as e:
        logger.error(f"LLM RAG generate failed: {e}")
        state["final_answer"] = f"抱歉，生成回答时出错：{str(e)[:200]}"
        state["error"] = str(e)

    # 构建知识引用
    merged = state.get("merged_contexts", [])
    refs = []
    for ctx in merged:
        for cid in ctx.chunk_ids[:3]:
            refs.append({
                "chunk_id": cid,
                "content_preview": ctx.content[:200],
                "score": ctx.relevance_score,
            })
    state["knowledge_refs"] = refs
    state["pipeline_stage"] = "generate"
    logger.info(f"[Pipeline] Generate: answer_len={len(state['final_answer'])}, refs={len(refs)}")
    return state


def fallback_node(state: RAGState) -> RAGState:
    """回退节点：生成失败时返回兜底回答"""
    state["final_answer"] = "抱歉，我暂时无法回答这个问题。请稍后重试或尝试换个问法。"
    state["pipeline_stage"] = "fallback"
    return state


# ═══════════════════════════════════════════════════════════
# 条件路由函数
# ═══════════════════════════════════════════════════════════

def route_after_query_understand(state: RAGState) -> str:
    if state.get("needs_retrieval", False):
        return "hybrid_search"
    return "generate"


def route_after_context_merge(state: RAGState) -> str:
    history = state.get("history_messages", [])
    if len(history) > 10:
        return "memory_compress"
    return "generate"


def route_after_generate(state: RAGState) -> str:
    if state.get("error"):
        return "fallback"
    return "end"


# ═══════════════════════════════════════════════════════════
# Graph 构建
# ═══════════════════════════════════════════════════════════

def build_rag_pipeline() -> StateGraph:
    workflow = StateGraph(RAGState)

    workflow.add_node("query_understand", query_understand_node)
    workflow.add_node("hybrid_search", hybrid_search_node)
    workflow.add_node("dynamic_rrf_fusion", dynamic_rrf_fusion_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("context_merge", context_merge_node)
    workflow.add_node("memory_compress", memory_compress_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)

    workflow.set_entry_point("query_understand")

    workflow.add_conditional_edges(
        "query_understand", route_after_query_understand,
        {"hybrid_search": "hybrid_search", "generate": "generate"},
    )
    workflow.add_edge("hybrid_search", "dynamic_rrf_fusion")
    workflow.add_edge("dynamic_rrf_fusion", "rerank")
    workflow.add_edge("rerank", "context_merge")
    workflow.add_conditional_edges(
        "context_merge", route_after_context_merge,
        {"memory_compress": "memory_compress", "generate": "generate"},
    )
    workflow.add_edge("memory_compress", "generate")
    workflow.add_conditional_edges(
        "generate", route_after_generate,
        {"end": END, "fallback": "fallback"},
    )
    workflow.add_edge("fallback", END)

    return workflow.compile()


# ─── 全局单例 ───
rag_pipeline = build_rag_pipeline()


def invoke_rag_pipeline(
    query: str,
    kb_ids: Optional[List[str]] = None,
    session_id: str = "",
    history_messages: Optional[List[dict]] = None,
) -> dict:
    """
    便捷函数：直接调用 RAG Pipeline 并返回结构化结果

    Args:
        query: 用户查询
        kb_ids: 知识库 ID 列表
        session_id: 会话 ID
        history_messages: 历史消息

    Returns:
        {answer, knowledge_refs, token_usage, search_info}
    """
    initial_state = {
        "query": query,
        "session_id": session_id,
        "kb_ids": kb_ids or [],
        "history_messages": history_messages or [],
    }

    try:
        final_state = rag_pipeline.invoke(initial_state)
        return {
            "answer": final_state.get("final_answer", ""),
            "knowledge_refs": final_state.get("knowledge_refs", []),
            "token_usage": final_state.get("token_usage", {}),
            "search_info": {
                "rrf_weights": final_state.get("rrf_weights", {}),
                "vector_hits": len(final_state.get("vector_results", [])),
                "keyword_hits": len(final_state.get("keyword_results", [])),
                "fused_count": len(final_state.get("fused_results", [])),
                "reranked_count": len(final_state.get("reranked_results", [])),
            },
        }
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}", exc_info=True)
        return {
            "answer": f"抱歉，处理您的问题时出错：{str(e)[:200]}",
            "knowledge_refs": [],
            "token_usage": {},
            "search_info": {"error": str(e)},
        }
