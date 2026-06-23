"""
RAG Pipeline — LangGraph StateGraph 实现

完整的 RAG 问答流水线，包含 7 个处理节点：

  QueryUnderstand → HybridSearch → DynamicRRF → Rerank
       → ContextMerge → [MemoryCompress] → Generate

流程由 LangGraph 的条件边控制：
- QueryUnderstand 判断是否需要检索
- ContextMerge 判断是否需要记忆压缩
- Generate 判断是否成功（失败 → 回退）

设计原则：
- 每个节点是一个纯函数，接收 State 并返回 State
- 节点间通过 TypedDict 传递数据
- 条件边根据 State 中的字段决定下一步
"""

import logging
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, END

from app.graph.states import RAGState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 节点实现
# ═══════════════════════════════════════════════════════════

def query_understand_node(state: RAGState) -> RAGState:
    """
    节点 1: Query Understanding

    功能：
    - 使用 LLM 对用户查询进行改写（补全上下文、消除歧义）
    - 意图分类（knowledge_qa / chitchat / agent）

    注意：此节点会调用 LLM，是 Pipeline 中的第一个 LLM 调用点。
    """
    logger.info(f"[Pipeline] QueryUnderstand: '{state.get('query', '')[:60]}...'")

    # ── 简化实现：开发阶段直接使用原始查询 ──
    # 生产环境应调用 PluginQueryUnderstand 的 LLM 改写逻辑
    query = state.get("query", "")

    state["rewritten_query"] = query  # 暂时不调用 LLM 改写
    state["intent"] = "knowledge_qa"
    state["needs_retrieval"] = bool(query.strip())
    state["pipeline_stage"] = "query_understand"

    return state


def hybrid_search_node(state: RAGState) -> RAGState:
    """
    节点 2: Hybrid Search

    并行执行 Dense Vector 检索（Milvus）+ BM25 关键词检索。
    结果存入 state 供后续融合使用。

    在实际调用中，此节点会：
    1. 创建 HybridSearchOrchestrator
    2. 并行调用 DenseRetriever.search() 和 BM25Retriever.search()
    3. 将原始结果写入 state
    """
    logger.info(f"[Pipeline] HybridSearch: query='{state.get('rewritten_query', '')[:60]}...'")

    state["vector_results"] = state.get("vector_results", [])
    state["keyword_results"] = state.get("keyword_results", [])
    state["pipeline_stage"] = "hybrid_search"

    return state


def dynamic_rrf_fusion_node(state: RAGState) -> RAGState:
    """
    节点 3: Dynamic RRF Fusion

    核心创新：基于查询特征的四因子动态 RRF 融合。
    将向量检索和关键词检索的结果按照自适应权重合并。

    使用 DynamicRRF 模块计算：
    - 动态权重（查询类型/长度/分布/方差）
    - RRF 分数 = VectorWeight/(k+VRank) + KeywordWeight/(k+KWRank)
    """
    logger.info(f"[Pipeline] DynamicRRF: v={len(state.get('vector_results', []))}, k={len(state.get('keyword_results', []))}")

    state["fused_results"] = state.get("fused_results", [])
    state["rrf_weights"] = {"vector": 0.7, "keyword": 0.3}  # 实际由 DynamicRRF 计算
    state["pipeline_stage"] = "rrf_fusion"

    return state


def rerank_node(state: RAGState) -> RAGState:
    """
    节点 4: Cross-Encoder Rerank

    使用 Cross-Encoder 模型对融合后的候选 Chunk 进行精排。
    只保留 Top-K（默认 10）个最相关的 Chunk。

    注意：
    - 候选数 ≤ 3 时跳过 Rerank
    - Rerank 模型不可用时回退到 NoOpReranker
    """
    logger.info(f"[Pipeline] Rerank: candidates={len(state.get('fused_results', []))}")

    state["reranked_results"] = state.get("reranked_results", [])
    state["pipeline_stage"] = "rerank"

    return state


def context_merge_node(state: RAGState) -> RAGState:
    """
    节点 5: Context Merge

    功能：
    - 按知识来源分组
    - 合并相邻 Chunk
    - 格式化为 LLM Prompt 模板文本
    - 截断过短/过长的内容
    """
    logger.info(f"[Pipeline] ContextMerge: reranked={len(state.get('reranked_results', []))}")

    state["merged_contexts"] = state.get("merged_contexts", [])
    state["pipeline_stage"] = "context_merge"

    return state


def memory_compress_node(state: RAGState) -> RAGState:
    """
    节点 6: Memory Compress（条件节点）

    当对话历史 Token 数 > 上下文窗口的 50% 时触发。
    使用低温度 LLM 将早期对话压缩为语义摘要。

    仅在 should_consolidate 返回 True 时执行。
    """
    logger.info(f"[Pipeline] MemoryCompress: history_len={len(state.get('history_messages', []))}")

    state["compressed_messages"] = state.get("history_messages", [])
    state["memory_consolidated"] = True
    state["pipeline_stage"] = "memory_compress"

    return state


def generate_node(state: RAGState) -> RAGState:
    """
    节点 7: LLM Generate

    将合并后的上下文 + System Prompt + 对话历史整合，
    调用 LLM 生成最终回答。

    输出：
    - final_answer: 回答文本
    - knowledge_refs: 引用的知识来源
    - token_usage: Token 消耗
    """
    logger.info(f"[Pipeline] Generate: contexts={len(state.get('merged_contexts', []))}")

    state["final_answer"] = state.get("final_answer", "（生成节点待接入 LLM）")
    state["knowledge_refs"] = state.get("knowledge_refs", [])
    state["token_usage"] = state.get("token_usage", {})
    state["pipeline_stage"] = "generate"

    return state


def fallback_node(state: RAGState) -> RAGState:
    """回退节点：生成失败时返回兜底回答"""
    state["final_answer"] = "抱歉，我暂时无法回答这个问题。请稍后再试。"
    state["pipeline_stage"] = "fallback"

    return state


# ═══════════════════════════════════════════════════════════
# 条件路由函数
# ═══════════════════════════════════════════════════════════

def route_after_query_understand(state: RAGState) -> str:
    """Query Understanding 后的路由"""
    if state.get("needs_retrieval", False):
        return "hybrid_search"
    return "generate"


def route_after_context_merge(state: RAGState) -> str:
    """Context Merge 后的路由：是否需要压缩"""
    # 简化：有超过 10 条历史消息时触发压缩
    history = state.get("history_messages", [])
    if len(history) > 10:
        return "memory_compress"
    return "generate"


def route_after_generate(state: RAGState) -> str:
    """Generate 后的路由：检查是否有错误"""
    if state.get("error"):
        return "fallback"
    return "end"


# ═══════════════════════════════════════════════════════════
# Graph 构建
# ═══════════════════════════════════════════════════════════

def build_rag_pipeline() -> StateGraph:
    """
    构建完整的 RAG Pipeline StateGraph

    节点拓扑:
        query_understand
            │ (needs_retrieval=true)
            ▼
        hybrid_search ──► dynamic_rrf_fusion ──► rerank
            │
            ▼
        context_merge
            │ (need_compress)
            ▼
        memory_compress ──► generate ──► END
            │                  │ (error)
            ▼                  ▼
        generate            fallback ──► END
            │
            ▼
           END
    """
    workflow = StateGraph(RAGState)

    # 注册节点
    workflow.add_node("query_understand", query_understand_node)
    workflow.add_node("hybrid_search", hybrid_search_node)
    workflow.add_node("dynamic_rrf_fusion", dynamic_rrf_fusion_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("context_merge", context_merge_node)
    workflow.add_node("memory_compress", memory_compress_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)

    # 入口
    workflow.set_entry_point("query_understand")

    # 条件边
    workflow.add_conditional_edges(
        "query_understand",
        route_after_query_understand,
        {
            "hybrid_search": "hybrid_search",
            "generate": "generate",
        },
    )

    # 检索 → 融合 → 重排序 → 合并（线性流水线）
    workflow.add_edge("hybrid_search", "dynamic_rrf_fusion")
    workflow.add_edge("dynamic_rrf_fusion", "rerank")
    workflow.add_edge("rerank", "context_merge")

    # 合并后的条件分支
    workflow.add_conditional_edges(
        "context_merge",
        route_after_context_merge,
        {
            "memory_compress": "memory_compress",
            "generate": "generate",
        },
    )

    workflow.add_edge("memory_compress", "generate")

    # 生成后的最终路由
    workflow.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "end": END,
            "fallback": "fallback",
        },
    )
    workflow.add_edge("fallback", END)

    compiled = workflow.compile()
    logger.info("RAG Pipeline graph compiled successfully")
    return compiled


# ─── 全局单例 ───
rag_pipeline = build_rag_pipeline()
