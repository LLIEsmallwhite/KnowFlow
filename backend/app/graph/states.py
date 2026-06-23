"""
LangGraph State 定义

定义 RAG Pipeline 和 Agent ReAct Pipeline 的 State TypedDict。
LangGraph 使用 TypedDict 作为节点间传递的状态对象。

关键设计：
- RAGState: RAG 问答流水线的每一步状态
- AgentState: ReAct Agent 循环的状态
"""

from typing import TypedDict, List, Dict, Optional, Any, Annotated
import operator


# ═══════════════════════════════════════════════════════════
# RAG Pipeline State
# ═══════════════════════════════════════════════════════════

class RAGState(TypedDict, total=False):
    """
    RAG Pipeline 的全局状态

    流经 Query Understanding → Hybrid Search → RRF Fusion →
    Rerank → Context Merge → Memory Compress → LLM Generate 七个阶段。
    """

    # ── 输入 ──
    query: str                              # 用户原始查询
    session_id: str                         # 会话 ID
    kb_ids: List[str]                       # 知识库 ID 列表
    history_messages: List[dict]            # 历史对话消息
    images: Optional[List[str]]             # 关联图片 URL

    # ── 中间状态 ──
    rewritten_query: str                    # LLM 改写后的查询
    intent: str                             # 查询意图 (knowledge_qa / chitchat / agent)
    needs_retrieval: bool                   # 是否需要知识检索

    # 检索中间结果
    vector_results: List[dict]              # Milvus 向量检索原始结果
    keyword_results: List[dict]             # BM25 关键词检索原始结果
    fused_results: List[dict]               # RRF 融合后的结果
    rrf_weights: Dict[str, float]           # 动态 RRF 权重 {vector, keyword}
    reranked_results: List[dict]            # Cross-Encoder Rerank 后的结果
    merged_contexts: List[dict]             # 合并后的上下文块

    # ── 记忆相关 ──
    compressed_messages: List[dict]         # 压缩后的消息列表
    memory_consolidated: bool               # 是否执行了记忆压缩

    # ── 输出 ──
    final_answer: str                       # LLM 最终回答
    knowledge_refs: List[dict]              # 知识引用列表
    token_usage: Dict[str, int]             # Token 消耗统计

    # ── 控制 ──
    error: Optional[str]                    # 错误信息
    pipeline_stage: str                     # 当前 Pipeline 阶段（调试用）


# ═══════════════════════════════════════════════════════════
# Agent ReAct State
# ═══════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    """
    ReAct Agent 循环的状态

    Think → ToolCall → Observe → Think → ... → FinalAnswer
    """

    # ── 输入 ──
    query: str                              # 用户查询
    session_id: str                         # 会话 ID
    system_prompt: str                      # System Prompt

    # ── 对话历史 ──
    # Plain dict messages (use operator.add for simple list appending)
    messages: Annotated[List[dict], operator.add]

    # ── 当前轮 ──
    current_thought: str                    # 当前思考内容
    tool_calls: List[dict]                  # 待执行的工具调用
    tool_results: List[dict]                # 工具执行结果
    iteration: int                          # 当前迭代轮次
    max_iterations: int                     # 最大迭代轮次

    # ── 积累 ──
    knowledge_refs: List[dict]              # 积累的知识引用
    total_tokens: int                       # 累计 Token 消耗

    # ── 输出 ──
    final_answer: Optional[str]             # 最终回答
    is_complete: bool                       # Agent 是否完成

    # ── 控制 ──
    error: Optional[str]                    # 错误信息
