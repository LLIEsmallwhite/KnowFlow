"""
Agent ReAct Pipeline — LangGraph StateGraph 实现

Think → ToolCall → Observe 的 ReAct 循环。
每个节点调用真实的 LLM 和工具。
"""

import json
import logging
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from app.core.config import settings
from app.graph.states import AgentState

logger = logging.getLogger(__name__)

# ─── 全局 LLM ───
_agent_llm: Optional[ChatOpenAI] = None


def _get_agent_llm():
    global _agent_llm
    if _agent_llm is None:
        _agent_llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_TOKENS,
            streaming=True,
        )
    return _agent_llm


# ═══════════════════════════════════════════════════════════
# 节点实现
# ═══════════════════════════════════════════════════════════

def agent_think_node(state: AgentState) -> AgentState:
    """
    Agent 思考节点 — 调用 LLM 决定下一步

    如果 LLM 返回 tool_calls → 进入 tool_execute
    如果 LLM 直接返回文本 → 进入 finalize
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)
    query = state.get("query", "")

    logger.info(f"[Agent] Think — iteration {iteration + 1}/{max_iter}")

    if iteration >= max_iter:
        logger.warning(f"[Agent] Max iterations reached, forcing finalize")
        state["tool_calls"] = []
        state["is_complete"] = True
        return state

    # 构建消息
    system_prompt = state.get("system_prompt", (
        "你是一个智能助手，可以使用工具来完成任务。\n"
        "当需要搜索知识库时调用 knowledge_search 工具。\n"
        "如果用户的问题可以直接回答，直接给出答案，不要调用工具。\n"
    ))

    messages = [SystemMessage(content=system_prompt)]

    # 加入历史
    history = state.get("messages", [])
    for m in history[-10:]:
        role = m.get("role", "")
        content = m.get("content", "")[:1000]
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "tool":
            messages.append(ToolMessage(
                content=content,
                tool_call_id=m.get("tool_call_id", ""),
                name=m.get("name", ""),
            ))

    # 如果是第一轮，加入当前查询
    if iteration == 0:
        messages.append(HumanMessage(content=query))

    # 调用 LLM（绑定工具）
    try:
        from app.agent.tools import ALL_TOOLS
        llm = _get_agent_llm()
        llm_with_tools = llm.bind_tools(ALL_TOOLS)
        resp = llm_with_tools.invoke(messages)

        # 检查是否有工具调用
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            state["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("args", {}),
                    },
                }
                for tc in resp.tool_calls
            ]
            state["current_thought"] = resp.content or ""
            logger.info(f"[Agent] Think → {len(state['tool_calls'])} tool(s) to execute")
        else:
            state["tool_calls"] = []
            state["current_thought"] = resp.content or ""
            logger.info(f"[Agent] Think → direct answer")

    except Exception as e:
        logger.error(f"[Agent] Think error: {e}")
        state["tool_calls"] = []
        state["current_thought"] = f"思考出错: {e}"

    state["iteration"] = iteration
    return state


def tool_execute_node(state: AgentState) -> AgentState:
    """
    工具执行节点 — 并行执行所有 tool_calls
    """
    tool_calls = state.get("tool_calls", [])
    iteration = state.get("iteration", 0)

    logger.info(f"[Agent] Execute — {len(tool_calls)} tool(s)")

    from app.agent.tools import ALL_TOOLS
    tool_map = {t.name: t for t in ALL_TOOLS}

    results = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args = func.get("arguments", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        tool_fn = tool_map.get(name)
        if tool_fn:
            try:
                result = tool_fn.invoke(args)
                results.append({
                    "tool_name": name,
                    "status": "success",
                    "result": str(result),
                })
            except Exception as e:
                results.append({
                    "tool_name": name,
                    "status": "error",
                    "error": str(e),
                })
        else:
            results.append({
                "tool_name": name,
                "status": "error",
                "error": f"Unknown tool: {name}",
            })

    state["tool_results"] = results
    state["iteration"] = iteration + 1

    # 追加 tool messages 到历史
    messages = state.get("messages", [])
    for tc, tr in zip(tool_calls, results):
        messages.append({
            "role": "tool",
            "content": json.dumps(tr, ensure_ascii=False),
            "name": tr.get("tool_name", ""),
            "tool_call_id": tc.get("id", ""),
        })
    state["messages"] = messages

    logger.info(f"[Agent] Execute done: {[(r['tool_name'], r['status']) for r in results]}")
    return state


def finalize_node(state: AgentState) -> AgentState:
    """
    终结节点 — 汇总所有信息生成最终回答
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])

    logger.info(f"[Agent] Finalize — {state.get('iteration', 0)} iterations")

    # 汇总工具结果
    context_parts = []
    for r in tool_results:
        if r.get("status") == "success":
            context_parts.append(f"[{r['tool_name']}]: {r['result']}")

    context_text = "\n\n".join(context_parts) if context_parts else ""

    # 生成最终回答
    try:
        llm = _get_agent_llm()
        messages = [
            SystemMessage(content="你是一个智能助手。根据工具执行结果回答用户问题。回答要准确、清晰。"),
        ]

        if context_text:
            messages.append(HumanMessage(
                content=f"工具执行结果：\n\n{context_text}\n\n---\n用户原始问题：{query}\n\n请基于以上信息回答："
            ))
        else:
            messages.append(HumanMessage(content=query))

        resp = llm.invoke(messages)
        state["final_answer"] = resp.content if resp.content else "抱歉，无法生成回答。"
    except Exception as e:
        logger.error(f"[Agent] Finalize error: {e}")
        state["final_answer"] = f"抱歉，生成回答时出错：{str(e)[:200]}"

    state["is_complete"] = True
    logger.info(f"[Agent] Complete: answer_len={len(state['final_answer'])}")
    return state


# ═══════════════════════════════════════════════════════════
# 条件路由
# ═══════════════════════════════════════════════════════════

def route_after_think(state: AgentState) -> str:
    if state.get("tool_calls"):
        return "tool_execute"
    return "finalize"


def route_after_execute(state: AgentState) -> str:
    if state.get("iteration", 0) >= state.get("max_iterations", 10):
        return "finalize"
    return "agent_think"


# ═══════════════════════════════════════════════════════════
# Graph 构建
# ═══════════════════════════════════════════════════════════

def build_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("agent_think", agent_think_node)
    workflow.add_node("tool_execute", tool_execute_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("agent_think")

    workflow.add_conditional_edges(
        "agent_think", route_after_think,
        {"tool_execute": "tool_execute", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "tool_execute", route_after_execute,
        {"agent_think": "agent_think", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()


agent_graph = build_agent_graph()
