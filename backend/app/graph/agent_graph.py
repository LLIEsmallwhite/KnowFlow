"""
Agent ReAct Pipeline — LangGraph StateGraph 实现

实现 Think → ToolCall → Observe 的 ReAct 循环：

   agent_think (LLM 思考 + 选择工具)
       │
       ├── 有工具调用 → tool_execute (并行执行工具)
       │       │
       │       ├── 未超限 → agent_think (继续思考)
       │       └── 已超限 → finalize (强制结束)
       │
       └── 无工具调用 → finalize (生成最终答案)

相比 LangChain AgentExecutor 的优势：
- 完全控制每一步的细节
- 支持并行工具调用
- 支持人机审批中断 (interrupt)
- 支持迭代上限保护

设计参考:
    - WeKnora agent/engine.go (ReAct 循环)
    - LangGraph Quick Start: Agent Supervisor
"""

import logging
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from app.graph.states import AgentState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 节点实现
# ═══════════════════════════════════════════════════════════

def agent_think_node(state: AgentState) -> AgentState:
    """
    Agent 思考节点

    调用 LLM 进行推理：
    1. 分析当前状态和对话历史
    2. 决定是否需要调用工具
    3. 如果需要工具，生成 tool_calls

    这是 ReAct 循环的「Think」阶段。
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)

    logger.info(f"[Agent] Think — iteration {iteration + 1}/{max_iter}")

    # ── 检查迭代上限 ──
    if iteration >= max_iter:
        logger.warning(f"[Agent] Max iterations ({max_iter}) reached — forcing finalize")
        state["is_complete"] = True
        state["tool_calls"] = []
        return state

    # ── LLM 思考（简化版） ──
    # 实际实现：
    # 1. 构建 messages = [system_prompt, ...history, user_query, ...tool_results]
    # 2. 调用 chat_model.bind_tools(tools)
    # 3. 解析响应中的 tool_calls

    # 暂时：无工具调用，直接进入 finalize
    state["tool_calls"] = state.get("tool_calls", [])
    state["iteration"] = iteration
    state["current_thought"] = ""

    return state


def tool_execute_node(state: AgentState) -> AgentState:
    """
    工具执行节点

    执行 agent_think 决定的 tool_calls：
    - 支持并行执行多个工具调用
    - 每个工具调用的结果追加为 tool message
    - 工具执行错误不中断流程，记录在 tool message 中

    这是 ReAct 循环的「Act」阶段。
    """
    tool_calls = state.get("tool_calls", [])
    iteration = state.get("iteration", 0)

    logger.info(f"[Agent] Execute — {len(tool_calls)} tool(s) in iteration {iteration + 1}")

    # ── 执行工具（简化版） ──
    results = []
    for tc in tool_calls:
        tool_name = tc.get("function", {}).get("name", "unknown")
        try:
            # 实际调用: tool_registry.execute(tool_name, arguments)
            result = {"tool_name": tool_name, "status": "success", "result": "tool_result"}
        except Exception as e:
            result = {"tool_name": tool_name, "status": "error", "error": str(e)}
        results.append(result)

    state["tool_results"] = results
    state["iteration"] = iteration + 1

    return state


def finalize_node(state: AgentState) -> AgentState:
    """
    终结节点

    生成最终回答：
    - 汇总所有收集到的知识引用
    - 调用 LLM 生成面向用户的自然语言回答
    - 标记 Agent 执行完成

    这是 ReAct 循环的「终结」阶段。
    """
    logger.info(f"[Agent] Finalize — {state.get('iteration', 0)} iterations, {len(state.get('knowledge_refs', []))} refs")

    state["is_complete"] = True
    state["final_answer"] = state.get(
        "final_answer",
        "（Agent 最终回答待生成 — 当前为骨架阶段）",
    )

    return state


# ═══════════════════════════════════════════════════════════
# 条件路由
# ═══════════════════════════════════════════════════════════

def route_after_think(state: AgentState) -> str:
    """Agent 思考后的路由"""
    # 有工具调用 → 执行工具
    if state.get("tool_calls"):
        return "tool_execute"
    # 没有工具调用 → 生成最终答案
    return "finalize"


def route_after_execute(state: AgentState) -> str:
    """工具执行后的路由"""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)

    # 迭代上限 → 强制终结
    if iteration >= max_iter:
        logger.info(f"[Agent] Route to finalize (iteration limit {max_iter})")
        return "finalize"

    # 继续思考
    return "agent_think"


# ═══════════════════════════════════════════════════════════
# Graph 构建
# ═══════════════════════════════════════════════════════════

def build_agent_graph() -> StateGraph:
    """
    构建 ReAct Agent StateGraph

    节点拓扑:
        agent_think
            │ (有 tool_calls)
            ▼
        tool_execute
            │ (未超限)
            ▼
        agent_think  ← 循环
            │ (无 tool_calls 或 超限)
            ▼
        finalize ──► END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("agent_think", agent_think_node)
    workflow.add_node("tool_execute", tool_execute_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("agent_think")

    # Think → 条件分支
    workflow.add_conditional_edges(
        "agent_think",
        route_after_think,
        {
            "tool_execute": "tool_execute",
            "finalize": "finalize",
        },
    )

    # Execute → 条件分支（循环或终结）
    workflow.add_conditional_edges(
        "tool_execute",
        route_after_execute,
        {
            "agent_think": "agent_think",
            "finalize": "finalize",
        },
    )

    workflow.add_edge("finalize", END)

    compiled = workflow.compile()
    logger.info("Agent graph compiled successfully")
    return compiled


# ─── 全局单例 ───
agent_graph = build_agent_graph()
