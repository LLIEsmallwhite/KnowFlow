"""
LLM 记忆压缩器 (Memory Consolidator)

当多轮对话的 Token 数超过上下文窗口的 50% 时，
自动使用 LLM 对历史对话进行语义摘要压缩。

核心策略：
1. 始终保留 System Prompt（不可变）
2. 始终保留当前轮对话（最后 user + assistant + tool）
3. 对中间历史用低温度 LLM（0.3）做摘要
4. LLM 调用失败时自动退避为 rawArchive（纯文本截断）
5. 最多重试 3 次

为什么不用滑动窗口截断？
- 滑动窗口会丢失早期关键信息（如用户初始需求、中间决策）
- LLM 摘要保留了语义压缩，信息密度远高于原始对话
- 配合退避方案保证了可靠性

设计参考:
    - WeKnora agent/memory/consolidator.go
    - LangChain ConversationSummaryMemory
"""

import logging
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from app.core.config import settings
from app.memory.token_estimator import TokenEstimator

logger = logging.getLogger(__name__)


# 压缩用的 System Prompt
CONSOLIDATION_SYSTEM_PROMPT = """\
You are a conversation summarizer. Your task is to create a concise but
comprehensive summary of a conversation between a user and an AI assistant.

The summary should:
- Be written in the same language as the original conversation
- Preserve all key facts, numbers, and specific details
- Include the outcomes of any tool executions
- Note any errors or issues encountered
- Be concise — aim for 30% or less of the original length

Output only the summary, no preamble or explanation."""


class MemoryConsolidator:
    """
    LLM 记忆压缩器

    使用方式:
        consolidator = MemoryConsolidator(max_context_tokens=32000)
        if consolidator.should_consolidate(messages):
            messages = consolidator.consolidate(messages)
    """

    # 默认压缩触发阈值（上下文窗口的百分比）
    DEFAULT_THRESHOLD = 0.5

    # 压缩用 LLM 的最大输出 Token
    SUMMARY_MAX_TOKENS = 2000

    # 最大重试次数
    MAX_RETRY = 3

    def __init__(
        self,
        max_context_tokens: int = None,
        threshold: float = None,
        summary_model: str = None,
        summary_temperature: float = 0.3,
    ):
        """
        Args:
            max_context_tokens: 最大上下文 Token 数
            threshold: 触发压缩的比例（默认 0.5）
            summary_model: 压缩用的模型（默认从配置读取）
            summary_temperature: 压缩时的温度（低温度 = 保守）
        """
        self.max_context_tokens = max_context_tokens or settings.MAX_CONTEXT_TOKENS
        self.threshold = threshold or settings.MEMORY_CONSOLIDATION_THRESHOLD
        self.estimator = TokenEstimator()

        # 压缩用 LLM — 使用更便宜的模型
        self.summary_llm = ChatOpenAI(
            model=summary_model or settings.SUMMARY_LLM_MODEL,
            temperature=summary_temperature,
            max_tokens=self.SUMMARY_MAX_TOKENS,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_BASE_URL,
        )

        logger.info(
            f"MemoryConsolidator initialized: "
            f"max_tokens={self.max_context_tokens}, "
            f"threshold={self.threshold}, "
            f"model={summary_model or settings.SUMMARY_LLM_MODEL}"
        )

    # ─── 判断是否需要压缩 ───
    def should_consolidate(self, messages: List[dict]) -> bool:
        """
        判断是否需要触发压缩

        Args:
            messages: 当前消息列表

        Returns:
            True 如果 Token 数超过阈值
        """
        if self.max_context_tokens <= 0:
            return False

        token_count = self.estimator.count_messages(messages)
        trigger = int(self.max_context_tokens * self.threshold)

        if token_count > trigger:
            logger.info(
                f"Memory consolidation triggered: "
                f"{token_count} tokens > {trigger} (threshold={self.threshold})"
            )
            return True

        return False

    def should_consolidate_with_api(
        self,
        messages: List[dict],
        last_api_usage: dict,
        last_sent_count: int,
    ) -> bool:
        """
        结合 API Usage 判断是否需要压缩（更精确）
        """
        token_count = self.estimator.count_with_api_usage(
            messages, last_api_usage, last_sent_count
        )
        trigger = int(self.max_context_tokens * self.threshold)
        return token_count > trigger

    # ─── 执行压缩 ───
    def consolidate(self, messages: List[dict]) -> List[dict]:
        """
        压缩对话历史

        Args:
            messages: 当前消息列表（需包含 system + 历史 + 当前轮）

        Returns:
            压缩后的消息列表（system + 摘要 + 近期历史 + 当前轮）
        """
        if len(messages) <= 3:
            return messages  # 太少消息，不需要压缩

        # Step 1: 提取 System Prompt
        system_msg = messages[0] if messages[0].get("role") == "system" else None

        # Step 2: 定位最后一轮用户消息
        last_user_idx = len(messages) - 1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx <= 1:
            return messages  # 只有一轮对话，无需压缩

        # Step 3: 分割历史与当前轮
        history = messages[1:last_user_idx]         # 可压缩的历史
        current_turn = messages[last_user_idx:]     # 必须保留的当前轮

        if len(history) < 2:
            return messages  # 历史消息太少，无需压缩

        # Step 4: 计算保留边界
        target_tokens = int(self.max_context_tokens * self.threshold)
        keep_count = self.estimator.find_keep_boundary(
            history, system_msg or {"role": "system", "content": ""},
            current_turn, target_tokens,
        )

        if keep_count >= len(history):
            return messages  # 历史全在预算内，无需压缩

        to_consolidate = history[:len(history) - keep_count]
        to_keep = history[len(history) - keep_count:]

        # Step 5: LLM 摘要压缩
        summary = self._summarize_with_retry(to_consolidate)
        if summary is None:
            # 退避：纯文本截断
            summary = self._raw_archive(to_consolidate)
            logger.warning("Memory consolidation fell back to raw archive")

        # Step 6: 重组消息
        result = []
        if system_msg:
            result.append(system_msg)
        result.append({
            "role": "system",
            "content": (
                f"[Memory Summary — {len(to_consolidate)} "
                f"earlier messages summarized]\n\n{summary}"
            ),
        })
        result.extend(to_keep)
        result.extend(current_turn)

        logger.info(
            f"Memory consolidated: {len(to_consolidate)} messages → "
            f"summary ({len(summary)} chars), "
            f"kept {len(to_keep)} history + {len(current_turn)} current-turn"
        )
        return result

    # ─── LLM 摘要（含重试） ───
    def _summarize_with_retry(self, messages: List[dict]) -> Optional[str]:
        """LLM 摘要压缩，最多重试 3 次"""
        prompt = self._build_consolidation_prompt(messages)

        for attempt in range(1, self.MAX_RETRY + 1):
            try:
                response = self.summary_llm.invoke([
                    SystemMessage(content=CONSOLIDATION_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
                if response.content and len(response.content.strip()) > 0:
                    return response.content.strip()
                logger.warning(f"Empty summary response (attempt {attempt})")
            except Exception as e:
                logger.error(
                    f"Summary attempt {attempt}/{self.MAX_RETRY} failed: {e}"
                )
                if attempt == self.MAX_RETRY:
                    return None

        return None

    def _build_consolidation_prompt(self, messages: List[dict]) -> str:
        """构建摘要 Prompt"""
        parts = [
            "Summarize the following conversation history, preserving:\n",
            "1. Key facts and decisions made\n",
            "2. Tool execution results and their outcomes\n",
            "3. User's original intent and requirements\n",
            "4. Any errors encountered and how they were resolved\n\n",
            "Conversation to summarize:\n\n",
        ]

        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = self._truncate_for_prompt(msg.get("content", ""), 2000)

            if msg.get("tool_calls"):
                names = [tc.get("function", {}).get("name", "?")
                         for tc in msg["tool_calls"]]
                parts.append(f"**{role}** [tools: {', '.join(names)}]: {content}\n\n")
            else:
                parts.append(f"**{role}**: {content}\n\n")

        return "".join(parts)

    def _raw_archive(self, messages: List[dict]) -> str:
        """
        退避方案：纯文本截断归档

        当 LLM 摘要全部失败时使用。
        保留每条消息的角色和截断后的内容。
        """
        lines = ["Conversation archive (auto-summarization failed):\n"]
        for msg in messages:
            role = msg.get("role", "?").upper()
            content = self._truncate_for_prompt(msg.get("content", ""), 500)
            lines.append(f"- [{role}]: {content}")
        return "\n".join(lines)

    @staticmethod
    def _truncate_for_prompt(text: str, max_chars: int) -> str:
        """安全截断（按字符而非字节）"""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
