"""
对话消息模型

存储每次对话的用户消息、助手回复、工具调用结果等。
支持 Knowledge References 追踪（回答引用了哪些 Chunk）。
"""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Text, Integer, Boolean, func, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Message(Base):
    """对话消息表

    角色 (role) 遵循 OpenAI Chat Completions API 标准:
    - user: 用户消息
    - assistant: 助手回复
    - system: 系统消息（记忆压缩摘要）
    - tool: 工具调用结果（Agent 模式）
    """

    __tablename__ = "messages"

    # ─── 主键 ───
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="消息唯一标识 (UUID v4)",
    )

    # ─── 归属 ───
    session_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="所属会话 ID",
    )

    # ─── 消息内容 ───
    role: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="角色: user / assistant / system / tool",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="消息文本内容",
    )

    # ─── 多模态支持 ───
    # 图片 URL 列表，以 JSON 数组存储
    image_urls: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="关联的图片 URL 列表",
    )

    # ─── 工具调用 (Agent 模式) ───
    tool_calls: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="工具调用记录 (OpenAI tool_calls 格式)",
    )
    tool_name: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="工具名称（tool 角色时使用）",
    )

    # ─── 知识引用 ───
    # 记录本次回答引用了哪些 Chunk
    knowledge_references: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="知识引用列表 [{chunk_id, content, score, ...}]",
    )

    # ─── Token 使用记录 ───
    token_usage: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Token 消耗: {prompt_tokens, completion_tokens, total_tokens}",
    )

    # ─── 元信息 ───
    metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="扩展元信息",
    )

    # ─── 消息序号 ───
    sequence_num: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="会话内消息序号（从 0 递增）",
    )

    # ─── 状态标记 ───
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # ─── 时间戳 ───
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="消息创建时间",
    )

    def __repr__(self) -> str:
        content_preview = self.content[:30] if self.content else ""
        return f"<Message(id={self.id}, role={self.role}, seq={self.sequence_num}, preview='{content_preview}...')>"
