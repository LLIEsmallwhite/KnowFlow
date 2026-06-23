"""
对话会话模型

每个 Session 代表一次完整的多轮对话。
一个会话关联一个知识库，包含多条 Message。
"""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Integer, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Session(Base):
    """对话会话表

    会话支持两种模式：
    - knowledge_qa: 基于知识库的 RAG 问答
    - agent: 基于 ReAct Agent 的多步推理对话
    """

    __tablename__ = "sessions"

    # ─── 主键 ───
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="会话唯一标识 (UUID v4)",
    )

    # ─── 归属 ───
    user_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="所属用户 ID",
    )
    knowledge_base_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        index=True,
        comment="关联的知识库 ID（可跨 KB 检索时为空）",
    )

    # ─── 基本信息 ───
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="New Chat",
        comment="会话标题（自动生成或用户设置）",
    )
    session_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="knowledge_qa",
        comment="会话类型: knowledge_qa / agent",
    )

    # ─── 模式标记 ───
    is_pinned: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="是否置顶",
    )
    enable_memory: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        comment="是否启用记忆压缩",
    )
    enable_web_search: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="是否启用联网搜索",
    )

    # ─── 统计 ───
    message_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="消息总数（冗余计数）",
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="累计 Token 消耗",
    )

    # ─── 状态 ───
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # ─── 时间戳 ───
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, title={self.title}, type={self.session_type})>"
