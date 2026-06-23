"""
知识库模型

知识库是文档的逻辑分组单元，每个知识库拥有独立的：
- 索引配置（Embedding 模型、分块策略）
- 检索配置（向量/关键词权重偏好）
- 关联的 Milvus Collection 分区
"""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Text, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class KnowledgeBase(Base):
    """知识库表

    每个知识库对应 Milvus 中的一个 Partition。
    支持三种类型：document（文档型）、faq（问答型）、wiki（Wiki 型）。
    """

    __tablename__ = "knowledge_bases"

    # ─── 主键 ───
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="知识库唯一标识 (UUID v4)",
    )

    # ─── 基本信息 ───
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="知识库名称",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="知识库描述",
    )
    kb_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="document",
        comment="知识库类型: document / faq / wiki",
    )

    # ─── 模型配置 ───
    embedding_model_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="Embedding 模型标识",
    )
    llm_model_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="默认对话模型标识",
    )

    # ─── 索引配置 (JSON) ───
    # 存储分块策略、chunk_size、overlap 等配置
    chunking_config: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="分块配置 (JSON 字符串)",
    )

    # ─── 检索配置 (JSON) ───
    # 存储 RRF 权重偏好、检索阈值等
    retrieval_config: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="检索配置 (JSON 字符串)",
    )

    # ─── 向量索引配置 ───
    is_vector_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        comment="是否启用向量检索",
    )
    is_keyword_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        comment="是否启用关键词检索",
    )

    # ─── 归属 ───
    created_by: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        comment="创建者用户 ID",
    )

    # ─── 状态 ───
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    # ─── 统计计数 ───
    document_count: Mapped[int] = mapped_column(
        default=0,
        comment="文档数量（冗余计数，加速查询）",
    )
    chunk_count: Mapped[int] = mapped_column(
        default=0,
        comment="片段数量（冗余计数，加速查询）",
    )

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
        return f"<KnowledgeBase(id={self.id}, name={self.name}, type={self.kb_type})>"
