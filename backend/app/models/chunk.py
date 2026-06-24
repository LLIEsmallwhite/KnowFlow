"""
文本片段 (Chunk) 模型

文档经分块后，每个 Chunk 是最小的检索和索引单元。
Chunk 的向量存储在 Milvus 中，文本内容和元信息存储在 PostgreSQL。
"""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Text, Integer, Boolean, func, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Chunk(Base):
    """文本片段表

    - 每个 Chunk 对应文档中的一个连续文本段
    - 向量在 Milvus 中通过 chunk_id 关联
    - 支持 Parent-Child 分块模式（child 用于检索，parent 返回给 LLM）
    """

    __tablename__ = "chunks"

    # ─── 主键 ───
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="片段唯一标识 (UUID v4)",
    )

    # ─── 归属 ───
    document_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="所属文档 ID",
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="所属知识库 ID（冗余，加速过滤检索）",
    )

    # ─── 内容 ───
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="片段文本内容",
    )
    content_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="内容 SHA256（去重用）",
    )

    # ─── 位置信息 ───
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="在文档中的片段序号（从 0 开始）",
    )
    start_at: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="在原始文档中的起始字符位置",
    )
    end_at: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="在原始文档中的结束字符位置",
    )

    # ─── 分块类型 ───
    chunk_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="text",
        comment="片段类型: text (普通文本) / faq (问答对) / parent (父片段) / child (子片段)",
    )

    # ─── Parent-Child 关系 ───
    parent_chunk_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        index=True,
        comment="父片段 ID（子片段指向父片段）",
    )

    # ─── 元信息 ───
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
        comment="扩展元信息 (JSON): 页码/章节标题/breadcrumb 等",
    )

    # ─── 向量关联 ───
    # 向量本身存储在 Milvus 中，这里只记录向量 ID 用于关联
    vector_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Milvus 中的向量 ID（用于更新/删除）",
    )

    # ─── 安全标签 (denormalized for pre-filter) ───
    security_level: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, comment="密级: 0=公开 1=内部 2=机密 3=绝密",
    )
    department: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, comment="所属部门 (继承自 KB)",
    )

    # ─── 状态标记 ───
    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        comment="是否激活（可禁用某些片段不参与检索）",
    )
    is_indexed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="是否已完成向量索引",
    )

    # ─── 时间戳 ───
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        content_preview = self.content[:50] if self.content else ""
        return f"<Chunk(id={self.id}, doc={self.document_id}, idx={self.chunk_index}, preview='{content_preview}...')>"
