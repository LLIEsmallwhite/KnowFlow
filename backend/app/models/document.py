"""
文档模型

存储上传到知识库的文档元信息。
实际文件存储在 MinIO 对象存储中。
文档解析和向量化通过 Celery 异步任务完成。
"""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Text, Integer, Float, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Document(Base):
    """文档表

    记录每个上传文档的元信息。
    解析状态流转: pending → processing → completed / failed
    """

    __tablename__ = "documents"

    # ─── 主键 ───
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="文档唯一标识 (UUID v4)",
    )

    # ─── 归属 ───
    knowledge_base_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="所属知识库 ID",
    )

    # ─── 文件信息 ───
    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="文档标题（通常自动提取或使用文件名）",
    )
    file_name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="原始文件名",
    )
    file_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="文件类型: pdf / docx / md / txt / html / csv / xlsx / pptx / image",
    )
    file_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        comment="MinIO 中的存储路径 (bucket/key)",
    )
    file_size: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="文件大小（字节）",
    )
    file_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="文件 SHA256 哈希（用于去重检测）",
    )

    # ─── 处理状态 ───
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        comment="处理状态: pending / processing / completed / failed",
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="失败时的错误信息",
    )
    progress: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        comment="处理进度 (0.0 - 1.0)",
    )

    # ─── 分块统计 ───
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="解析后的片段数量",
    )
    token_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="总 Token 估算数",
    )

    # ─── 来源信息 ───
    source: Mapped[Optional[str]] = mapped_column(
        String(64),
        default="upload",
        comment="来源: upload / feishu / notion / web",
    )
    source_url: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
        comment="来源 URL（如网页抓取）",
    )

    # ─── 安全标签 ───
    security_level: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, comment="继承自 KB 的密级",
    )

    # ─── 状态标记 ───
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_indexed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="是否已完成 Milvus 索引",
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
        return f"<Document(id={self.id}, title={self.title}, status={self.status})>"
