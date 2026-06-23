"""
用户模型

提供基础的认证和用户信息存储。
密码使用 bcrypt 哈希存储。
"""

import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class User(Base):
    """用户表

    存储用户认证信息和基本资料。
    未来可扩展 RBAC 角色系统。
    """

    __tablename__ = "users"

    # ─── 主键 ───
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="用户唯一标识 (UUID v4)",
    )

    # ─── 基本信息 ───
    username: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="用户名（登录用）",
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="邮箱地址",
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="bcrypt 哈希密码",
    )

    # ─── 状态 ───
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        comment="账户是否激活",
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="是否为超级管理员",
    )

    # ─── 时间戳 ───
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="注册时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="最后更新时间",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username})>"
