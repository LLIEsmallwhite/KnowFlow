"""
User model with RBAC roles and department scoping.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Boolean, Integer, JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

# Role enum-like constants
ROLE_ADMIN = "admin"
ROLE_MANAGER = "manager"
ROLE_MEMBER = "member"

# Security levels (ascending clearance)
SEC_PUBLIC = 0
SEC_INTERNAL = 1
SEC_CONFIDENTIAL = 2
SEC_TOP_SECRET = 3


class User(Base):
    """User table with RBAC fields."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # ─── RBAC ───
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ROLE_MEMBER,
        comment="Role: admin / manager / member",
    )
    clearance_level: Mapped[int] = mapped_column(
        Integer, nullable=False, default=SEC_INTERNAL,
        comment="Maximum security level user can access: 0=public 1=internal 2=confidential 3=top_secret",
    )
    departments: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list,
        comment="List of department names user has access to, e.g. ['engineering','product']",
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN or self.is_superuser

    @property
    def can_access_all(self) -> bool:
        """Admin/superuser bypass all permission checks."""
        return self.is_admin

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"
