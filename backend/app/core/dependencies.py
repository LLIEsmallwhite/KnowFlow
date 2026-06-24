"""
FastAPI Dependencies — auth + permission injection.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.auth import verify_token
from app.core.database import get_db
from app.models.user import User
from app.core.permissions import build_permission_filter, PermissionFilter

security_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Return full User object from JWT. In DEBUG mode, returns a default dev user."""
    # DEBUG mode: auto-create/return a dev admin user
    if settings.DEBUG:
        result = await db.execute(select(User).where(User.username == "dev").limit(1))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(
                username="dev", email="dev@knowflow.local",
                hashed_password="", role="admin", clearance_level=999,
                departments=["engineering", "product", "hr"],
                is_superuser=True,
            )
            db.add(user)
            await db.flush()
            await db.refresh(user)
        return user

    # Production: enforce JWT
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未提供认证 Token")
    user_id = verify_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 无效或已过期")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")
    return user


async def get_current_user_id(
    user: User = Depends(get_current_user),
) -> str:
    """Convenience: return user.id string."""
    return user.id


async def get_permission_filter(
    user: User = Depends(get_current_user),
) -> PermissionFilter:
    """Build permission filter for current user."""
    return build_permission_filter(user)
