"""
FastAPI Dependency Injection Module

Provides reusable dependencies:
- get_current_user_id: JWT auth (DEV mode allows anonymous)
- get_db: database session
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings
from app.core.auth import verify_token

# ─── Auth scheme ───
security_scheme = HTTPBearer(auto_error=False)


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Optional[str]:
    """
    Extract current user ID from JWT token.

    In DEBUG mode, allows anonymous access (returns None).
    In production, requires valid JWT.
    """
    if settings.DEBUG and credentials is None:
        return None

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证 Token",
        )

    token = credentials.credentials
    user_id = verify_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效或已过期",
        )
    return user_id
