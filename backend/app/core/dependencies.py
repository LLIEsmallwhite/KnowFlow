"""
FastAPI 依赖注入模块

提供常用的可注入依赖，如：
- 当前用户识别
- 数据库会话
- LLM 实例获取
- Embedding 模型实例获取
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

# ─── 认证方案 ───
# 使用 HTTP Bearer Token（JWT）
security_scheme = HTTPBearer(auto_error=False)


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Optional[str]:
    """
    从 JWT Token 中提取当前用户 ID

    开发阶段允许无认证访问（返回 None）。
    生产环境应强制验证 JWT Token。
    """
    if credentials is None:
        # 开发模式：允许匿名访问
        return None

    # TODO: 实现 JWT 验证逻辑
    # token = credentials.credentials
    # payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    # return payload.get("sub")

    return None
