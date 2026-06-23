"""
KnowFlow API 路由模块

统一注册所有 API 路由到 FastAPI 应用。
新增 API 时在此处添加 router 即可。
"""

from fastapi import APIRouter
from app.api.chat import router as chat_router
from app.api.knowledge_base import router as kb_router

# 创建主路由
api_router = APIRouter()

# 注册子路由
api_router.include_router(chat_router)
api_router.include_router(kb_router)

__all__ = ["api_router"]
