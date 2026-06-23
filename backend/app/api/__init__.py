"""
KnowFlow API 路由模块

统一注册所有 API 路由到 FastAPI 应用。
新增 API 时在此处添加 router 即可。
"""

from fastapi import APIRouter
from app.api.chat import router as chat_router
from app.api.knowledge_base import router as kb_router
from app.api.conversation import router as conv_router
from app.api.agent import router as agent_router
from app.api.auth import router as auth_router

# Create main router
api_router = APIRouter()

# Register sub-routers
api_router.include_router(chat_router)
api_router.include_router(kb_router)
api_router.include_router(conv_router)
api_router.include_router(agent_router)
api_router.include_router(auth_router)

__all__ = ["api_router"]
