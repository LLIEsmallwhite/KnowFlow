"""
KnowFlow — 企业级 RAG 知识库问答助手
FastAPI 应用入口

启动方式:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting...")
    print(f"   Debug: {settings.DEBUG}")
    print(f"   LLM: {settings.LLM_PROVIDER}/{settings.LLM_MODEL}")
    print(f"   Embedding: {settings.EMBEDDING_PROVIDER}/{settings.EMBEDDING_MODEL}")
    print(f"   Vector DB: Milvus @ {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    print(f"   Langfuse: {'Enabled' if settings.LANGFUSE_ENABLED else 'Disabled'}")
    yield
    # 关闭时执行
    print(f"👋 {settings.APP_NAME} shutting down...")


# ─── 创建 FastAPI 应用 ───
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="企业级 RAG 知识库问答助手 — LangChain + LangGraph + Milvus + Streamlit",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS 中间件 ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 注册路由 ───
app.include_router(api_router)

# ─── 健康检查 ───
@app.get("/health", tags=["System"])
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.get("/", tags=["System"])
async def root():
    """根路径"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


# ─── 开发模式直接运行 ───
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
