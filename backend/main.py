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
from app.core.database import init_db, close_db
from app.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting...")
    print(f"   Debug: {settings.DEBUG}")
    print(f"   LLM: {settings.LLM_PROVIDER}/{settings.LLM_MODEL}")
    print(f"   Embedding: {settings.EMBEDDING_PROVIDER}/{settings.EMBEDDING_MODEL}")
    print(f"   Vector DB: Milvus @ {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    print(f"   Langfuse: {'Enabled' if settings.LANGFUSE_ENABLED else 'Disabled'}")

    # Initialize database tables
    try:
        await init_db()
        print("   PostgreSQL: connected & tables created")
    except Exception as e:
        print(f"   PostgreSQL: unavailable ({e}) — running without DB persistence")

    # Verify Redis
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        r.ping()
        r.close()
        print("   Redis: connected")
    except Exception as e:
        print(f"   Redis: unavailable ({e})")

    # Verify Milvus
    try:
        from pymilvus import MilvusClient as MC
        mc = MC(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}", timeout=3)
        collections = mc.list_collections()
        mc.close()
        print(f"   Milvus: connected ({len(collections)} collections)")
    except Exception as e:
        print(f"   Milvus: unavailable ({e})")

    # Auto-rebuild BM25 from DB chunks
    try:
        from app.retrieval import shared_bm25
        from app.services.kb_crud import kb_crud
        from app.services.chunk_crud import chunk_crud
        from app.retrieval.bm25_retriever import build_bm25_index_from_db
        from app.core.database import async_session_factory
        async with async_session_factory() as db:
            kbs = await kb_crud.list(db)
            total = 0
            for kb in kbs:
                indexable = await chunk_crud.get_indexable(db, kb.id)
                if indexable:
                    build_bm25_index_from_db(kb.id, indexable, shared_bm25)
                    total += len(indexable)
            print(f"   BM25: rebuilt from DB ({total} chunks across {len(kbs)} KBs)")
    except Exception as e:
        print(f"   BM25: rebuild failed ({e})")

    yield
    # Shutdown
    await close_db()
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
