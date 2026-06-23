"""
数据库会话管理模块

使用 SQLAlchemy 2.0 异步引擎连接 PostgreSQL。
提供 FastAPI 依赖注入式的数据库会话管理。
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# ─── 异步引擎 ───
# echo=True 在 DEBUG 模式下打印 SQL
engine = create_async_engine(
    settings.DATABASE_URL_ASYNC,
    echo=settings.DEBUG,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,  # 连接前检查可用性
)

# ─── 会话工厂 ───
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # 提交后不使对象过期，避免异步上下文中的懒加载问题
)


# ─── 声明式基类 ───
class Base(DeclarativeBase):
    """所有 SQLAlchemy 模型的基类"""
    pass


# ─── FastAPI 依赖注入 ───
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI 依赖：提供数据库会话

    用法:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    初始化数据库表结构（开发用）
    生产环境建议使用 Alembic 迁移。
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """关闭数据库连接"""
    await engine.dispose()
