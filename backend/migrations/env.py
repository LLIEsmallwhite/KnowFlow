"""
Alembic 迁移环境配置

使用 SQLAlchemy async engine 进行迁移。
自动从 app.models 导入所有模型以检测变更。
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Alembic Config 对象
config = context.config

# 日志配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ─── 导入所有模型（确保 Alembic 检测到） ───
from app.core.database import Base
from app.models import (
    User,
    KnowledgeBase,
    Document,
    Chunk,
    Session,
    Message,
)

# 目标元数据
target_metadata = Base.metadata

# SQLAlchemy URL（从应用配置读取，而非 alembic.ini）
from app.core.config import settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


def run_migrations_offline() -> None:
    """
    离线模式迁移
    生成 SQL 脚本而非直接执行。
    用法: alembic upgrade head --sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在给定连接上执行迁移"""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """异步模式迁移"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """在线模式迁移（直接执行 SQL）"""
    asyncio.run(run_async_migrations())


# ─── 入口 ───
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
