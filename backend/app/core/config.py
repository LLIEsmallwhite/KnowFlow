"""
KnowFlow 全局配置模块

使用 Pydantic Settings 从环境变量 / .env 文件加载配置。
所有配置项均提供合理的默认值，便于本地开发。
"""

from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """KnowFlow 全局配置

    配置优先级: 环境变量 > .env 文件 > 默认值
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # ─── 应用基础 ───
    APP_NAME: str = "KnowFlow"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me-in-production"
    LOG_LEVEL: str = "INFO"

    # ─── PostgreSQL ───
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "knowflow"
    DB_PASSWORD: str = "knowflow_secret"
    DB_NAME: str = "knowflow"

    @property
    def DATABASE_URL(self) -> str:
        """同步数据库连接字符串 (Alembic 使用)"""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def DATABASE_URL_ASYNC(self) -> str:
        """异步数据库连接字符串 (SQLAlchemy async 使用)"""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # ─── Redis ───
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        """Redis 连接字符串"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ─── Milvus ───
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "knowflow_knowledge"

    # ─── MinIO 对象存储 ───
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "knowflow-documents"
    MINIO_SECURE: bool = False

    # ─── LLM 配置 ───
    LLM_PROVIDER: Literal["openai", "azure", "ollama", "deepseek"] = "openai"
    LLM_MODEL: str = "gpt-4o"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096

    # 廉价模型 — 用于记忆压缩等非核心任务
    SUMMARY_LLM_MODEL: str = "gpt-4o-mini"
    SUMMARY_LLM_TEMPERATURE: float = 0.3

    # ─── Embedding 模型 ───
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_DIMENSION: int = 1024

    # ─── Rerank 模型 ───
    RERANK_PROVIDER: Literal["local", "cohere"] = "local"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_API_KEY: str = ""
    RERANK_BASE_URL: str = ""

    # ─── Langfuse 可观测性 ───
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ─── 检索配置 ───
    RRF_K: int = 60
    RRF_VECTOR_WEIGHT_BASE: float = 0.7
    RRF_KEYWORD_WEIGHT_BASE: float = 0.3
    VECTOR_SEARCH_TOP_K: int = 50
    KEYWORD_SEARCH_TOP_K: int = 50
    VECTOR_THRESHOLD: float = 0.15
    KEYWORD_THRESHOLD: float = 0.30
    RERANK_TOP_K: int = 10
    RERANK_THRESHOLD: float = 0.2

    # ─── 记忆压缩 ───
    MAX_CONTEXT_TOKENS: int = 32000
    MEMORY_CONSOLIDATION_THRESHOLD: float = 0.5

    # ─── Agent 配置 ───
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_PARALLEL_TOOL_CALLS: bool = True
    AGENT_LLM_TIMEOUT: int = 120

    # ─── Web Search ───
    WEB_SEARCH_ENABLED: bool = False
    WEB_SEARCH_PROVIDER: str = "duckduckgo"

    # ─── Celery ───
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"


# 全局配置单例
settings = Settings()
