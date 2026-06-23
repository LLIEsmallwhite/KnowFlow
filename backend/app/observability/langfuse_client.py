"""
Langfuse 可观测性集成

提供 LLM 调用、检索、Rerank 的全链路追踪。

集成方式：
- LangChain Callback: 自动追踪所有 ChatOpenAI 调用
- 自定义 Span: 手动追踪检索/Rerank 等非 LLM 操作
- Token 追踪: 每次 LLM 调用的 Token 消耗自动上报

环境变量配置：
- LANGFUSE_ENABLED=true 启用
- LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 认证
- LANGFUSE_HOST=https://cloud.langfuse.com (自建则改为你的地址)

设计参考:
    - WeKnora tracing/langfuse 集成
    - Langfuse Python SDK 文档
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from app.core.config import settings

logger = logging.getLogger(__name__)


class LangfuseManager:
    """
    Langfuse 可观测性管理器（单例模式）

    提供 Langfuse 客户端的懒加载初始化。
    当 LANGFUSE_ENABLED=false 时，所有操作都是空操作（零性能开销）。

    使用方式:
        manager = LangfuseManager()
        with manager.trace("retrieval", session_id="...") as span:
            span.log({"vector_hits": 42})
    """

    _instance: Optional["LangfuseManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._client = None
        return cls._instance

    @property
    def enabled(self) -> bool:
        """Langfuse 是否可用"""
        return settings.LANGFUSE_ENABLED and bool(
            settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY
        )

    @property
    def client(self):
        """懒加载 Langfuse 客户端"""
        if not self.enabled:
            return None

        if self._client is None and not self._initialized:
            self._init_client()

        return self._client

    def _init_client(self):
        """初始化 Langfuse 客户端"""
        self._initialized = True
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
                release=settings.APP_VERSION,
            )
            logger.info(f"Langfuse initialized: host={settings.LANGFUSE_HOST}")
        except ImportError:
            logger.warning("langfuse package not installed. Run: pip install langfuse")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self._client = None

    def get_langchain_callback(self):
        """
        获取 LangChain Callback Handler

        将此 callback 传入 ChatOpenAI(callbacks=[callback])
        即可自动追踪所有 LLM 调用。

        Returns:
            CallbackHandler 或 None
        """
        if not self.enabled or not self.client:
            return None

        try:
            from langfuse.callback import CallbackHandler
            return CallbackHandler()
        except ImportError:
            return None

    @contextmanager
    def trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ):
        """
        创建自定义 Trace（上下文管理器）

        用法:
            with langfuse_mgr.trace("hybrid_search", session_id="sess_123") as span:
                span.log({"vector_hits": 10, "keyword_hits": 5})

        Args:
            name: Trace 名称（显示在 Langfuse UI 中）
            session_id: 会话 ID
            metadata: 元数据
            input_data: 输入数据
        """
        if not self.enabled or not self.client:
            # 空操作 context manager
            yield _NoOpSpan()
            return

        trace = None
        try:
            trace = self.client.trace(
                name=name,
                session_id=session_id,
                metadata=metadata or {},
                input=input_data,
            )
            yield _LangfuseSpan(trace)
        except Exception as e:
            logger.debug(f"Langfuse trace error (non-fatal): {e}")
            yield _NoOpSpan()
        finally:
            if trace:
                try:
                    self.client.flush()
                except Exception:
                    pass


class _NoOpSpan:
    """空 Span — 当 Langfuse 未启用时使用"""

    def log(self, data: Dict[str, Any]):
        pass

    def finish(self, output=None, metadata=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _LangfuseSpan:
    """Langfuse Trace 包装器"""

    def __init__(self, trace):
        self.trace = trace
        self._span = None

    def log(self, data: Dict[str, Any]):
        """记录数据到当前 Span"""
        if self._span:
            pass  # TODO: 使用 span.update()

    def span(self, name: str, input_data=None) -> "_LangfuseSpan":
        """创建子 Span"""
        if self.trace:
            child = self.trace.span(name=name, input=input_data)
            return _LangfuseSpan(child)
        return self

    def finish(self, output=None, metadata=None):
        """结束 Span"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ─── 全局单例 ───
langfuse_manager = LangfuseManager()
