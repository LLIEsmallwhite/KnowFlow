"""
Langfuse observability integration.

Provides full tracing for LLM calls, retrieval, and reranking.
Enable via LANGFUSE_ENABLED=true + LANGFUSE_PUBLIC_KEY/SECRET_KEY.
"""

import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from app.core.config import settings

logger = logging.getLogger(__name__)


class LangfuseManager:
    """Singleton Langfuse observability manager."""

    _instance: Optional["LangfuseManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._client = None
        return cls._instance

    @property
    def enabled(self) -> bool:
        return settings.LANGFUSE_ENABLED and bool(
            settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY
        )

    @property
    def client(self):
        if not self.enabled:
            return None
        if self._client is None and not self._initialized:
            self._init_client()
        return self._client

    def _init_client(self):
        self._initialized = True
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
                release=settings.APP_VERSION,
            )
            logger.info("Langfuse initialized: host=%s", settings.LANGFUSE_HOST)
        except ImportError:
            logger.warning("langfuse not installed. Run: pip install langfuse")
            self._client = None
        except Exception as e:
            logger.error("Failed to init Langfuse: %s", e)
            self._client = None

    def get_langchain_callback(self):
        """Get LangChain CallbackHandler (v4 API)."""
        if not self.enabled or not self.client:
            return None
        try:
            from langfuse.langchain import CallbackHandler
            return CallbackHandler()
        except ImportError:
            try:
                from langfuse.callback import CallbackHandler
                return CallbackHandler()
            except ImportError:
                logger.warning("langfuse callback not available")
                return None

    @contextmanager
    def trace(self, name: str, session_id: str = None, metadata: dict = None,
              input_data: dict = None):
        """Create a trace span. Use as context manager."""
        if not self.enabled or not self.client:
            yield _NoOpSpan()
            return

        span = None
        start = time.time()
        try:
            span = self.client.trace(
                name=name,
                session_id=session_id,
                metadata=metadata or {},
                input=input_data,
            )
            yield _LangfuseSpan(span, start)
        except Exception as e:
            logger.debug("Langfuse trace error: %s", e)
            yield _NoOpSpan()
        finally:
            try:
                if span:
                    span.end()
                if self.client:
                    self.client.flush()
            except Exception:
                pass


class _NoOpSpan:
    def log(self, **kw): pass
    def finish(self, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class _LangfuseSpan:
    def __init__(self, trace, start_time: float = None):
        self._trace = trace
        self._start = start_time or time.time()

    def log(self, **data):
        """Log metadata to the trace."""
        if self._trace:
            try:
                self._trace.update(metadata=data)
            except Exception:
                pass

    def finish(self, output=None, metadata=None):
        """Finish the trace with output."""
        if self._trace:
            try:
                self._trace.update(
                    output=output,
                    metadata=metadata or {},
                )
            except Exception:
                pass

    def __enter__(self): return self
    def __exit__(self, *a): pass


# Global singleton
langfuse_manager = LangfuseManager()
