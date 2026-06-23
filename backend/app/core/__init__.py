"""
Core module for KnowFlow.

Provides:
- settings: Global Pydantic Settings instance
- get_db: FastAPI dependency for async DB sessions
- Base: SQLAlchemy declarative base class
"""

from app.core.config import settings
from app.core.database import Base, get_db, init_db, close_db, engine

__all__ = [
    "settings",
    "Base",
    "get_db",
    "init_db",
    "close_db",
    "engine",
]
