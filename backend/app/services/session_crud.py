"""
Session CRUD Service

Provides async PostgreSQL operations for Session model.
"""

import logging
from typing import List, Optional
from sqlalchemy import select, update, delete, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.session import Session

logger = logging.getLogger(__name__)


class SessionCRUD:
    """Async CRUD for sessions table."""

    async def create(
        self,
        db: AsyncSession,
        user_id: str,
        title: str = "New Chat",
        knowledge_base_id: Optional[str] = None,
        session_type: str = "knowledge_qa",
        enable_memory: bool = True,
        enable_web_search: bool = False,
    ) -> Session:
        session = Session(
            user_id=user_id,
            title=title,
            knowledge_base_id=knowledge_base_id,
            session_type=session_type,
            enable_memory=enable_memory,
            enable_web_search=enable_web_search,
        )
        db.add(session)
        await db.flush()
        await db.refresh(session)
        logger.info("Session created: id=%s, title=%s", session.id, title)
        return session

    async def get(self, db: AsyncSession, session_id: str) -> Optional[Session]:
        result = await db.execute(
            select(Session).where(
                Session.id == session_id,
                Session.is_active == True,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
    ) -> List[Session]:
        stmt = (
            select(Session)
            .where(
                Session.user_id == user_id,
                Session.is_active == True,
            )
            .order_by(Session.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def update_title(
        self,
        db: AsyncSession,
        session_id: str,
        title: str,
    ) -> Optional[Session]:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(title=title)
        )
        await db.flush()
        return await self.get(db, session_id)

    async def increment_message_count(
        self,
        db: AsyncSession,
        session_id: str,
        delta: int = 1,
        token_delta: int = 0,
    ) -> None:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(
                message_count=Session.message_count + delta,
                total_tokens=Session.total_tokens + token_delta,
            )
        )
        await db.flush()

    async def delete(self, db: AsyncSession, session_id: str) -> bool:
        result = await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(is_active=False)
        )
        return result.rowcount > 0


# Singleton
session_crud = SessionCRUD()
