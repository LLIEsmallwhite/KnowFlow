"""
Message CRUD Service

Provides async PostgreSQL operations for Message model.
"""

import logging
from typing import List, Optional, Dict
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message

logger = logging.getLogger(__name__)


class MessageCRUD:
    """Async CRUD for messages table."""

    async def create(
        self,
        db: AsyncSession,
        session_id: str,
        role: str,
        content: str,
        sequence_num: Optional[int] = None,
        knowledge_references: Optional[List[Dict]] = None,
        token_usage: Optional[Dict] = None,
        tool_calls: Optional[List] = None,
        tool_name: Optional[str] = None,
    ) -> Message:
        if sequence_num is None:
            # Auto-increment per session
            last = await self._last_seq(db, session_id)
            sequence_num = last + 1

        msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            sequence_num=sequence_num,
            knowledge_references=knowledge_references,
            token_usage=token_usage,
            tool_calls=tool_calls,
            tool_name=tool_name,
        )
        db.add(msg)
        await db.flush()
        await db.refresh(msg)
        return msg

    async def _last_seq(self, db: AsyncSession, session_id: str) -> int:
        result = await db.execute(
            select(func.max(Message.sequence_num)).where(
                Message.session_id == session_id
            )
        )
        return result.scalar() or -1

    async def list_by_session(
        self,
        db: AsyncSession,
        session_id: str,
        skip: int = 0,
        limit: int = 500,
    ) -> List[Message]:
        stmt = (
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.is_active == True,
            )
            .order_by(Message.sequence_num)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_by_session(self, db: AsyncSession, session_id: str) -> int:
        result = await db.execute(
            select(func.count())
            .select_from(Message)
            .where(
                Message.session_id == session_id,
                Message.is_active == True,
            )
        )
        return result.scalar() or 0


# Singleton
message_crud = MessageCRUD()
