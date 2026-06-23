"""
Conversation API

Session and message management endpoints.
"""

import uuid
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.session_crud import session_crud
from app.services.message_crud import message_crud

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/conversations", tags=["历史对话"])

# Default user ID (until auth is implemented)
DEFAULT_USER_ID = "default-user"


# ─── Models ───

class CreateSessionRequest(BaseModel):
    title: str = Field("New Chat", max_length=255)
    kb_id: Optional[str] = None
    session_type: str = Field("knowledge_qa", pattern="^(knowledge_qa|agent)$")


class SessionResponse(BaseModel):
    id: str
    title: str
    session_type: str
    message_count: int
    total_tokens: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sequence_num: int
    knowledge_references: Optional[List[dict]] = None
    token_usage: Optional[dict] = None
    created_at: Optional[str] = None


# ─── Endpoints ───

@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
):
    """List all conversations for the current user."""
    sessions = await session_crud.list_by_user(
        db, DEFAULT_USER_ID, skip=skip, limit=limit
    )
    return [
        SessionResponse(
            id=s.id,
            title=s.title,
            session_type=s.session_type,
            message_count=s.message_count,
            total_tokens=s.total_tokens,
            created_at=s.created_at.isoformat() if s.created_at else None,
            updated_at=s.updated_at.isoformat() if s.updated_at else None,
        )
        for s in sessions
    ]


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    req: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation session."""
    session = await session_crud.create(
        db,
        user_id=DEFAULT_USER_ID,
        title=req.title,
        knowledge_base_id=req.kb_id,
        session_type=req.session_type,
    )
    return SessionResponse(
        id=session.id,
        title=session.title,
        session_type=session.session_type,
        message_count=session.message_count,
        total_tokens=session.total_tokens,
        created_at=session.created_at.isoformat() if session.created_at else None,
        updated_at=session.updated_at.isoformat() if session.updated_at else None,
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get session details."""
    session = await session_crud.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return SessionResponse(
        id=session.id,
        title=session.title,
        session_type=session.session_type,
        message_count=session.message_count,
        total_tokens=session.total_tokens,
        created_at=session.created_at.isoformat() if session.created_at else None,
        updated_at=session.updated_at.isoformat() if session.updated_at else None,
    )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation session."""
    session = await session_crud.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    await session_crud.delete(db, session_id)
    return {"status": "deleted", "session_id": session_id}


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
async def list_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 500,
):
    """List messages in a session."""
    session = await session_crud.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = await message_crud.list_by_session(db, session_id, skip=skip, limit=limit)
    return [
        MessageResponse(
            id=m.id,
            role=m.role,
            content=m.content,
            sequence_num=m.sequence_num,
            knowledge_references=m.knowledge_references,
            token_usage=m.token_usage,
            created_at=m.created_at.isoformat() if m.created_at else None,
        )
        for m in messages
    ]
