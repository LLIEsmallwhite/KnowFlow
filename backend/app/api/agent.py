"""
Agent Chat API

ReAct Agent endpoints using LangGraph agent graph.
"""

import json
import asyncio
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.graph.agent_graph import agent_graph
from app.services.session_crud import session_crud
from app.services.message_crud import message_crud

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])
DEFAULT_USER_ID = "default-user"


class AgentChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None
    kb_ids: List[str] = Field(default_factory=list)
    stream: bool = Field(False)
    max_iterations: int = Field(10, ge=1, le=20)
    enable_web_search: bool = Field(False)


class AgentChatResponse(BaseModel):
    answer: str
    session_id: str
    iterations: int


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(
    req: AgentChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Synchronous agent Q&A using ReAct graph."""
    title = req.query[:50] + ("..." if len(req.query) > 50 else "")
    session = await session_crud.create(
        db, user_id=DEFAULT_USER_ID, title=title,
        session_type="agent",
    )
    await message_crud.create(db, session_id=session.id, role="user",
                              content=req.query)

    initial_state = {
        "query": req.query,
        "kb_ids": req.kb_ids,
        "iteration": 0,
        "max_iterations": req.max_iterations,
        "messages": [],
        "tool_calls": [],
        "tool_results": [],
        "current_thought": "",
        "final_answer": "",
        "is_complete": False,
    }

    result = agent_graph.invoke(initial_state)

    await message_crud.create(
        db, session_id=session.id, role="assistant",
        content=result.get("final_answer", ""),
    )
    await session_crud.increment_message_count(db, session.id, delta=2)

    return AgentChatResponse(
        answer=result.get("final_answer", ""),
        session_id=session.id,
        iterations=result.get("iteration", 0),
    )


@router.post("/chat/stream")
async def agent_chat_stream(
    req: AgentChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """SSE streaming agent Q&A."""
    title = req.query[:50] + ("..." if len(req.query) > 50 else "")
    session = await session_crud.create(
        db, user_id=DEFAULT_USER_ID, title=title,
        session_type="agent",
    )
    await message_crud.create(db, session_id=session.id, role="user",
                              content=req.query)
    sid = session.id

    async def event_generator():
        initial_state = {
            "query": req.query,
            "kb_ids": req.kb_ids,
            "iteration": 0,
            "max_iterations": req.max_iterations,
            "messages": [],
            "tool_calls": [],
            "tool_results": [],
            "current_thought": "",
            "final_answer": "",
            "is_complete": False,
        }

        try:
            async for event in agent_graph.astream(initial_state):
                for node_name, node_state in event.items():
                    thought = node_state.get("current_thought", "")
                    if thought:
                        yield f"data: {json.dumps({'type': 'thought', 'data': thought, 'node': node_name}, ensure_ascii=False)}\n\n"

                    tool_results = node_state.get("tool_results", [])
                    for tr in tool_results:
                        yield f"data: {json.dumps({'type': 'tool_result', 'data': tr}, ensure_ascii=False)}\n\n"

                    if node_state.get("is_complete"):
                        answer = node_state.get("final_answer", "")
                        yield f"data: {json.dumps({'type': 'done', 'data': {'answer': answer, 'iterations': node_state.get('iteration', 0)}}, ensure_ascii=False)}\n\n"

            # Save assistant message
            final_answer = ""
            try:
                async with db.bind.connect() as conn:
                    from sqlalchemy.ext.asyncio import AsyncSession as AS
                    s = AS(conn)
                    await message_crud.create(
                        s, session_id=sid, role="assistant",
                        content=final_answer or "（完成）",
                    )
                    await s.commit()
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)[:300]})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
