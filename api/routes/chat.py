
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import AuthenticatedUser, DatabaseSession
from database.models import Conversation, Message, MessageRole

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# -------------------------------------------------------------------
# Request and response schemas
# -------------------------------------------------------------------

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Question about the MEA Annual Report 2025.",
    )
    conversation_id: uuid.UUID | None = Field(
        default=None,
        description="Existing conversation ID, or omit to start a new chat.",
    )


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
    question: str
    answer: str
    citations: list[Any] = Field(default_factory=list)
    context_count: int = 0
    latency_ms: int = 0
    created_at: datetime


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    citations: list[Any] = Field(default_factory=list)
    model_name: str | None = None
    context_count: int | None = None
    latency_ms: int | None = None
    created_at: datetime


class ConversationMessagesResponse(BaseModel):
    conversation_id: uuid.UUID
    title: str
    messages: list[MessageResponse]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _get_answer(graph_state: dict[str, Any]) -> str:
    """Extract the final answer from the RAG graph's output state."""

    response = graph_state.get("response")
    answer = graph_state.get("answer")

    # Some graph implementations put the answer in a response object.
    if isinstance(response, dict):
        answer = (
            response.get("answer")
            or response.get("content")
            or answer
        )
    elif isinstance(response, str) and response.strip():
        answer = response

    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    raise RuntimeError(
        "The RAG graph completed without returning a valid answer."
    )


def _get_citations(graph_state: dict[str, Any]) -> list[Any]:
    """Extract citations without inventing citation information."""

    citations = graph_state.get("citations", [])

    response = graph_state.get("response")
    if not citations and isinstance(response, dict):
        citations = response.get("citations", [])

    if citations is None:
        return []

    if isinstance(citations, list):
        return citations

    # A single citation object can be normalized to a list.
    if isinstance(citations, dict):
        return [citations]

    logger.warning("Unexpected citation format returned by RAG graph.")
    return []


def _get_context_count(graph_state: dict[str, Any]) -> int:
    """Count the retrieved documents when the graph exposes them."""

    documents = graph_state.get("retrieved_docs", [])

    if isinstance(documents, list):
        return len(documents)

    return 0


def _get_model_name() -> str | None:
    return os.getenv("GROQ_MODEL") or None


async def _get_owned_conversation(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Conversation:
    """Return a conversation only if it belongs to the logged-in user."""

    result = await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    conversation = result.scalar_one_or_none()

    if conversation is None:
        # Do not reveal whether another user's conversation exists.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return conversation


# -------------------------------------------------------------------
# POST /api/chat
# Ask a question or continue a conversation.
# -------------------------------------------------------------------

@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat(
    payload: ChatRequest,
    request: Request,
    session: DatabaseSession,
    current_user: AuthenticatedUser,
) -> ChatResponse:
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    graph = getattr(request.app.state, "rag_graph", None)

    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The RAG application is not ready yet. Please try again.",
        )

    # Find an existing conversation belonging to this user,
    # or create a new conversation.
    if payload.conversation_id is not None:
        conversation = await _get_owned_conversation(
            session=session,
            conversation_id=payload.conversation_id,
            user_id=current_user.id,
        )
    else:
        conversation = Conversation(
            user_id=current_user.id,
            title=question[:100],
        )
        session.add(conversation)
        await session.flush()

    conversation_id = conversation.id

    # Fetch recent conversation history for the RAG graph.
    history_result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(desc(Message.created_at), desc(Message.id))
        .limit(20)
    )
    previous_messages = list(reversed(history_result.scalars().all()))

    conversation_history = [
        {
            "role": message.role.value,
            "content": message.content,
        }
        for message in previous_messages
        if message.role in (MessageRole.USER, MessageRole.ASSISTANT)
    ]

    # Persist the user's question before calling the model.
    user_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=question,
        citations=[],
    )
    session.add(user_message)

    conversation.updated_at = datetime.now(
        user_message.created_at.tzinfo
        if user_message.created_at is not None
        else None
    ) if user_message.created_at is not None else conversation.updated_at

    # Commit the user message independently so the question is retained
    # even if the model provider or retrieval pipeline subsequently fails.
    await session.commit()
    await session.refresh(user_message)

    started_at = time.perf_counter()

    # Build the initial state expected by the existing LangGraph workflow.
    initial_state: dict[str, Any] = {
        "conversation_id": str(conversation_id),
        "user_id": str(current_user.id),
        "question": question,
        "conversation_history": conversation_history,
        "status": "pending",
    }

    try:
        graph_state = await graph.ainvoke(initial_state)
        answer = _get_answer(graph_state)
        citations = _get_citations(graph_state)
        context_count = _get_context_count(graph_state)
        latency_ms = int((time.perf_counter() - started_at) * 1000)

    except Exception:
        logger.exception(
            "RAG request failed for conversation %s",
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The RAG pipeline could not generate an answer. "
                "Your question has been saved; please try again."
            ),
        ) from None

    # Save the generated answer and its actual citations.
    assistant_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.ASSISTANT,
        content=answer,
        citations=citations,
        model_name=_get_model_name(),
        context_count=context_count,
        latency_ms=latency_ms,
    )
    session.add(assistant_message)

    conversation.updated_at = datetime.now().astimezone()

    try:
        await session.commit()
        await session.refresh(assistant_message)
    except Exception:
        await session.rollback()
        logger.exception(
            "Failed to persist assistant response for conversation %s",
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The answer was generated but could not be saved.",
        ) from None

    return ChatResponse(
        conversation_id=conversation_id,
        user_message_id=user_message.id,
        assistant_message_id=assistant_message.id,
        question=question,
        answer=answer,
        citations=citations,
        context_count=context_count,
        latency_ms=latency_ms,
        created_at=assistant_message.created_at,
    )


# -------------------------------------------------------------------
# GET /api/chat/conversations
# List the current user's conversations.
# -------------------------------------------------------------------

@router.get(
    "/conversations",
    response_model=list[ConversationSummary],
)
async def list_conversations(
    session: DatabaseSession,
    current_user: AuthenticatedUser,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[ConversationSummary]:
    result = await session.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(
            desc(Conversation.updated_at),
            desc(Conversation.id),
        )
        .offset(offset)
        .limit(limit)
    )

    conversations = result.scalars().all()

    return [
        ConversationSummary.model_validate(conversation)
        for conversation in conversations
    ]


# -------------------------------------------------------------------
# GET /api/chat/conversations/{conversation_id}/messages
# Read messages from one of the current user's conversations.
# -------------------------------------------------------------------

@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
)
async def get_conversation_messages(
    conversation_id: uuid.UUID,
    session: DatabaseSession,
    current_user: AuthenticatedUser,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ConversationMessagesResponse:
    conversation = await _get_owned_conversation(
        session=session,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at, Message.id)
        .offset(offset)
        .limit(limit)
    )

    messages = result.scalars().all()

    return ConversationMessagesResponse(
        conversation_id=conversation.id,
        title=conversation.title,
        messages=[
            MessageResponse(
                id=message.id,
                role=message.role.value,
                content=message.content,
                citations=message.citations or [],
                model_name=message.model_name,
                context_count=message.context_count,
                latency_ms=message.latency_ms,
                created_at=message.created_at,
            )
            for message in messages
        ],
    )
