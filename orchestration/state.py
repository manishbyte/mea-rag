
from __future__ import annotations

from typing import Any, Literal, TypedDict


class ChatMessage(TypedDict, total=False):
    """A conversation message loaded from PostgreSQL."""

    message_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str

    # Citations attached to assistant messages.
    citations: list[dict[str, Any]]

    # Additional message information, if available.
    metadata: dict[str, Any]


class RAGState(TypedDict, total=False):
    """LangGraph state for a PostgreSQL-backed RAG chatbot.

    PostgreSQL is the source of truth for persistent conversations.
    This state contains data required to process one chat turn.
    """

    # --------------------------------------------------
    # Conversation and user identity
    # --------------------------------------------------
    conversation_id: str
    user_id: str

    # --------------------------------------------------
    # Current user request
    # --------------------------------------------------
    question: str
    current_message_id: str
    current_message_created_at: str

    # --------------------------------------------------
    # Conversation context loaded from PostgreSQL
    # --------------------------------------------------
    conversation_history: list[ChatMessage]

    # --------------------------------------------------
    # Retrieval results
    # --------------------------------------------------
    retrieved_documents: list[dict[str, Any]]
    retrieval_metadata: dict[str, Any]

    # --------------------------------------------------
    # Refusal handling
    # --------------------------------------------------
    should_refuse: bool
    refusal_reason: str | None
    refusal_message: str | None

    # --------------------------------------------------
    # Generated answer
    # --------------------------------------------------
    answer: str
    model: str | None
    context_count: int

    # --------------------------------------------------
    # Citations and answer validation
    # --------------------------------------------------
    citations: list[dict[str, Any]]
    validation_result: dict[str, Any]
    is_valid: bool
    validation_issues: list[str]

    # --------------------------------------------------
    # Final assistant message
    # --------------------------------------------------
    assistant_message: ChatMessage
    response: dict[str, Any]

    # --------------------------------------------------
    # Execution status
    # --------------------------------------------------
    status: Literal[
        "pending",
        "processing",
        "completed",
        "refused",
        "failed",
    ]

    error: str | None
