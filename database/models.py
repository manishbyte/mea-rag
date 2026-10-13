
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


class TimestampMixin:
    """Common creation and update timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class User(Base, TimestampMixin):
    """Registered application user."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    email: Mapped[str] = mapped_column(
        String(320),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    full_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("true"),
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text("false"),
        nullable=False,
    )

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Conversation(Base, TimestampMixin):
    """A chat session owned by one user."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(200),
        default="New conversation",
        server_default=text("'New conversation'"),
        nullable=False,
    )

    is_archived: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text("false"),
        nullable=False,
    )

    user: Mapped[User] = relationship(
        back_populates="conversations",
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )

    __table_args__ = (
        Index(
            "ix_conversations_user_updated",
            "user_id",
            "updated_at",
        ),
    )


class MessageRole(str, enum.Enum):
    """Allowed message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(Base):
    """A message stored within a conversation."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    role: Mapped[MessageRole] = mapped_column(
        Enum(
            MessageRole,
            name="message_role",
            values_callable=lambda enum_class: [
                member.value for member in enum_class
            ],
        ),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
        server_default=text("'[]'"),
        nullable=False,
    )

    # Optional metadata about the RAG execution.
    model_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    context_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    latency_ms: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        server_default=text("'{}'"),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    conversation: Mapped[Conversation] = relationship(
        back_populates="messages",
    )

    __table_args__ = (
        Index(
            "ix_messages_conversation_created",
            "conversation_id",
            "created_at",
        ),
    )


class IndexingState(str, enum.Enum):
    """Lifecycle states for document indexing."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentIndexStatus(Base, TimestampMixin):
    """
    Tracks ingestion and indexing of a source document.

    This record will let FastAPI determine whether a PDF needs
    processing after deployment or application restart.
    """

    __tablename__ = "document_index_status"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    document_key: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    source_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )

    file_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    state: Mapped[IndexingState] = mapped_column(
        Enum(
            IndexingState,
            name="indexing_state",
            values_callable=lambda enum_class: [
                member.value for member in enum_class
            ],
        ),
        default=IndexingState.PENDING,
        server_default=text("'pending'"),
        nullable=False,
        index=True,
    )

    total_pages: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    total_chunks: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    embedded_chunks: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "document_key",
            name="uq_document_index_status_document_key",
        ),
    )
