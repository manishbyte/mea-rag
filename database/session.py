
from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

load_dotenv()


def _get_database_url() -> str:
    """
    Read and normalize DATABASE_URL.

    Supports PostgreSQL URLs supplied by local installations
    and cloud providers.
    """
    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Add it to your .env file."
        )

    # Normalize common PostgreSQL URL formats for SQLAlchemy asyncio.
    if database_url.startswith("postgres://"):
        database_url = database_url.replace(
            "postgres://",
            "postgresql+asyncpg://",
            1,
        )
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace(
            "postgresql://",
            "postgresql+asyncpg://",
            1,
        )

    if not database_url.startswith("postgresql+asyncpg://"):
        raise ValueError(
            "DATABASE_URL must use PostgreSQL. Supported formats include "
            "postgresql:// and postgresql+asyncpg://."
        )

    return database_url


def _create_engine() -> AsyncEngine:
    """Create the async SQLAlchemy engine."""
    database_url = _get_database_url()

    is_sqlite = database_url.startswith("sqlite")

    # Pool settings are suitable for a small API deployment.
    # SQLAlchemy's async PostgreSQL engine uses connection pooling.
    engine_options: dict[str, Any] = {
        "pool_pre_ping": True,
        "echo": os.getenv("SQL_ECHO", "false").lower() == "true",
    }

    if not is_sqlite:
        engine_options.update(
            {
                "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
                "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "5")),
                "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "1800")),
            }
        )

    return create_async_engine(
        database_url,
        **engine_options,
    )


# One engine and session factory per application process.
engine: AsyncEngine = _create_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.

    The session is always closed after the request finishes.
    """
    async with AsyncSessionLocal() as session:
        yield session


async def check_database_connection() -> bool:
    """Return True when PostgreSQL is reachable."""
    from sqlalchemy import text

    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        return True

    except Exception:
        logger.exception("PostgreSQL connectivity check failed")
        return False


async def close_database() -> None:
    """Dispose of pooled database connections during application shutdown."""
    await engine.dispose()
    logger.info("Database connection pool closed")
