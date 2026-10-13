
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
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not configured."
        )

    # Normalize PostgreSQL URL schemes for SQLAlchemy asyncpg.
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

    return database_url


def _create_engine() -> AsyncEngine:
    from sqlalchemy.engine import make_url

    database_url = _get_database_url()
    url = make_url(database_url)

    # Move SSL configuration from URL query parameters to
    # asyncpg's connect arguments.
    query = dict(url.query)
    sslmode = query.pop("sslmode", None)

    connect_args = {}

    if sslmode:
        if sslmode in ("require", "verify-ca", "verify-full"):
            connect_args["ssl"] = sslmode
        elif sslmode in ("disable", "allow", "prefer"):
            connect_args["ssl"] = False

    url = url.set(query=query)

    engine_options = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "connect_args": connect_args,
    }

    return create_async_engine(
        url,
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
