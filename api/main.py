
from __future__ import annotations

import inspect
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

load_dotenv()

from api.routes.auth import router as auth_router
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router
from database.models import Base
from database.session import close_database, engine

logger = logging.getLogger(__name__)

APP_NAME = "MEA Annual Report RAG API"
APP_VERSION = "1.0.0"
API_PREFIX = "/api"


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


async def _initialize_database() -> None:
    """Verify PostgreSQL and initialize missing database tables."""
    async with engine.begin() as connection:
        await connection.execute(text("SELECT 1"))
        await connection.run_sync(Base.metadata.create_all)

    logger.info("PostgreSQL verified and tables initialized.")


async def _close_resources(resources: Any) -> None:
    """Close application resources when they expose close/aclose."""
    if resources is None:
        return

    if isinstance(resources, dict):
        candidates = list(resources.values())
    elif isinstance(resources, (list, tuple, set)):
        candidates = list(resources)
    else:
        candidates = [resources]

    seen: set[int] = set()

    for resource in candidates:
        if resource is None or id(resource) in seen:
            continue

        seen.add(id(resource))

        close_method = getattr(resource, "aclose", None)
        if not callable(close_method):
            close_method = getattr(resource, "close", None)

        if not callable(close_method):
            continue

        try:
            result = close_method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(
                "Failed to close resource %s",
                type(resource).__name__,
            )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize shared resources at startup and release them at shutdown."""
    _configure_logging()

    application.state.rag_graph = None
    application.state.rag_resources = None
    application.state.startup_complete = False

    try:
        await _initialize_database()

        # The existing root main.py returns (graph, resources).
        from main import build_application

        graph, resources = await build_application()

        application.state.rag_graph = graph
        application.state.rag_resources = resources
        application.state.startup_complete = True

        logger.info("%s started successfully.", APP_NAME)

        yield

    except Exception:
        logger.exception("Application startup or runtime failed.")
        raise

    finally:
        application.state.startup_complete = False
        application.state.rag_graph = None

        await _close_resources(application.state.rag_resources)
        application.state.rag_resources = None

        await close_database()
        logger.info("%s shut down.", APP_NAME)


app = FastAPI(
    title=APP_NAME,
    description=(
        "Authenticated question-answering API for the Ministry of "
        "External Affairs Annual Report 2025, powered by a hybrid "
        "retrieval-augmented generation pipeline."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Register each router exactly once.
app.include_router(auth_router, prefix=API_PREFIX)
app.include_router(health_router, prefix=API_PREFIX)
app.include_router(chat_router, prefix=API_PREFIX)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": f"{API_PREFIX}/health/live",
    }


@app.get("/api/health/startup", tags=["Health"])
async def startup_status() -> dict[str, str]:
    if not app.state.startup_complete:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application startup is not complete.",
        )

    return {"status": "started"}
