
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


async def _initialize_database() -> None:
    """Verify PostgreSQL and initialize missing database tables."""
    logger.info("Checking PostgreSQL connection and initializing tables.")

    async with engine.begin() as connection:
        await connection.execute(text("SELECT 1"))
        await connection.run_sync(Base.metadata.create_all)

    logger.info("PostgreSQL verified and tables initialized.")


def _build_rag_application_sync() -> tuple[Any, Any]:
    """
    Import and construct RAG components inside a worker thread.

    This keeps synchronous imports and component initialization
    from blocking the FastAPI event loop.
    """
    logger.info("Importing root RAG application module.")

    from main import build_application_sync

    logger.info("Root RAG module imported.")
    logger.info("Constructing RAG application components.")

    result = build_application_sync()

    logger.info("RAG application construction finished.")

    return result


async def _close_resources(resources: Any) -> None:
    """Close application resources that expose close/aclose methods."""
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


async def _initialize_rag(application: FastAPI) -> None:
    """Initialize RAG in the background and expose its actual status."""
    resources: Any = None
    started_at = time.monotonic()

    application.state.rag_status = "initializing"
    application.state.rag_error = None

    try:
        logger.info("Starting background RAG initialization.")
        logger.info(
            "RAG initialization is running in a worker thread."
        )

        graph, resources = await asyncio.to_thread(
            _build_rag_application_sync
        )

        if application.state.shutting_down:
            logger.warning(
                "RAG initialization finished during shutdown."
            )
            await _close_resources(resources)
            resources = None
            return

        if graph is None:
            raise RuntimeError(
                "RAG initialization returned an empty graph."
            )

        application.state.rag_graph = graph
        application.state.rag_resources = resources
        application.state.startup_complete = True
        application.state.rag_status = "ready"
        application.state.rag_error = None

        elapsed = time.monotonic() - started_at

        logger.info(
            "RAG initialization completed successfully in %.2f seconds.",
            elapsed,
        )
        logger.info("%s is ready to answer questions.", APP_NAME)

    except asyncio.CancelledError:
        application.state.rag_status = "stopped"
        application.state.startup_complete = False

        logger.warning("Background RAG initialization was cancelled.")

        await _close_resources(resources)
        raise

    except Exception as exc:
        application.state.rag_graph = None
        application.state.startup_complete = False
        application.state.rag_status = "failed"
        application.state.rag_error = (
            f"{type(exc).__name__}: {str(exc)[:500]}"
        )

        logger.exception(
            "RAG initialization failed after %.2f seconds.",
            time.monotonic() - started_at,
        )

        await _close_resources(resources)

        logger.error(
            "The API remains available, but RAG chat is not ready."
        )


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """Initialize PostgreSQL and start RAG initialization in background."""
    _configure_logging()

    application.state.rag_graph = None
    application.state.rag_resources = None
    application.state.startup_complete = False
    application.state.rag_status = "initializing"
    application.state.rag_error = None
    application.state.shutting_down = False
    application.state.rag_task = None

    try:
        logger.info("Initializing database.")

        await _initialize_database()

        logger.info(
            "Database initialized successfully."
        )

        application.state.rag_task = asyncio.create_task(
            _initialize_rag(application),
            name="mea-rag-initialization",
        )

        logger.info(
            "API startup is continuing while RAG initializes "
            "in the background."
        )

        yield

    except Exception:
        logger.exception("Application startup or runtime failed.")
        raise

    finally:
        application.state.shutting_down = True
        application.state.startup_complete = False

        rag_task = application.state.rag_task

        if rag_task is not None and not rag_task.done():
            rag_task.cancel()

            await asyncio.gather(
                rag_task,
                return_exceptions=True,
            )

        application.state.rag_graph = None

        await _close_resources(
            application.state.rag_resources
        )

        application.state.rag_resources = None

        try:
            await close_database()
        except Exception:
            logger.exception("Failed to close database connections.")

        application.state.rag_status = "stopped"

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
    allow_methods=[
        "GET",
        "POST",
        "PATCH",
        "DELETE",
        "OPTIONS",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
    ],
)


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
async def startup_status() -> dict[str, Any]:
    """Return the current API and RAG initialization status."""
    return {
        "status": "running",
        "rag_status": app.state.rag_status,
        "rag_error": app.state.rag_error,
    }


@app.get("/api/health/ready", tags=["Health"])
async def readiness() -> dict[str, str]:
    """Return success only when the RAG pipeline is ready."""
    if not app.state.startup_complete:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "rag_status": app.state.rag_status,
                "rag_error": app.state.rag_error,
            },
        )

    return {
        "status": "ready",
        "rag_status": "ready",
    }
