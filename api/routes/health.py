
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import DocumentIndexStatus
from database.session import AsyncSessionLocal, check_database_connection

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """
    Confirm that the FastAPI process is alive.

    This endpoint intentionally does not depend on PostgreSQL,
    the PDF, or the embedding index.
    """
    return {
        "status": "alive",
        "service": "mea-rag-api",
    }


@router.get("/ready")
async def readiness_check(request: Request) -> JSONResponse:
    """
    Check whether the API's required dependencies are ready.

    A running API is not considered fully ready until PostgreSQL
    is reachable and the document index is marked ready.
    """
    checks: dict[str, Any] = {
        "api": "ready",
        "database": "unknown",
        "document_index": "unknown",
    }

    # 1. Check PostgreSQL.
    database_ready = await check_database_connection()
    checks["database"] = "ready" if database_ready else "unavailable"

    # 2. Check document indexing state in PostgreSQL.
    index_ready = False

    if database_ready:
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(DocumentIndexStatus).where(
                        DocumentIndexStatus.document_key
                        == "mea_annual_report_2025"
                    )
                )
                document_status = result.scalar_one_or_none()

                if document_status is None:
                    checks["document_index"] = "not_initialized"
                else:
                    checks["document_index"] = document_status.state.value
                    checks["document"] = {
                        "key": document_status.document_key,
                        "total_pages": document_status.total_pages,
                        "total_chunks": document_status.total_chunks,
                        "embedded_chunks": document_status.embedded_chunks,
                        "error": document_status.error_message,
                    }

                    index_ready = document_status.state.value == "ready"

        except Exception:
            logger.exception("Failed to read document indexing status")
            checks["document_index"] = "status_check_failed"

    # Also check whether application startup initialized the RAG graph.
    application_state = request.app.state
    graph = getattr(application_state, "rag_graph", None)

    checks["rag_graph"] = (
        "initialized" if graph is not None else "not_initialized"
    )

    # The application is ready only when all critical dependencies are ready.
    ready = (
        database_ready
        and index_ready
        and graph is not None
    )

    checks["status"] = "ready" if ready else "not_ready"

    response_status = (
        status.HTTP_200_OK
        if ready
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=response_status,
        content=checks,
    )


@router.get("/database")
async def database_health_check() -> JSONResponse:
    """Check database connectivity independently."""
    try:
        connected = await check_database_connection()
    except Exception:
        logger.exception("Database health check failed")
        connected = False

    if connected:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "database": "connected",
            },
        )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "unhealthy",
            "database": "unavailable",
        },
    )
