
from __future__ import annotations

import asyncio
from typing import Any

from indexing.index_manager import IndexManager


class VectorSearch:
    """Retrieve semantically relevant MEA report chunks."""

    def __init__(
        self,
        index_manager: IndexManager | None = None,
        default_top_k: int = 10,
        max_concurrent_searches: int = 4,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1.")

        if max_concurrent_searches < 1:
            raise ValueError(
                "max_concurrent_searches must be at least 1."
            )

        self.index_manager = index_manager or IndexManager()
        self.default_top_k = default_top_k
        self._semaphore = asyncio.Semaphore(max_concurrent_searches)

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the vector index and return ranked chunks."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty.")

        result_limit = (
            self.default_top_k if top_k is None else top_k
        )

        if result_limit < 1:
            raise ValueError("top_k must be at least 1.")

        async with self._semaphore:
            results = await self.index_manager.search(
                query=query.strip(),
                top_k=result_limit,
                where=where,
            )

        return [
            {
                **result,
                "retrieval_method": "vector",
                "vector_rank": result.get("rank", rank),
            }
            for rank, result in enumerate(results, start=1)
        ]
