
from __future__ import annotations

import asyncio
from typing import Any

from retrieval.bm25_search import BM25Search
from retrieval.vector_search import VectorSearch


class HybridSearch:
    """Run vector and BM25 retrieval concurrently."""

    def __init__(
        self,
        vector_search: VectorSearch | None = None,
        bm25_search: BM25Search | None = None,
        default_top_k: int = 10,
        max_concurrent_searches: int = 4,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1.")

        if max_concurrent_searches < 1:
            raise ValueError(
                "max_concurrent_searches must be at least 1."
            )

        self.vector_search = vector_search or VectorSearch(
            default_top_k=default_top_k
        )
        self.bm25_search = bm25_search or BM25Search(
            default_top_k=default_top_k
        )

        self.default_top_k = default_top_k
        self._semaphore = asyncio.Semaphore(max_concurrent_searches)

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute vector and BM25 searches concurrently.

        Returns each retriever's results separately. Ranking fusion
        is intentionally handled by the fusion module.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty.")

        result_limit = (
            self.default_top_k if top_k is None else top_k
        )

        if result_limit < 1:
            raise ValueError("top_k must be at least 1.")

        clean_query = query.strip()

        async with self._semaphore:
            vector_results, bm25_results = await asyncio.gather(
                self.vector_search.search(
                    query=clean_query,
                    top_k=result_limit,
                    where=where,
                ),
                self.bm25_search.search(
                    query=clean_query,
                    top_k=result_limit,
                    where=where,
                ),
            )

        return {
            "query": clean_query,
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "vector_count": len(vector_results),
            "bm25_count": len(bm25_results),
        }
