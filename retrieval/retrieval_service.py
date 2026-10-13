
from __future__ import annotations

import asyncio
from typing import Any

from retrieval.fusion import ReciprocalRankFusion
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker


class RetrievalService:
    """Coordinate hybrid retrieval, fusion, and reranking."""

    def __init__(
        self,
        hybrid_search: HybridSearch | None = None,
        fusion: ReciprocalRankFusion | None = None,
        reranker: Reranker | None = None,
        candidate_top_k: int = 20,
        final_top_k: int = 5,
    ) -> None:
        if candidate_top_k < 1:
            raise ValueError("candidate_top_k must be at least 1.")

        if final_top_k < 1:
            raise ValueError("final_top_k must be at least 1.")

        if final_top_k > candidate_top_k:
            raise ValueError(
                "final_top_k cannot exceed candidate_top_k."
            )

        self.hybrid_search = hybrid_search or HybridSearch(
            default_top_k=candidate_top_k
        )

        self.fusion = fusion or ReciprocalRankFusion(
            rrf_k=60,
            default_top_k=candidate_top_k,
        )

        self.reranker = reranker or Reranker(
            default_top_k=final_top_k
        )

        self.candidate_top_k = candidate_top_k
        self.final_top_k = final_top_k

        self._search_semaphore = asyncio.Semaphore(4)

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve, fuse, and rerank relevant document chunks.

        Returns the final chunks and intermediate retrieval results
        for debugging, evaluation, and observability.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty.")

        result_limit = (
            self.final_top_k if top_k is None else top_k
        )

        if result_limit < 1:
            raise ValueError("top_k must be at least 1.")

        if result_limit > self.candidate_top_k:
            raise ValueError(
                "top_k cannot exceed candidate_top_k."
            )

        clean_query = query.strip()

        async with self._search_semaphore:
            # Stage 1: Run vector search and BM25 concurrently.
            hybrid_results = await self.hybrid_search.search(
                query=clean_query,
                top_k=self.candidate_top_k,
                where=where,
            )

            vector_results = hybrid_results["vector_results"]
            bm25_results = hybrid_results["bm25_results"]

            # Stage 2: Combine rankings with RRF.
            fused_results = self.fusion.fuse(
                vector_results=vector_results,
                bm25_results=bm25_results,
                top_k=self.candidate_top_k,
            )

            # Stage 3: Rerank the fused candidates.
            reranked_results = await self.reranker.rerank(
                query=clean_query,
                documents=fused_results,
                top_k=result_limit,
            )

        return {
            "query": clean_query,
            "results": reranked_results,
            "result_count": len(reranked_results),
            "vector_count": len(vector_results),
            "bm25_count": len(bm25_results),
            "fused_count": len(fused_results),
            "candidate_top_k": self.candidate_top_k,
            "final_top_k": result_limit,
            "retrieval_pipeline": [
                "vector_search",
                "bm25_search",
                "reciprocal_rank_fusion",
                "cross_encoder_reranking",
            ],
        }

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Convenience method returning only the final chunks."""

        response = await self.retrieve(
            query=query,
            top_k=top_k,
            where=where,
        )

        return response["results"]
