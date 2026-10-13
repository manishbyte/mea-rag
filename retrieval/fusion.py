
from __future__ import annotations

from typing import Any, Sequence


class ReciprocalRankFusion:
    """Merge ranked retrieval results using Reciprocal Rank Fusion."""

    def __init__(
        self,
        rrf_k: int = 60,
        default_top_k: int = 10,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1.")

        self.rrf_k = rrf_k
        self.default_top_k = default_top_k

    @staticmethod
    def _get_chunk_id(result: dict[str, Any]) -> str | None:
        """Return a usable chunk identifier."""
        chunk_id = result.get("chunk_id")

        if chunk_id is None:
            return None

        return str(chunk_id)

    @staticmethod
    def _get_rank(
        result: dict[str, Any],
        fallback_rank: int,
        rank_key: str,
    ) -> int:
        """Get a valid positive rank from a result."""
        rank = result.get(rank_key, result.get("rank", fallback_rank))

        try:
            parsed_rank = int(rank)
        except (TypeError, ValueError):
            return fallback_rank

        return parsed_rank if parsed_rank > 0 else fallback_rank

    def fuse(
        self,
        vector_results: Sequence[dict[str, Any]],
        bm25_results: Sequence[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Combine vector and BM25 results into one ranked list.

        RRF score:
            sum(1 / (rrf_k + rank)) for each retrieval method

        A chunk found by both methods receives contributions from both.
        """

        result_limit = (
            self.default_top_k if top_k is None else top_k
        )

        if result_limit < 1:
            raise ValueError("top_k must be at least 1.")

        fused: dict[str, dict[str, Any]] = {}

        retrieval_lists = [
            (vector_results, "vector_rank", "vector"),
            (bm25_results, "bm25_rank", "bm25"),
        ]

        for results, rank_key, method in retrieval_lists:
            seen_in_list: set[str] = set()

            for fallback_rank, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    continue

                chunk_id = self._get_chunk_id(result)

                if chunk_id is None:
                    continue

                # Avoid counting duplicate IDs twice within one retriever.
                if chunk_id in seen_in_list:
                    continue

                seen_in_list.add(chunk_id)

                rank = self._get_rank(
                    result=result,
                    fallback_rank=fallback_rank,
                    rank_key=rank_key,
                )

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        **result,
                        "chunk_id": chunk_id,
                        "rrf_score": 0.0,
                        "retrieval_methods": [],
                        "retrieval_ranks": {},
                    }

                entry = fused[chunk_id]

                # Preserve useful fields if the other retriever supplied them.
                for key, value in result.items():
                    if entry.get(key) is None and value is not None:
                        entry[key] = value

                entry["rrf_score"] += 1.0 / (self.rrf_k + rank)
                entry["retrieval_methods"].append(method)
                entry["retrieval_ranks"][method] = rank

        ranked_results = sorted(
            fused.values(),
            key=lambda item: (
                -item["rrf_score"],
                item["chunk_id"],
            ),
        )

        final_results: list[dict[str, Any]] = []

        for final_rank, result in enumerate(
            ranked_results[:result_limit],
            start=1,
        ):
            result["fusion_rank"] = final_rank
            result["retrieval_method"] = "hybrid"
            final_results.append(result)

        return final_results
