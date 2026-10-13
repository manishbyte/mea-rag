
"""Deterministic retrieval metrics for RAG evaluation."""

from __future__ import annotations

import math
from typing import Any, Iterable


def _unique(values: Iterable[Any]) -> list[str]:
    """Preserve ranking order while removing duplicate IDs."""
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value is None:
            continue

        value = str(value)

        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


class RetrievalMetrics:
    """Compute retrieval metrics against labelled relevant chunk IDs."""

    @staticmethod
    def recall_at_k(
        retrieved_ids: Iterable[Any],
        relevant_ids: Iterable[Any],
        k: int = 5,
    ) -> float | None:
        relevant = set(_unique(relevant_ids))

        if not relevant:
            return None

        retrieved = _unique(retrieved_ids)[:k]
        return len(set(retrieved) & relevant) / len(relevant)

    @staticmethod
    def precision_at_k(
        retrieved_ids: Iterable[Any],
        relevant_ids: Iterable[Any],
        k: int = 5,
    ) -> float | None:
        relevant = set(_unique(relevant_ids))

        if not relevant:
            return None

        if k < 1:
            raise ValueError("k must be at least 1.")

        retrieved = _unique(retrieved_ids)[:k]

        # Missing results count as non-relevant positions.
        return len(set(retrieved) & relevant) / k

    @staticmethod
    def reciprocal_rank(
        retrieved_ids: Iterable[Any],
        relevant_ids: Iterable[Any],
        k: int = 5,
    ) -> float | None:
        relevant = set(_unique(relevant_ids))

        if not relevant:
            return None

        for rank, chunk_id in enumerate(_unique(retrieved_ids)[:k], start=1):
            if chunk_id in relevant:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def hit_rate_at_k(
        retrieved_ids: Iterable[Any],
        relevant_ids: Iterable[Any],
        k: int = 5,
    ) -> float | None:
        recall = RetrievalMetrics.recall_at_k(
            retrieved_ids, relevant_ids, k
        )

        if recall is None:
            return None

        return float(recall > 0)

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: Iterable[Any],
        relevant_ids: Iterable[Any],
        k: int = 5,
    ) -> float | None:
        relevant = set(_unique(relevant_ids))

        if not relevant:
            return None

        retrieved = _unique(retrieved_ids)[:k]

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, chunk_id in enumerate(retrieved, start=1)
            if chunk_id in relevant
        )

        ideal_relevant_count = min(len(relevant), k)
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )

        return dcg / ideal_dcg if ideal_dcg else 0.0

    @classmethod
    def evaluate_query(
        cls,
        retrieved_ids: Iterable[Any],
        relevant_ids: Iterable[Any],
        k: int = 5,
    ) -> dict[str, Any]:
        retrieved = _unique(retrieved_ids)
        relevant = _unique(relevant_ids)

        return {
            "retrieved_count": len(retrieved),
            "relevant_count": len(relevant),
            f"recall_at_{k}": cls.recall_at_k(retrieved, relevant, k),
            f"precision_at_{k}": cls.precision_at_k(
                retrieved, relevant, k
            ),
            f"mrr_at_{k}": cls.reciprocal_rank(
                retrieved, relevant, k
            ),
            f"hit_rate_at_{k}": cls.hit_rate_at_k(
                retrieved, relevant, k
            ),
            f"ndcg_at_{k}": cls.ndcg_at_k(
                retrieved, relevant, k
            ),
            "matched_chunk_ids": sorted(
                set(retrieved[:k]) & set(relevant)
            ),
            "retrieved_chunk_ids": retrieved[:k],
            "relevant_chunk_ids": relevant,
        }

    @staticmethod
    def aggregate(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Average each metric over questions with labelled ground truth."""
        if not results:
            return {}

        metric_names = [
            key
            for key in results[0]
            if key.startswith(
                ("recall_at_", "precision_at_", "mrr_at_",
                 "hit_rate_at_", "ndcg_at_")
            )
        ]

        aggregate: dict[str, Any] = {}

        for metric in metric_names:
            values = [
                row[metric]
                for row in results
                if isinstance(row.get(metric), (int, float))
                and row[metric] is not None
            ]

            aggregate[metric] = (
                sum(values) / len(values) if values else None
            )
            aggregate[f"{metric}_evaluated_questions"] = len(values)

        return aggregate
