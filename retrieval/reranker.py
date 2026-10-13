
from __future__ import annotations

import asyncio
from typing import Any, Sequence

from sentence_transformers import CrossEncoder


class Reranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        default_top_k: int = 5,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1.")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if max_length < 1:
            raise ValueError("max_length must be at least 1.")

        self.model_name = model_name
        self.default_top_k = default_top_k
        self.batch_size = batch_size
        self.max_length = max_length

        self._model: CrossEncoder | None = None
        self._model_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def _ensure_model_loaded(self) -> None:
        """Load the cross-encoder once without blocking the event loop."""

        if self._model is not None:
            return

        async with self._model_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(
                    CrossEncoder,
                    self.model_name,
                    max_length=self.max_length,
                )

    def _predict_sync(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        """Run cross-encoder inference in the current worker thread."""

        if self._model is None:
            raise RuntimeError("Reranker model has not been loaded.")

        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        return [float(score) for score in scores]

    async def rerank(
        self,
        query: str,
        documents: Sequence[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Score and reorder retrieved chunks by query relevance.

        Each document must contain a non-empty 'text' field.
        Other document fields are preserved in the output.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty.")

        result_limit = (
            self.default_top_k if top_k is None else top_k
        )

        if result_limit < 1:
            raise ValueError("top_k must be at least 1.")

        valid_documents: list[dict[str, Any]] = []

        for document in documents:
            if not isinstance(document, dict):
                continue

            text = document.get("text")

            if isinstance(text, str) and text.strip():
                valid_documents.append(document)

        if not valid_documents:
            return []

        await self._ensure_model_loaded()

        clean_query = query.strip()

        pairs = [
            (clean_query, document["text"])
            for document in valid_documents
        ]

        # Protect model inference if multiple requests share this instance.
        async with self._inference_lock:
            scores = await asyncio.to_thread(
                self._predict_sync,
                pairs,
            )

        ranked_documents = []

        for document, score in zip(valid_documents, scores):
            ranked_documents.append(
                {
                    **document,
                    "reranker_score": score,
                }
            )

        ranked_documents.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        results: list[dict[str, Any]] = []

        for rank, document in enumerate(
            ranked_documents[:result_limit],
            start=1,
        ):
            results.append(
                {
                    **document,
                    "reranker_rank": rank,
                    "retrieval_method": "hybrid_reranked",
                }
            )

        return results
