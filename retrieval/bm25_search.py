
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Search:
    """Retrieve relevant MEA report chunks using BM25 keyword scoring."""

    def __init__(
        self,
        chunks_path: str | Path | None = None,
        default_top_k: int = 10,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1.")

        project_root = Path(__file__).resolve().parent.parent

        self.chunks_path = Path(
            chunks_path
            if chunks_path is not None
            else project_root / "data" / "processed" / "mea_chunks.json"
        )

        self.default_top_k = default_top_k

        self._documents: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

        self._load_lock = asyncio.Lock()
        self._loaded = False

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text into lowercase word tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def _load_index_sync(
        self,
    ) -> tuple[list[dict[str, Any]], BM25Okapi]:
        """Load chunks and construct the BM25 index."""

        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_path}\n"
                "Run the ingestion pipeline before BM25 search."
            )

        with self.chunks_path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        if not isinstance(chunks, list):
            raise ValueError(
                "Expected mea_chunks.json to contain a JSON list."
            )

        documents: list[dict[str, Any]] = []
        tokenized_corpus: list[list[str]] = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            text = chunk.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            tokens = self._tokenize(text)
            if not tokens:
                continue

            metadata = chunk.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            documents.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "text": text,
                    "metadata": metadata,
                }
            )
            tokenized_corpus.append(tokens)

        if not documents:
            raise ValueError(
                "No searchable text chunks were found in the chunks file."
            )

        return documents, BM25Okapi(tokenized_corpus)

    async def _ensure_loaded(self) -> None:
        """Build the index once, without blocking the event loop."""

        if self._loaded:
            return

        async with self._load_lock:
            if self._loaded:
                return

            self._documents, self._bm25 = await asyncio.to_thread(
                self._load_index_sync
            )
            self._loaded = True

    @staticmethod
    def _matches_filter(
        metadata: dict[str, Any],
        where: dict[str, Any] | None,
    ) -> bool:
        """Match metadata fields using exact equality."""

        if not where:
            return True

        return all(
            metadata.get(key) == expected_value
            for key, expected_value in where.items()
        )

    def _search_sync(
        self,
        query_tokens: list[str],
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Score and rank chunks using BM25."""

        if self._bm25 is None:
            raise RuntimeError("BM25 index has not been initialized.")

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        results: list[dict[str, Any]] = []

        for index in ranked_indices:
            score = float(scores[index])

            # Zero-score chunks have no keyword match.
            if score <= 0:
                break

            document = self._documents[index]
            metadata = document["metadata"]

            if not self._matches_filter(metadata, where):
                continue

            rank = len(results) + 1

            results.append(
                {
                    "chunk_id": document["chunk_id"],
                    "text": document["text"],
                    "metadata": metadata,
                    "source": metadata.get("source"),
                    "page_number": metadata.get("page_number"),
                    "bm25_score": score,
                    "rank": rank,
                    "bm25_rank": rank,
                    "retrieval_method": "bm25",
                }
            )

            if len(results) >= top_k:
                break

        return results

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the highest-scoring keyword matches."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty.")

        result_limit = (
            self.default_top_k if top_k is None else top_k
        )

        if result_limit < 1:
            raise ValueError("top_k must be at least 1.")

        await self._ensure_loaded()

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        return await asyncio.to_thread(
            self._search_sync,
            query_tokens,
            result_limit,
            where,
        )

    async def index_status(self) -> dict[str, Any]:
        """Report whether the BM25 index has been loaded."""

        await self._ensure_loaded()

        return {
            "loaded": self._loaded,
            "chunk_count": len(self._documents),
            "chunks_path": str(self.chunks_path),
            "retrieval_method": "bm25",
        }
