
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Sequence

import chromadb


class ChromaStore:
    """Manage persistent vector storage using ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "mea_annual_report",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        self._client: chromadb.ClientAPI | None = None
        self._collection: Any = None
        self._initialize_lock = asyncio.Lock()

    def _initialize_sync(self) -> None:
        """Initialize the persistent client and collection."""

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory.resolve())
            )

        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    async def initialize(self) -> None:
        """Initialize ChromaDB without blocking the event loop."""

        async with self._initialize_lock:
            if self._collection is None:
                await asyncio.to_thread(self._initialize_sync)

    @staticmethod
    def _clean_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, str | int | float | bool]:
        """Keep only metadata types supported by ChromaDB."""

        cleaned: dict[str, str | int | float | bool] = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value

        return cleaned

    def _upsert_sync(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update vectors in ChromaDB."""

        self._initialize_sync()

        if not (
            len(ids)
            == len(documents)
            == len(embeddings)
            == len(metadatas)
        ):
            raise ValueError(
                "IDs, documents, embeddings and metadata "
                "must have equal lengths."
            )

        if not ids:
            return

        if any(not document.strip() for document in documents):
            raise ValueError("Documents must not be empty.")

        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs found in this batch.")

        dimensions = {len(vector) for vector in embeddings}

        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError(
                "All embeddings must have the same non-zero dimension."
            )

        cleaned_metadatas = [
            self._clean_metadata(metadata)
            for metadata in metadatas
        ]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=cleaned_metadatas,
        )

    async def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """Insert or update a batch of embedded documents."""

        await self.initialize()

        await asyncio.to_thread(
            self._upsert_sync,
            list(ids),
            list(documents),
            [list(vector) for vector in embeddings],
            list(metadatas),
        )

    def _search_sync(
        self,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Search the collection using vector similarity."""

        self._initialize_sync()

        if n_results < 1:
            raise ValueError("n_results must be at least 1.")

        collection_count = self._collection.count()

        if collection_count == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection_count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )

        return hits

    async def search(
        self,
        query_embedding: Sequence[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search stored vectors without blocking the event loop."""

        await self.initialize()

        if not query_embedding:
            raise ValueError("Query embedding must not be empty.")

        return await asyncio.to_thread(
            self._search_sync,
            list(query_embedding),
            n_results,
            where,
        )

    def _count_sync(self) -> int:
        self._initialize_sync()
        return self._collection.count()

    async def count(self) -> int:
        """Return the number of stored chunks."""

        await self.initialize()
        return await asyncio.to_thread(self._count_sync)

    def _delete_collection_sync(self) -> None:
        """Delete the collection and its indexed records."""

        self._initialize_sync()
        self._client.delete_collection(self.collection_name)
        self._collection = None

    async def delete_collection(self) -> None:
        """Delete the collection asynchronously."""

        await self.initialize()
        await asyncio.to_thread(self._delete_collection_sync)
