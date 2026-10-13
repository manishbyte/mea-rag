
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from indexing.embedding_model import EmbeddingModel
from indexing.chroma_store import ChromaStore
from indexing.index_builder import IndexBuilder


logger = logging.getLogger(__name__)


class IndexManager:
    """Coordinate vector indexing and semantic retrieval."""

    def __init__(
        self,
        chunks_path: str | Path = "data/processed/mea_chunks.json",
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "mea_annual_report",
        batch_size: int = 32,
    ) -> None:
        self.chunks_path = Path(chunks_path)

        self.embedder = EmbeddingModel(batch_size=batch_size)

        self.store = ChromaStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        self.builder = IndexBuilder(
            chunks_path=chunks_path,
            persist_directory=persist_directory,
            collection_name=collection_name,
            batch_size=batch_size,
        )

        self._search_semaphore = asyncio.Semaphore(4)

    async def initialize(self) -> None:
        """Initialize the persistent vector store."""

        await self.store.initialize()

    async def build_index(self) -> dict[str, Any]:
        """Build or update the vector index from processed chunks."""

        logger.info("Starting MEA vector index build.")

        return await self.builder.build()

    async def index_status(self) -> dict[str, Any]:
        """Return the current vector index status."""

        await self.initialize()

        count = await self.store.count()

        return {
            "collection_name": self.store.collection_name,
            "persist_directory": str(
                self.store.persist_directory.resolve()
            ),
            "indexed_chunks": count,
            "chunks_file_exists": self.chunks_path.is_file(),
            "status": "ready" if count > 0 else "empty",
        }

    async def search(
        self,
        query: str,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find relevant chunks using semantic vector similarity."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        async with self._search_semaphore:
            await self.initialize()

            if await self.store.count() == 0:
                raise RuntimeError(
                    "The vector index is empty. "
                    "Run build_index() before searching."
                )

            query_embedding = await self.embedder.embed_query(query)

            results = await self.store.search(
                query_embedding=query_embedding,
                n_results=top_k,
                where=where,
            )

            return [
                {
                    **result,
                    "rank": rank,
                    "source": result["metadata"].get(
                        "source", "unknown"
                    ),
                    "page_number": result["metadata"].get(
                        "page_number"
                    ),
                }
                for rank, result in enumerate(results, start=1)
            ]

    async def close(self) -> None:
        """Release application references to local model and store."""

        # PersistentClient stores data automatically.
        # No explicit commit or close is required for this client.
        self._search_semaphore = asyncio.Semaphore(4)
        logger.info("Index manager is ready to be discarded.")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    manager = IndexManager()

    status = await manager.index_status()
    print("=== Index Status ===")

    for key, value in status.items():
        print(f"{key}: {value}")

    if status["indexed_chunks"] == 0:
        print("\nIndex is empty. Building the index...")
        summary = await manager.build_index()

        for key, value in summary.items():
            print(f"{key}: {value}")

    query = "What are India's major foreign policy priorities?"

    print(f"\n=== Search: {query} ===")

    results = await manager.search(query, top_k=5)

    for result in results:
        print(
            f"\nRank: {result['rank']} | "
            f"Source: {result['source']} | "
            f"Page: {result['page_number']}"
        )
        print(f"Distance: {result['distance']:.4f}")
        print(result["text"][:700])


if __name__ == "__main__":
    asyncio.run(main())
