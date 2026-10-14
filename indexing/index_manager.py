
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from indexing.embedding_model import EmbeddingModel
    from indexing.chroma_store import ChromaStore
    from indexing.index_builder import IndexBuilder


logger = logging.getLogger(__name__)


class IndexManager:
    """Manage embedding, indexing, and vector search operations."""

    def __init__(
        self,
        chunks_path: str | Path = "data/processed/mea_chunks.json",
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "mea_annual_report",
        batch_size: int = 32,
    ) -> None:
        # Defer heavy ML and vector-store imports until construction.
        from indexing.embedding_model import EmbeddingModel
        from indexing.chroma_store import ChromaStore
        from indexing.index_builder import IndexBuilder

        self.chunks_path = Path(chunks_path)

        self.embedder: EmbeddingModel = EmbeddingModel(
            batch_size=batch_size,
        )

        self.store: ChromaStore = ChromaStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        self.builder: IndexBuilder = IndexBuilder(
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
        """Build the vector index from processed document chunks."""

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
        """Search indexed document chunks using query embeddings."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty.")

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

            formatted_results: list[dict[str, Any]] = []

            for rank, result in enumerate(results, start=1):
                metadata = result.get("metadata") or {}

                formatted_results.append(
                    {
                        "rank": rank,
                        "chunk_id": result.get("chunk_id"),
                        "text": result.get("text", ""),
                        "metadata": metadata,
                        "distance": result.get("distance"),
                        "source": metadata.get("source"),
                        "page": metadata.get("page"),
                    }
                )

            return formatted_results

    async def close(self) -> None:
        """Release manager-level resources."""

        # Chroma PersistentClient persists data automatically.
        # There is no explicit commit required.
        logger.info("Index manager is ready to be discarded.")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    manager = IndexManager()

    try:
        status = await manager.index_status()

        print("\n=== MEA Vector Index Status ===")

        for key, value in status.items():
            print(f"{key}: {value}")

        if status["indexed_chunks"] == 0:
            print("\nVector index is empty. Building index...")
            summary = await manager.build_index()

            print("\n=== MEA Vector Index Summary ===")

            for key, value in summary.items():
                print(f"{key}: {value}")

        query = "What are India's major foreign policy priorities?"
        results = await manager.search(query, top_k=5)

        print(f"\n=== Search Results: {query} ===")

        for result in results:
            print(f"\nRank: {result['rank']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Source: {result['source']}")
            print(f"Page: {result['page']}")
            print(f"Distance: {result['distance']}")
            print(f"Text: {result['text'][:1000]}")

    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
