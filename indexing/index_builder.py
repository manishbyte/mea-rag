
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from indexing.embedding_model import EmbeddingModel
    from indexing.chroma_store import ChromaStore


logger = logging.getLogger(__name__)


class IndexBuilder:
    """Build a persistent vector index from processed document chunks."""

    def __init__(
        self,
        chunks_path: str | Path = "data/processed/mea_chunks.json",
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "mea_annual_report",
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        # Defer heavy ML and vector-store imports until construction.
        from indexing.embedding_model import EmbeddingModel
        from indexing.chroma_store import ChromaStore

        self.chunks_path = Path(chunks_path)
        self.batch_size = batch_size

        self.embedder: EmbeddingModel = EmbeddingModel(
            batch_size=batch_size,
        )

        self.store: ChromaStore = ChromaStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    @staticmethod
    def _load_chunks_sync(
        chunks_path: Path,
    ) -> list[dict[str, Any]]:
        """Load and validate the processed chunk JSON file."""

        if not chunks_path.is_file():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}. "
                "Run the ingestion pipeline first."
            )

        with chunks_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "Expected the chunks JSON file to contain a list."
            )

        validated_chunks: list[dict[str, Any]] = []

        for index, chunk in enumerate(data):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk {index} must be an object.")

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            metadata = chunk.get("metadata", {})

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"Chunk {index} has an invalid ID.")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Chunk {chunk_id} has empty or invalid text."
                )

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Chunk {chunk_id} has invalid metadata."
                )

            validated_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                }
            )

        ids = [chunk["chunk_id"] for chunk in validated_chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs found in input.")

        return validated_chunks

    async def load_chunks(self) -> list[dict[str, Any]]:
        """Load chunks without blocking the event loop."""

        return await asyncio.to_thread(
            self._load_chunks_sync,
            self.chunks_path,
        )

    async def build(self) -> dict[str, Any]:
        """Embed chunks in batches and upsert them into ChromaDB."""

        logger.info("Loading processed document chunks.")

        chunks = await self.load_chunks()

        if not chunks:
            raise ValueError(
                "No chunks found. Check the ingestion output."
            )

        logger.info(
            "Loaded %d chunks. Initializing embedding model.",
            len(chunks),
        )

        await self.embedder.load()
        await self.store.initialize()

        total_chunks = len(chunks)
        indexed_chunks = 0

        logger.info("Building index for %d chunks.", total_chunks)

        for start in range(0, total_chunks, self.batch_size):
            batch = chunks[start : start + self.batch_size]

            texts = [chunk["text"] for chunk in batch]
            ids = [chunk["chunk_id"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]

            # Generate vectors in a worker thread through EmbeddingModel.
            embeddings = await self.embedder.embed_texts(texts)

            if len(embeddings) != len(batch):
                raise RuntimeError(
                    "Embedding count does not match batch size."
                )

            await self.store.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            indexed_chunks += len(batch)

            logger.info(
                "Indexed %d/%d chunks.",
                indexed_chunks,
                total_chunks,
            )

        stored_count = await self.store.count()

        summary = {
            "input_chunks": total_chunks,
            "indexed_chunks": indexed_chunks,
            "collection_count": stored_count,
            "collection_name": self.store.collection_name,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": (
                await self.embedder.get_embedding_dimension()
            ),
            "persist_directory": str(
                self.store.persist_directory.resolve()
            ),
            "status": "success",
        }

        logger.info("Index build completed: %s", summary)

        return summary


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    builder = IndexBuilder(batch_size=32)
    summary = await builder.build()

    print("\n=== MEA Vector Index Summary ===")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
