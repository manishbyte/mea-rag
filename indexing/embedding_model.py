
from __future__ import annotations

import asyncio
from typing import Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Generate local embeddings without blocking the event loop."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Model loading is deferred until the first operation.
        self._model: SentenceTransformer | None = None
        self._model_lock = asyncio.Lock()

    def _load_model_sync(self) -> SentenceTransformer:
        """Load the model in a worker thread."""

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        return self._model

    async def load(self) -> None:
        """Load the embedding model asynchronously."""

        async with self._model_lock:
            if self._model is None:
                await asyncio.to_thread(self._load_model_sync)

    def _embed_sync(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings using the synchronous model API."""

        model = self._load_model_sync()

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embeddings.tolist()

    async def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Embed multiple texts asynchronously."""

        cleaned_texts = [
            text.strip()
            for text in texts
            if isinstance(text, str) and text.strip()
        ]

        if not cleaned_texts:
            return []

        await self.load()

        return await asyncio.to_thread(
            self._embed_sync,
            cleaned_texts,
        )

    async def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for one search query."""

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        embeddings = await self.embed_texts([query])

        return embeddings[0]

    @property
    def embedding_dimension(self) -> int:
        """Return the model's output vector dimension."""

        if self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call await load() first."
            )

        return self._model.get_sentence_embedding_dimension()

    async def get_embedding_dimension(self) -> int:
        """Get the embedding dimension after loading the model."""

        await self.load()
        return await asyncio.to_thread(
            lambda: self.embedding_dimension
        )
