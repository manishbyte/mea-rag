
from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any


class DocumentChunker:
    """Create page-aware, overlapping chunks for RAG indexing."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and "
                "smaller than chunk_size."
            )

        if min_chunk_size < 0 or min_chunk_size > chunk_size:
            raise ValueError(
                "min_chunk_size must be between 0 and chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace while retaining paragraph boundaries."""

        text = text.replace("\x00", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _split_text(self, text: str) -> list[str]:
        """Split text near paragraph or sentence boundaries."""

        text = self._clean_text(text)

        if not text:
            return []

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                # Prefer a paragraph boundary, then a sentence,
                # then a word boundary.
                search_start = start + int(self.chunk_size * 0.6)

                paragraph_end = text.rfind("\n\n", search_start, end)
                sentence_end = max(
                    text.rfind(". ", search_start, end),
                    text.rfind("? ", search_start, end),
                    text.rfind("! ", search_start, end),
                )
                word_end = text.rfind(" ", search_start, end)

                if paragraph_end != -1:
                    end = paragraph_end
                elif sentence_end != -1:
                    end = sentence_end + 1
                elif word_end != -1:
                    end = word_end

            chunk = text[start:end].strip()

            if chunk and (
                len(chunk) >= self.min_chunk_size
                or not chunks
            ):
                chunks.append(chunk)

            if end >= text_length:
                break

            # Overlap preserves context between adjacent chunks.
            next_start = max(0, end - self.chunk_overlap)

            # Ensure progress even if boundary selection changes.
            start = next_start if next_start > start else end

        return chunks

    def _chunk_pages_sync(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Chunk pages independently so each chunk has a valid page."""

        all_chunks: list[dict[str, Any]] = []

        for page in pages:
            page_number = page.get("page_number")
            text = page.get("text", "")

            if not isinstance(page_number, int) or page_number < 1:
                raise ValueError(
                    f"Invalid page number: {page_number!r}"
                )

            if not isinstance(text, str):
                raise TypeError(
                    f"Page {page_number}: text must be a string."
                )

            if not text.strip():
                continue

            parts = self._split_text(text)

            for chunk_index, chunk_text in enumerate(parts, start=1):
                document_id = str(page.get("document_id", ""))
                source = str(page.get("source", "unknown"))

                # Deterministic ID for repeatable indexing.
                identity = (
                    f"{document_id}|{source}|"
                    f"{page_number}|{chunk_index}|{chunk_text}"
                )
                chunk_id = hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()

                metadata = {
                    key: value
                    for key, value in page.items()
                    if key not in {"text", "char_count", "is_empty"}
                    and isinstance(value, (str, int, float, bool))
                }

                metadata.update(
                    {
                        "page_number": page_number,
                        "source": source,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                        "chunk_char_count": len(chunk_text),
                    }
                )

                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "metadata": metadata,
                    }
                )

        return all_chunks

    async def chunk_pages(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Chunk pages without blocking the asyncio event loop."""

        return await asyncio.to_thread(
            self._chunk_pages_sync,
            pages,
        )
