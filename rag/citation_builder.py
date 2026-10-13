
from __future__ import annotations

import re
from typing import Any, Sequence


class CitationBuilder:
    """Build source citations from retrieved document metadata."""

    def __init__(
        self,
        default_source_title: str = "MEA Annual Report 2025",
    ) -> None:
        if not default_source_title.strip():
            raise ValueError("default_source_title must not be empty.")

        self.default_source_title = default_source_title.strip()

    @staticmethod
    def _get_metadata(
        document: dict[str, Any],
    ) -> dict[str, Any]:
        """Return document metadata, if available."""

        metadata = document.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    def _get_source_title(
        self,
        document: dict[str, Any],
    ) -> str:
        """Get a readable source title without inventing metadata."""

        metadata = self._get_metadata(document)

        source = document.get("source") or metadata.get("source")

        if not isinstance(source, str) or not source.strip():
            return self.default_source_title

        source = source.strip()

        # Convert a filename such as mea_annual_report.pdf
        # into a readable title when it matches the expected report.
        filename = source.replace("\\", "/").rsplit("/", 1)[-1]
        normalized = re.sub(r"[_-]+", " ", filename)
        normalized = re.sub(r"\.pdf$", "", normalized, flags=re.IGNORECASE)
        normalized = normalized.strip()

        if normalized.lower() in {
            "mea annual report",
            "mea annual report 2025",
        }:
            return self.default_source_title

        return normalized or self.default_source_title

    def _get_page_number(
        self,
        document: dict[str, Any],
    ) -> int | None:
        """Return a valid positive page number, if available."""

        metadata = self._get_metadata(document)

        page = document.get("page_number")
        if page is None:
            page = metadata.get("page_number")

        if isinstance(page, bool):
            return None

        try:
            page_number = int(page)
        except (TypeError, ValueError):
            return None

        return page_number if page_number > 0 else None

    def build_citation(
        self,
        document: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build one citation from a retrieved chunk."""

        if not isinstance(document, dict):
            return None

        source_title = self._get_source_title(document)
        page_number = self._get_page_number(document)

        if page_number is None:
            # Do not fabricate a page number.
            citation_text = f"[{source_title}]"
        else:
            citation_text = f"[{source_title}, p. {page_number}]"

        return {
            "citation": citation_text,
            "source": source_title,
            "page_number": page_number,
            "chunk_id": document.get("chunk_id"),
            "text": document.get("text", ""),
        }

    def build_citations(
        self,
        documents: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build unique citations, preserving retrieval order."""

        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, int | None]] = set()

        for document in documents:
            citation = self.build_citation(document)

            if citation is None:
                continue

            key = (
                citation["source"],
                citation["page_number"],
            )

            if key in seen:
                continue

            seen.add(key)
            citations.append(citation)

        return citations

    def attach_citations(
        self,
        answer: str,
        documents: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Return the answer with citation metadata.

        This method does not claim that every sentence is supported.
        Sentence-level verification belongs in answer_validator.py.
        """

        if not isinstance(answer, str):
            raise TypeError("answer must be a string.")

        citations = self.build_citations(documents)

        return {
            "answer": answer.strip(),
            "citations": [
                {
                    "citation": item["citation"],
                    "source": item["source"],
                    "page_number": item["page_number"],
                    "chunk_id": item["chunk_id"],
                }
                for item in citations
            ],
        }
