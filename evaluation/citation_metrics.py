
"""Citation coverage and page-matching metrics for RAG answers."""

from __future__ import annotations

import re
from typing import Any


_PAGE_PATTERNS = [
    re.compile(
        r"\[\s*(?:page|p\.?)\s*(\d+)\s*\]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\(\s*(?:page|p\.?)\s*(\d+)\s*\)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpage\s+(\d+)\b",
        re.IGNORECASE,
    ),
]


def _page_number(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_cited_pages(answer: str) -> list[int]:
    pages: list[int] = []

    for pattern in _PAGE_PATTERNS:
        pages.extend(
            int(match.group(1))
            for match in pattern.finditer(answer)
        )

    # Keep first occurrence order.
    return list(dict.fromkeys(pages))


def _get_document_pages(
    documents: list[dict[str, Any]] | None,
) -> list[int]:
    pages: list[int] = []

    for document in documents or []:
        metadata = document.get("metadata", {}) or {}

        candidates = [
            metadata.get("page_number"),
            metadata.get("page"),
            metadata.get("page_number_start"),
            document.get("page_number"),
            document.get("page"),
        ]

        for candidate in candidates:
            page = _page_number(candidate)

            if page is not None:
                pages.append(page)
                break

    return list(dict.fromkeys(pages))


def _split_sentences(answer: str) -> list[str]:
    """A lightweight sentence splitter for citation coverage diagnostics."""
    if not answer.strip():
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
        if sentence.strip()
    ]


def _sentence_has_citation(sentence: str) -> bool:
    return bool(_extract_cited_pages(sentence))


class CitationMetrics:
    """Calculate structural citation metrics; not semantic proof of support."""

    @classmethod
    def evaluate(
        cls,
        answer: str,
        retrieved_documents: list[dict[str, Any]] | None = None,
        reference_pages: list[int] | None = None,
    ) -> dict[str, Any]:
        cited_pages = _extract_cited_pages(answer)
        retrieved_pages = _get_document_pages(retrieved_documents)

        references: list[int] = []

        for value in reference_pages or []:
            page = _page_number(value)
            if page is not None:
                references.append(page)

        cited_set = set(cited_pages)
        retrieved_set = set(retrieved_pages)
        reference_set = set(references)

        supported_pages = sorted(cited_set & retrieved_set)
        unsupported_pages = sorted(cited_set - retrieved_set)

        if cited_set:
            citation_precision = (
                len(cited_set & retrieved_set) / len(cited_set)
                if retrieved_set
                else None
            )
        else:
            citation_precision = 0.0

        citation_recall = (
            len(cited_set & reference_set) / len(reference_set)
            if reference_set
            else None
        )

        sentences = _split_sentences(answer)
        factual_sentences = [
            sentence
            for sentence in sentences
            if re.search(r"\w", sentence)
        ]
        cited_sentences = [
            sentence
            for sentence in factual_sentences
            if _sentence_has_citation(sentence)
        ]

        citation_coverage = (
            len(cited_sentences) / len(factual_sentences)
            if factual_sentences
            else None
        )

        return {
            "has_citations": bool(cited_pages),
            "citation_count": len(cited_pages),
            "cited_pages": cited_pages,
            "retrieved_pages": retrieved_pages,
            "reference_pages": sorted(reference_set),
            "supported_pages": supported_pages,
            "unsupported_pages": unsupported_pages,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "citation_coverage": citation_coverage,
            "sentence_count": len(factual_sentences),
            "sentences_with_citations": len(cited_sentences),
            "sentences_without_citations": (
                len(factual_sentences) - len(cited_sentences)
            ),
            "reference_pages_available": bool(reference_set),
            "retrieved_pages_available": bool(retrieved_set),
            "note": (
                "Page matching is structural only. It does not verify "
                "whether a cited page semantically supports the claim."
            ),
        }
