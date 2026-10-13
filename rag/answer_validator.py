
from __future__ import annotations

import re
from typing import Any, Sequence

from rag.citation_builder import CitationBuilder


class AnswerValidator:
    """Validate answer structure and citation consistency."""

    CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        citation_builder: CitationBuilder | None = None,
        require_citations: bool = True,
        require_citation_per_sentence: bool = False,
    ) -> None:
        self.citation_builder = citation_builder or CitationBuilder()
        self.require_citations = require_citations
        self.require_citation_per_sentence = require_citation_per_sentence

    @staticmethod
    def _normalize_citation(citation: str) -> str:
        """Normalize whitespace for citation comparison."""
        return re.sub(r"\s+", " ", citation.strip()).casefold()

    @staticmethod
    def _extract_sentences(answer: str) -> list[str]:
        """Split prose into sentences while ignoring empty fragments."""
        sentences = []

        for paragraph in answer.splitlines():
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # Ignore standalone Markdown headings.
            if paragraph.startswith("#"):
                continue

            parts = AnswerValidator.SENTENCE_PATTERN.split(paragraph)

            for part in parts:
                sentence = part.strip()

                # Remove common Markdown list prefixes.
                sentence = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", sentence)

                if sentence:
                    sentences.append(sentence)

        return sentences

    def validate(
        self,
        answer: str,
        retrieved_documents: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Validate an answer against citations available from retrieved documents.

        Returns:
            is_valid: Whether the answer passes configured structural checks.
            issues: Human-readable validation issues.
            citations_found: Citation strings found in the answer.
            unknown_citations: Citations not present in retrieved documents.
            uncited_sentences: Sentences without an inline citation.
            citation_count: Number of unique citations found.
        """
        if not isinstance(answer, str):
            raise TypeError("answer must be a string.")

        documents = retrieved_documents or []
        issues: list[str] = []

        if not answer.strip():
            return {
                "is_valid": False,
                "issues": ["The generated answer is empty."],
                "citations_found": [],
                "unknown_citations": [],
                "uncited_sentences": [],
                "citation_count": 0,
            }

        available_citations = self.citation_builder.build_citations(documents)

        allowed = {
            self._normalize_citation(
                citation["citation"].strip("[]")
            )
            for citation in available_citations
        }

        citations_found = self.CITATION_PATTERN.findall(answer)
        normalized_found = [
            self._normalize_citation(citation)
            for citation in citations_found
        ]

        unknown_citations = [
            citation
            for citation in citations_found
            if self._normalize_citation(citation) not in allowed
        ]

        if unknown_citations:
            issues.append(
                "The answer contains citations that do not match "
                "the retrieved documents."
            )

        if self.require_citations and not citations_found:
            issues.append("The answer does not contain any citations.")

        sentences = self._extract_sentences(answer)
        uncited_sentences = [
            sentence
            for sentence in sentences
            if not self.CITATION_PATTERN.search(sentence)
        ]

        if self.require_citation_per_sentence and uncited_sentences:
            issues.append(
                "One or more sentences do not contain an inline citation."
            )

        if self.require_citations and not available_citations:
            issues.append(
                "No citations can be verified because no usable "
                "retrieved-document citations are available."
            )

        return {
            "is_valid": not issues,
            "issues": issues,
            "citations_found": list(dict.fromkeys(citations_found)),
            "unknown_citations": list(dict.fromkeys(unknown_citations)),
            "uncited_sentences": uncited_sentences,
            "citation_count": len(set(normalized_found)),
        }

    def validate_or_raise(
        self,
        answer: str,
        retrieved_documents: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Validate an answer and raise ValueError if it fails."""
        result = self.validate(answer, retrieved_documents)

        if not result["is_valid"]:
            details = "; ".join(result["issues"])
            raise ValueError(f"Answer validation failed: {details}")

        return result
