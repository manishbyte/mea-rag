
from __future__ import annotations

from typing import Any, Sequence


class RefusalHandler:
    """Handle questions that cannot be answered reliably from retrieved evidence."""

    DEFAULT_REFUSAL_MESSAGE = (
        "I couldn't find sufficient supporting information in the "
        "MEA Annual Report 2025 to answer this question reliably. "
        "Please try rephrasing your question or asking about a topic "
        "covered in the report."
    )

    INVALID_QUESTION_MESSAGE = (
        "Please provide a non-empty question about the MEA Annual Report 2025."
    )

    def __init__(
        self,
        min_documents: int = 1,
        min_text_characters: int = 40,
        refusal_message: str | None = None,
    ) -> None:
        if min_documents < 1:
            raise ValueError("min_documents must be at least 1.")

        if min_text_characters < 1:
            raise ValueError("min_text_characters must be at least 1.")

        if refusal_message is not None and not refusal_message.strip():
            raise ValueError("refusal_message cannot be empty.")

        self.min_documents = min_documents
        self.min_text_characters = min_text_characters
        self.refusal_message = (
            refusal_message.strip()
            if refusal_message is not None
            else self.DEFAULT_REFUSAL_MESSAGE
        )

    @staticmethod
    def _get_document_text(document: dict[str, Any]) -> str:
        text = document.get("text", "")
        return text.strip() if isinstance(text, str) else ""

    def validate_question(self, question: str) -> dict[str, Any]:
        """Check whether the user supplied a usable question."""
        if not isinstance(question, str):
            raise TypeError("question must be a string.")

        normalized_question = question.strip()

        if not normalized_question:
            return {
                "is_valid": False,
                "reason": "empty_question",
                "message": self.INVALID_QUESTION_MESSAGE,
            }

        return {
            "is_valid": True,
            "reason": None,
            "message": None,
        }

    def assess_evidence(
        self,
        retrieved_documents: Sequence[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Determine whether retrieved results contain enough text to proceed.

        This is a basic evidence-presence check, not a semantic relevance
        or factual-support evaluation.
        """
        documents = retrieved_documents or []

        if not isinstance(documents, (list, tuple)):
            raise TypeError("retrieved_documents must be a list or tuple.")

        usable_documents = [
            document
            for document in documents
            if isinstance(document, dict)
            and len(self._get_document_text(document)) >= self.min_text_characters
        ]

        sufficient = len(usable_documents) >= self.min_documents

        return {
            "sufficient_evidence": sufficient,
            "reason": None if sufficient else "insufficient_retrieved_evidence",
            "retrieved_count": len(documents),
            "usable_document_count": len(usable_documents),
            "message": None if sufficient else self.refusal_message,
        }

    def should_refuse(
        self,
        question: str,
        retrieved_documents: Sequence[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Decide whether processing should stop before answer generation."""
        question_result = self.validate_question(question)

        if not question_result["is_valid"]:
            return {
                "should_refuse": True,
                "reason": question_result["reason"],
                "message": question_result["message"],
                "evidence": None,
            }

        evidence_result = self.assess_evidence(retrieved_documents)

        if not evidence_result["sufficient_evidence"]:
            return {
                "should_refuse": True,
                "reason": evidence_result["reason"],
                "message": evidence_result["message"],
                "evidence": evidence_result,
            }

        return {
            "should_refuse": False,
            "reason": None,
            "message": None,
            "evidence": evidence_result,
        }

    def build_refusal(
        self,
        reason: str,
        message: str | None = None,
    ) -> dict[str, Any]:
        """Build a consistent refusal response for the application."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string.")

        if message is not None and not message.strip():
            raise ValueError("message cannot be empty.")

        return {
            "refused": True,
            "reason": reason.strip(),
            "answer": message.strip() if message else self.refusal_message,
            "citations": [],
        }
