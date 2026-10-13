
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orchestration.state import ChatMessage, RAGState
from rag.answer_generator import AnswerGenerator
from rag.answer_validator import AnswerValidator
from rag.citation_builder import CitationBuilder
from rag.refusal_handler import RefusalHandler
from retrieval.retrieval_service import RetrievalService


class RAGNodes:
    """Node implementations for the PostgreSQL-backed MEA RAG chatbot.

    Dependencies are injected so the graph can reuse the existing
    retrieval and RAG services without duplicating their configuration.

    Conversation history is loaded from PostgreSQL by the application
    layer and passed into the graph state.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        answer_generator: AnswerGenerator,
        citation_builder: CitationBuilder | None = None,
        answer_validator: AnswerValidator | None = None,
        refusal_handler: RefusalHandler | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator
        self.citation_builder = citation_builder or CitationBuilder()
        self.answer_validator = answer_validator or AnswerValidator(
            citation_builder=self.citation_builder
        )
        self.refusal_handler = refusal_handler or RefusalHandler()

    @staticmethod
    def _now() -> str:
        """Return a timezone-aware UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _build_contextual_question(state: RAGState) -> str:
        """Include recent chat history to clarify follow-up questions.

        This is a lightweight history-aware approach. The original
        question remains unchanged in the state for the UI and database.
        """
        question = state.get("question", "").strip()
        history = state.get("conversation_history", [])

        if not history:
            return question

        recent_messages = history[-6:]
        history_lines: list[str] = []

        for message in recent_messages:
            role = message.get("role", "")
            content = message.get("content", "").strip()

            if role in {"user", "assistant"} and content:
                history_lines.append(f"{role.capitalize()}: {content}")

        if not history_lines:
            return question

        return (
            "Use the following previous conversation only to understand "
            "the user's references and follow-up question.\n\n"
            "Previous conversation:\n"
            + "\n".join(history_lines)
            + "\n\nCurrent question:\n"
            + question
        )

    async def retrieve_node(self, state: RAGState) -> dict[str, Any]:
        """Retrieve relevant evidence using the existing retrieval pipeline."""
        try:
            question = state.get("question", "").strip()

            if not question:
                return {
                    "retrieved_documents": [],
                    "retrieval_metadata": {},
                    "status": "refused",
                    "should_refuse": True,
                    "refusal_reason": "empty_question",
                    "refusal_message": self.refusal_handler.INVALID_QUESTION_MESSAGE,
                }

            contextual_question = self._build_contextual_question(state)

            retrieval_result = await self.retrieval_service.retrieve(
                query=contextual_question
            )

            documents = retrieval_result.get("results", [])

            return {
                "retrieved_documents": documents,
                "retrieval_metadata": {
                    "retrieval_pipeline": retrieval_result.get(
                        "retrieval_pipeline", []
                    ),
                    "vector_result_count": retrieval_result.get(
                        "vector_result_count"
                    ),
                    "bm25_result_count": retrieval_result.get(
                        "bm25_result_count"
                    ),
                    "candidate_count": retrieval_result.get(
                        "candidate_count"
                    ),
                    "result_count": len(documents),
                    "contextualized_query": contextual_question,
                },
                "status": "processing",
                "error": None,
            }

        except Exception as exc:
            return {
                "retrieved_documents": [],
                "retrieval_metadata": {},
                "status": "failed",
                "error": f"Retrieval failed: {exc}",
            }

    async def refusal_node(self, state: RAGState) -> dict[str, Any]:
        """Check whether the question has enough retrieved evidence."""
        try:
            result = self.refusal_handler.should_refuse(
                question=state.get("question", ""),
                retrieved_documents=state.get("retrieved_documents", []),
            )

            return {
                "should_refuse": result["should_refuse"],
                "refusal_reason": result.get("reason"),
                "refusal_message": result.get("message"),
                "status": (
                    "refused"
                    if result["should_refuse"]
                    else "processing"
                ),
            }

        except Exception as exc:
            return {
                "should_refuse": True,
                "refusal_reason": "evidence_check_failed",
                "refusal_message": self.refusal_handler.DEFAULT_REFUSAL_MESSAGE,
                "status": "failed",
                "error": f"Evidence check failed: {exc}",
            }

    async def generate_node(self, state: RAGState) -> dict[str, Any]:
        """Generate an answer grounded in the retrieved documents."""
        try:
            contextual_question = self._build_contextual_question(state)
            documents = state.get("retrieved_documents", [])

            result = await self.answer_generator.generate(
                question=contextual_question,
                retrieved_documents=documents,
            )

            answer = result.get("answer", "").strip()

            if not answer:
                return {
                    "answer": "",
                    "model": result.get("model"),
                    "context_count": result.get("context_count", 0),
                    "status": "failed",
                    "error": "The answer generator returned an empty answer.",
                }

            return {
                "answer": answer,
                "model": result.get("model"),
                "context_count": result.get("context_count", len(documents)),
                "status": "processing",
                "error": None,
            }

        except Exception as exc:
            return {
                "answer": "",
                "status": "failed",
                "error": f"Answer generation failed: {exc}",
            }

    async def citation_node(self, state: RAGState) -> dict[str, Any]:
        """Build citations from the documents used for answer generation."""
        try:
            result = self.citation_builder.attach_citations(
                answer=state.get("answer", ""),
                documents=state.get("retrieved_documents", []),
            )

            return {
                "answer": result["answer"],
                "citations": result["citations"],
                "status": "processing",
            }

        except Exception as exc:
            return {
                "citations": [],
                "status": "failed",
                "error": f"Citation building failed: {exc}",
            }

    async def validation_node(self, state: RAGState) -> dict[str, Any]:
        """Validate answer structure and citation consistency."""
        try:
            result = self.answer_validator.validate(
                answer=state.get("answer", ""),
                retrieved_documents=state.get("retrieved_documents", []),
            )

            return {
                "validation_result": result,
                "is_valid": result["is_valid"],
                "validation_issues": result["issues"],
                "status": (
                    "completed"
                    if result["is_valid"]
                    else "failed"
                ),
                "error": (
                    None
                    if result["is_valid"]
                    else "Answer validation failed: "
                    + "; ".join(result["issues"])
                ),
            }

        except Exception as exc:
            return {
                "validation_result": {
                    "is_valid": False,
                    "issues": [str(exc)],
                },
                "is_valid": False,
                "validation_issues": [str(exc)],
                "status": "failed",
                "error": f"Answer validation failed: {exc}",
            }

    async def refusal_response_node(
        self, state: RAGState
    ) -> dict[str, Any]:
        """Prepare the final response when the request is refused."""
        reason = state.get("refusal_reason") or "insufficient_evidence"
        message = (
            state.get("refusal_message")
            or self.refusal_handler.DEFAULT_REFUSAL_MESSAGE
        )

        refusal = self.refusal_handler.build_refusal(
            reason=reason,
            message=message,
        )

        assistant_message: ChatMessage = {
            "message_id": self._new_message_id(),
            "role": "assistant",
            "content": refusal["answer"],
            "created_at": self._now(),
            "citations": [],
            "metadata": {
                "refused": True,
                "reason": reason,
            },
        }

        response = {
            "conversation_id": state.get("conversation_id"),
            "question": state.get("question", ""),
            "answer": refusal["answer"],
            "citations": [],
            "refused": True,
            "reason": reason,
            "status": "refused",
        }

        return {
            "answer": refusal["answer"],
            "citations": [],
            "assistant_message": assistant_message,
            "response": response,
            "status": "refused",
            "error": None,
        }

    async def finalize_node(self, state: RAGState) -> dict[str, Any]:
        """Build the final successful chatbot response."""
        answer = state.get("answer", "").strip()
        citations = state.get("citations", [])

        assistant_message: ChatMessage = {
            "message_id": self._new_message_id(),
            "role": "assistant",
            "content": answer,
            "created_at": self._now(),
            "citations": citations,
            "metadata": {
                "model": state.get("model"),
                "context_count": state.get("context_count", 0),
                "is_valid": state.get("is_valid", False),
            },
        }

        response = {
            "conversation_id": state.get("conversation_id"),
            "question": state.get("question", ""),
            "answer": answer,
            "citations": citations,
            "refused": False,
            "status": "completed",
            "validation": state.get("validation_result", {}),
        }

        return {
            "assistant_message": assistant_message,
            "response": response,
            "status": "completed",
            "error": None,
        }

    async def error_response_node(
        self, state: RAGState
    ) -> dict[str, Any]:
        """Prepare a safe response when a workflow node fails."""
        error = state.get("error") or "The request could not be completed."

        response = {
            "conversation_id": state.get("conversation_id"),
            "question": state.get("question", ""),
            "answer": (
                "Sorry, I couldn't complete your request because "
                "an internal processing step failed. Please try again."
            ),
            "citations": [],
            "refused": False,
            "status": "failed",
            "error": error,
        }

        return {
            "response": response,
            "status": "failed",
        }

    @staticmethod
    def _new_message_id() -> str:
        """Generate a unique message identifier."""
        from uuid import uuid4

        return str(uuid4())
