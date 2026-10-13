
from __future__ import annotations

from typing import Any

import pytest

from orchestration.graph import create_rag_graph
from orchestration.nodes import RAGNodes


TEST_DOCUMENT = {
    "chunk_id": "test-chunk-1",
    "text": (
        "The Ministry of External Affairs publishes an annual report "
        "covering India's foreign relations and diplomatic activities."
    ),
    "source": "mea_annual_report.pdf",
    "page_number": 1,
    "metadata": {
        "source": "mea_annual_report.pdf",
        "page_number": 1,
    },
}


class FakeRetrievalService:
    """Simulates hybrid retrieval without loading real models."""

    def __init__(
        self,
        documents: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ):
        self.documents = (
            [dict(TEST_DOCUMENT)] if documents is None else documents
        )
        self.error = error
        self.calls: list[str] = []

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(query)

        if self.error:
            raise self.error

        return {
            "query": query,
            "results": self.documents[:top_k] if top_k else self.documents,
            "candidate_count": len(self.documents),
            "result_count": len(self.documents),
            "retrieval_pipeline": [
                "vector_search",
                "bm25_search",
                "rrf",
                "reranking",
            ],
        }


class FakeAnswerGenerator:
    """Simulates the LLM response."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        question: str,
        retrieved_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "question": question,
                "document_count": len(retrieved_documents),
            }
        )

        if self.error:
            raise self.error

        return {
            "question": question,
            "answer": (
                "The report covers India's foreign relations and "
                "diplomatic activities "
                "[MEA Annual Report 2025, p. 1]."
            ),
            "context_count": len(retrieved_documents),
            "context": "Test context",
            "model": "fake-test-model",
        }


class FakeCitationBuilder:
    """Returns predictable citation metadata."""

    def attach_citations(
        self,
        answer: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        citations = [
            {
                "source": document.get("source")
                or document.get("metadata", {}).get("source"),
                "page_number": document.get("page_number")
                or document.get("metadata", {}).get("page_number"),
                "citation": (
                    "[MEA Annual Report 2025, p. "
                    f"{document.get('page_number') or document.get('metadata', {}).get('page_number')}]"
                ),
            }
            for document in documents
        ]

        return {
            "answer": answer,
            "citations": citations,
        }


class FakeAnswerValidator:
    """Allows tests to control answer validation."""

    def __init__(self, valid: bool = True):
        self.valid = valid

    def validate(
        self,
        answer: str,
        retrieved_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "is_valid": self.valid,
            "issues": [] if self.valid else ["Test validation failure"],
            "citations_found": 1 if self.valid else 0,
            "unknown_citations": [],
            "uncited_sentences": [],
            "citation_count": len(retrieved_documents) if self.valid else 0,
        }


class FakeRefusalHandler:
    """Simulates the evidence/refusal policy."""

    def __init__(self, force_refusal: bool = False):
        self.force_refusal = force_refusal

    def should_refuse(
        self,
        question: str,
        retrieved_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        should_refuse = (
            self.force_refusal
            or not question.strip()
            or not retrieved_documents
        )

        return {
            "should_refuse": should_refuse,
            "reason": (
                "Insufficient evidence" if should_refuse else None
            ),
            "message": (
                "I could not find sufficient evidence in the "
                "MEA Annual Report 2025 to answer that question."
                if should_refuse
                else None
            ),
            "evidence": bool(retrieved_documents),
        }


def make_graph(
    *,
    documents: list[dict[str, Any]] | None = None,
    retrieval_error: Exception | None = None,
    generation_error: Exception | None = None,
    force_refusal: bool = False,
    validation_valid: bool = True,
):
    retrieval = FakeRetrievalService(
        documents=documents,
        error=retrieval_error,
    )
    generator = FakeAnswerGenerator(error=generation_error)

    nodes = RAGNodes(
        retrieval_service=retrieval,
        answer_generator=generator,
        citation_builder=FakeCitationBuilder(),
        answer_validator=FakeAnswerValidator(valid=validation_valid),
        refusal_handler=FakeRefusalHandler(
            force_refusal=force_refusal
        ),
    )

    return create_rag_graph(nodes), retrieval, generator


async def invoke_graph(
    graph,
    *,
    question: str = "What does the report cover?",
    history: list[dict[str, Any]] | None = None,
):
    return await graph.ainvoke(
        {
            "conversation_id": "test-conversation",
            "user_id": "test-user",
            "question": question,
            "conversation_history": history or [],
            "status": "pending",
        }
    )


@pytest.mark.asyncio
async def test_happy_path_completes_full_workflow():
    graph, retrieval, generator = make_graph()

    result = await invoke_graph(graph)

    assert result["status"] == "completed"
    assert result["answer"]
    assert result["citations"]
    assert result["validation_result"]["is_valid"] is True
    assert result["is_valid"] is True
    assert result["assistant_message"]["role"] == "assistant"
    assert result["response"]
    assert len(retrieval.calls) == 1
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_empty_question_is_refused():
    graph, _, generator = make_graph()

    result = await invoke_graph(graph, question="   ")

    assert result["status"] == "refused"
    assert result["should_refuse"] is True
    assert result["response"]
    assert len(generator.calls) == 0


@pytest.mark.asyncio
async def test_no_retrieved_documents_is_refused():
    graph, _, generator = make_graph(documents=[])

    result = await invoke_graph(graph)

    assert result["status"] == "refused"
    assert result["should_refuse"] is True
    assert len(generator.calls) == 0


@pytest.mark.asyncio
async def test_refusal_handler_can_refuse_available_evidence():
    graph, _, generator = make_graph(force_refusal=True)

    result = await invoke_graph(graph)

    assert result["status"] == "refused"
    assert result["should_refuse"] is True
    assert len(generator.calls) == 0


@pytest.mark.asyncio
async def test_retrieval_exception_uses_error_path():
    graph, _, generator = make_graph(
        retrieval_error=RuntimeError("Simulated retrieval failure")
    )

    result = await invoke_graph(graph)

    assert result["status"] == "failed"
    assert result["response"]
    assert len(generator.calls) == 0


@pytest.mark.asyncio
async def test_generation_exception_uses_error_path():
    graph, _, _ = make_graph(
        generation_error=RuntimeError("Simulated LLM failure")
    )

    result = await invoke_graph(graph)

    assert result["status"] == "failed"
    assert result["response"]


@pytest.mark.asyncio
async def test_validation_failure_uses_error_path():
    graph, _, _ = make_graph(validation_valid=False)

    result = await invoke_graph(graph)

    assert result["status"] == "failed"
    assert result["validation_result"]["is_valid"] is False
    assert result["response"]


@pytest.mark.asyncio
async def test_conversation_history_is_passed_to_graph():
    graph, retrieval, generator = make_graph()

    history = [
        {"role": "user", "content": "Tell me about the report."},
        {
            "role": "assistant",
            "content": "It covers foreign relations.",
        },
    ]

    result = await invoke_graph(
        graph,
        question="What topics does it cover?",
        history=history,
    )

    assert result["status"] == "completed"
    assert len(retrieval.calls) == 1
    assert len(generator.calls) == 1

    # The current implementation builds a contextualized question
    # from the recent conversation history.
    assert "What topics does it cover?" in generator.calls[0]["question"]


@pytest.mark.asyncio
async def test_graph_can_process_multiple_independent_turns():
    graph, _, _ = make_graph()

    first = await invoke_graph(
        graph,
        question="What does the report cover?",
    )
    second = await invoke_graph(
        graph,
        question="What is the purpose of the report?",
    )

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert first["conversation_id"] == second["conversation_id"]
    assert first["assistant_message"]["message_id"] != (
        second["assistant_message"]["message_id"]
    )
