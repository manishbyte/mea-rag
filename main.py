
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("mea_rag")


def print_section(title: str) -> None:
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def print_json(data: Any) -> None:
    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


async def close_resource(resource: Any) -> None:
    """Close a component if it exposes a sync or async close method."""
    if resource is None:
        return

    close_method = getattr(resource, "aclose", None)

    if not callable(close_method):
        close_method = getattr(resource, "close", None)

    if not callable(close_method):
        return

    try:
        result = close_method()

        if hasattr(result, "__await__"):
            await result

    except Exception:
        logger.exception(
            "Failed to close resource: %s",
            type(resource).__name__,
        )


async def close_resources(resources: list[Any]) -> None:
    """Close application resources in reverse construction order."""
    seen: set[int] = set()

    for resource in reversed(resources):
        if resource is None or id(resource) in seen:
            continue

        seen.add(id(resource))
        await close_resource(resource)


def build_application_sync() -> tuple[Any, list[Any]]:
    """
    Build the complete RAG pipeline.

    Heavy application imports are intentionally performed here instead
    of at module import time. This lets the API load this module without
    immediately importing every embedding, retrieval, and LLM dependency.
    """
    load_dotenv()

    logger.info("Initializing MEA RAG application components.")

    resources: list[Any] = []

    try:
        # 1. Import retrieval modules one at a time.
        logger.info("Importing IndexManager.")
        from indexing.index_manager import IndexManager
        logger.info("IndexManager module imported.")

        logger.info("Importing VectorSearch.")
        from retrieval.vector_search import VectorSearch
        logger.info("VectorSearch module imported.")

        logger.info("Importing BM25Search.")
        from retrieval.bm25_search import BM25Search
        logger.info("BM25Search module imported.")

        logger.info("Importing HybridSearch.")
        from retrieval.hybrid_search import HybridSearch
        logger.info("HybridSearch module imported.")

        logger.info("Importing ReciprocalRankFusion.")
        from retrieval.fusion import ReciprocalRankFusion
        logger.info("ReciprocalRankFusion module imported.")

        logger.info("Importing Reranker.")
        from retrieval.reranker import Reranker
        logger.info("Reranker module imported.")

        logger.info("Importing RetrievalService.")
        from retrieval.retrieval_service import RetrievalService
        logger.info("RetrievalService module imported.")

        # 2. Import LLM and answer-generation modules.
        logger.info("Importing LLMModels.")
        from llm.models import LLMModels
        logger.info("LLMModels module imported.")

        logger.info("Importing AnswerGenerator.")
        from rag.answer_generator import AnswerGenerator
        logger.info("AnswerGenerator module imported.")

        logger.info("Importing CitationBuilder.")
        from rag.citation_builder import CitationBuilder
        logger.info("CitationBuilder module imported.")

        logger.info("Importing AnswerValidator.")
        from rag.answer_validator import AnswerValidator
        logger.info("AnswerValidator module imported.")

        logger.info("Importing RefusalHandler.")
        from rag.refusal_handler import RefusalHandler
        logger.info("RefusalHandler module imported.")

        # 3. Import orchestration modules.
        logger.info("Importing RAGNodes.")
        from orchestration.nodes import RAGNodes
        logger.info("RAGNodes module imported.")

        logger.info("Importing create_rag_graph.")
        from orchestration.graph import create_rag_graph
        logger.info("Graph module imported.")

        # 4. Construct the retrieval components.
        logger.info("Constructing IndexManager.")
        index_manager = IndexManager()
        resources.append(index_manager)
        logger.info("IndexManager constructed.")

        logger.info("Constructing VectorSearch.")
        vector_search = VectorSearch(index_manager)
        logger.info("VectorSearch constructed.")

        logger.info("Constructing BM25Search.")
        bm25_search = BM25Search()
        resources.append(bm25_search)
        logger.info("BM25Search constructed.")

        logger.info("Constructing HybridSearch.")
        hybrid_search = HybridSearch(
            vector_search=vector_search,
            bm25_search=bm25_search,
        )
        logger.info("HybridSearch constructed.")

        logger.info("Constructing ReciprocalRankFusion.")
        fusion = ReciprocalRankFusion()
        logger.info("ReciprocalRankFusion constructed.")

        logger.info("Constructing Reranker.")
        reranker = Reranker()
        resources.append(reranker)
        logger.info("Reranker constructed.")

        logger.info("Constructing RetrievalService.")
        retrieval_service = RetrievalService(
            hybrid_search=hybrid_search,
            fusion=fusion,
            reranker=reranker,
        )
        logger.info("RetrievalService constructed.")

        # 5. Construct LLM and RAG components.
        logger.info("Constructing LLMModels.")
        llm = LLMModels()
        resources.append(llm)
        logger.info("LLMModels constructed.")

        logger.info("Constructing AnswerGenerator.")
        answer_generator = AnswerGenerator(llm=llm)
        resources.append(answer_generator)
        logger.info("AnswerGenerator constructed.")

        logger.info("Constructing CitationBuilder.")
        citation_builder = CitationBuilder()
        logger.info("CitationBuilder constructed.")

        logger.info("Constructing AnswerValidator.")
        answer_validator = AnswerValidator()
        logger.info("AnswerValidator constructed.")

        logger.info("Constructing RefusalHandler.")
        refusal_handler = RefusalHandler()
        logger.info("RefusalHandler constructed.")

        # 6. Build the graph.
        logger.info("Constructing RAGNodes.")
        nodes = RAGNodes(
            retrieval_service=retrieval_service,
            answer_generator=answer_generator,
            citation_builder=citation_builder,
            answer_validator=answer_validator,
            refusal_handler=refusal_handler,
        )
        logger.info("RAGNodes constructed.")

        logger.info("Building LangGraph workflow.")
        graph = create_rag_graph(nodes)
        logger.info("LangGraph workflow constructed.")

        logger.info("MEA RAG application initialized successfully.")

        return graph, resources

    except Exception:
        logger.exception("MEA RAG application construction failed.")

        # Clean up resources created before the failure.
        for resource in reversed(resources):
            try:
                close_method = getattr(resource, "close", None)

                if not callable(close_method):
                    close_method = getattr(resource, "aclose", None)

                if callable(close_method):
                    result = close_method()

                    # Synchronous builder cannot await async cleanup.
                    # Async cleanup is deferred to the caller only if
                    # the component supports a synchronous close method.
                    if hasattr(result, "__await__"):
                        logger.warning(
                            "Async cleanup required for %s after "
                            "initialization failure.",
                            type(resource).__name__,
                        )

            except Exception:
                logger.exception(
                    "Resource cleanup failed for %s.",
                    type(resource).__name__,
                )

        raise


async def build_application() -> tuple[Any, list[Any]]:
    """Build the application without blocking the current event loop."""
    return await asyncio.to_thread(build_application_sync)


async def run_query(
    graph: Any,
    question: str,
    conversation_id: str | None = None,
    user_id: str = "local-test-user",
    conversation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a question through the RAG workflow."""
    initial_state = {
        "conversation_id": conversation_id or str(uuid.uuid4()),
        "user_id": user_id,
        "question": question,
        "conversation_history": conversation_history or [],
        "status": "pending",
    }

    if hasattr(graph, "ainvoke"):
        result = await graph.ainvoke(initial_state)
    elif hasattr(graph, "invoke"):
        result = await asyncio.to_thread(
            graph.invoke,
            initial_state,
        )
    else:
        raise TypeError(
            "The graph must expose ainvoke() or invoke()."
        )

    if not isinstance(result, dict):
        raise TypeError(
            f"Expected graph output to be a dictionary, "
            f"got {type(result).__name__}."
        )

    return result


def display_response(result: dict[str, Any]) -> None:
    """Display the full chatbot response."""
    print_section("FINAL ANSWER")
    print(result.get("answer") or "No answer was returned.")

    print_section("WORKFLOW STATUS")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Model: {result.get('model') or 'Not reported'}")
    print(f"Context chunks: {result.get('context_count', 0)}")

    if result.get("should_refuse"):
        print(f"Refusal reason: {result.get('refusal_reason')}")

    print_section("CITATIONS")
    print_json(result.get("citations") or [])

    print_section("ANSWER VALIDATION")
    print_json(result.get("validation_result") or {})

    print_section("RETRIEVAL METADATA")
    print_json(result.get("retrieval_metadata") or {})

    print_section("RETRIEVED DOCUMENTS")
    documents = result.get("retrieved_documents") or []

    if not documents:
        print("No documents were retrieved.")

    for number, document in enumerate(documents, start=1):
        metadata = document.get("metadata") or {}

        source = (
            document.get("source")
            or metadata.get("source")
            or "Unknown"
        )
        page = (
            document.get("page_number")
            or metadata.get("page_number")
            or "Unknown"
        )
        document_text = document.get("text") or ""

        print(f"\nDocument {number}")
        print(f"Chunk ID: {document.get('chunk_id', 'Unknown')}")
        print(f"Source: {source}")
        print(f"Page: {page}")
        print(
            "Score:",
            document.get(
                "rerank_score",
                document.get("distance", "N/A"),
            ),
        )
        print(f"Text: {document_text[:700]}")

        if len(document_text) > 700:
            print("... [text truncated for display]")

    if result.get("error"):
        print_section("ERROR DETAILS")
        print(result["error"])

    print_section("RESPONSE OBJECT")
    print_json(result.get("response") or {})


async def main() -> None:
    """Run the interactive chatbot."""
    resources: list[Any] = []
    conversation_id = str(uuid.uuid4())
    user_id = "local-test-user"
    conversation_history: list[dict[str, Any]] = []

    try:
        graph, resources = await build_application()

        print_section("MEA ANNUAL REPORT 2025 CHATBOT")
        print("Ask a question about the report.")
        print("Commands: /history, /new, /quit")
        print(f"Conversation ID: {conversation_id}")

        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chatbot...")
                break

            if not question:
                print("Please enter a question.")
                continue

            command = question.lower()

            if command in {"/quit", "/exit"}:
                print("Goodbye.")
                break

            if command == "/history":
                print_section("CURRENT SESSION HISTORY")
                print_json(conversation_history)
                continue

            if command == "/new":
                conversation_id = str(uuid.uuid4())
                conversation_history = []
                print(f"Started a new conversation: {conversation_id}")
                continue

            started_at = datetime.now(timezone.utc)

            try:
                result = await run_query(
                    graph=graph,
                    question=question,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    conversation_history=conversation_history,
                )

                display_response(result)

                finished_at = datetime.now(timezone.utc)

                print_section("EXECUTION SUMMARY")
                print(
                    "Duration:",
                    f"{(finished_at - started_at).total_seconds():.2f} seconds",
                )
                print(f"Status: {result.get('status', 'unknown')}")

                conversation_history.append({
                    "message_id": str(uuid.uuid4()),
                    "role": "user",
                    "content": question,
                    "created_at": started_at.isoformat(),
                })

                assistant_message = result.get("assistant_message")

                if assistant_message:
                    conversation_history.append(
                        dict(assistant_message)
                    )
                elif result.get("answer"):
                    conversation_history.append({
                        "message_id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": result["answer"],
                        "created_at": finished_at.isoformat(),
                        "citations": result.get("citations", []),
                    })

            except Exception:
                logger.exception("Query execution failed")
                print("The query failed. See the logs for details.")

    except Exception:
        logger.exception("Application initialization failed")
        raise SystemExit(1)

    finally:
        await close_resources(resources)


if __name__ == "__main__":
    asyncio.run(main())
