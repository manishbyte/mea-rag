
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from orchestration.nodes import RAGNodes
from orchestration.routing import (
    route_after_citations,
    route_after_generation,
    route_after_refusal_check,
    route_after_retrieval,
    route_after_validation,
)
from orchestration.state import RAGState


def create_rag_graph(nodes: RAGNodes):
    """Build and compile the MEA RAG chatbot workflow.

    Args:
        nodes: An initialized RAGNodes instance with the retrieval,
            generation, citation, validation, and refusal services.

    Returns:
        A compiled LangGraph application ready for asynchronous
        invocation with RAGState input.
    """
    workflow = StateGraph(RAGState)

    # --------------------------------------------------
    # Register workflow nodes
    # --------------------------------------------------
    workflow.add_node("retrieve", nodes.retrieve_node)
    workflow.add_node("check_evidence", nodes.refusal_node)
    workflow.add_node("generate_answer", nodes.generate_node)
    workflow.add_node("build_citations", nodes.citation_node)
    workflow.add_node("validate_answer", nodes.validation_node)

    workflow.add_node(
        "refusal_response",
        nodes.refusal_response_node,
    )
    workflow.add_node(
        "finalize_response",
        nodes.finalize_node,
    )
    workflow.add_node(
        "error_response",
        nodes.error_response_node,
    )

    # --------------------------------------------------
    # Entry point
    # --------------------------------------------------
    workflow.add_edge(START, "retrieve")

    # --------------------------------------------------
    # Retrieval -> refusal check or error handling
    # --------------------------------------------------
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "refuse": "refusal_response",
            "continue": "check_evidence",
            "error": "error_response",
        },
    )

    # --------------------------------------------------
    # Evidence check -> generation or refusal
    # --------------------------------------------------
    workflow.add_conditional_edges(
        "check_evidence",
        route_after_refusal_check,
        {
            "refuse": "refusal_response",
            "continue": "generate_answer",
            "error": "error_response",
        },
    )

    # --------------------------------------------------
    # Generation -> citation building
    # --------------------------------------------------
    workflow.add_conditional_edges(
        "generate_answer",
        route_after_generation,
        {
            "continue": "build_citations",
            "error": "error_response",
        },
    )

    # --------------------------------------------------
    # Citation building -> answer validation
    # --------------------------------------------------
    workflow.add_conditional_edges(
        "build_citations",
        route_after_citations,
        {
            "continue": "validate_answer",
            "error": "error_response",
        },
    )

    # --------------------------------------------------
    # Validation -> final response or error handling
    # --------------------------------------------------
    workflow.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {
            "finalize": "finalize_response",
            "error": "error_response",
        },
    )

    # --------------------------------------------------
    # Terminal nodes
    # --------------------------------------------------
    workflow.add_edge("refusal_response", END)
    workflow.add_edge("finalize_response", END)
    workflow.add_edge("error_response", END)

    return workflow.compile()
