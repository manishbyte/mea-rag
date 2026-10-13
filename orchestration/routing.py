
from __future__ import annotations

from typing import Literal

from orchestration.state import RAGState


RetrievalRoute = Literal["refuse", "continue", "error"]
RefusalRoute = Literal["refuse", "continue", "error"]
ValidationRoute = Literal["finalize", "error"]


def route_after_retrieval(state: RAGState) -> RetrievalRoute:
    """Choose the next node after retrieval."""
    status = state.get("status")

    if status == "failed":
        return "error"

    if state.get("should_refuse", False):
        return "refuse"

    if not state.get("question", "").strip():
        return "refuse"

    if not state.get("retrieved_documents", []):
        return "refuse"

    return "continue"


def route_after_refusal_check(state: RAGState) -> RefusalRoute:
    """Choose whether to generate an answer or refuse."""
    if state.get("status") == "failed":
        return "error"

    if state.get("should_refuse", False):
        return "refuse"

    return "continue"


def route_after_generation(state: RAGState) -> Literal["continue", "error"]:
    """Route empty answers or failed generation to error handling."""
    if state.get("status") == "failed":
        return "error"

    if not state.get("answer", "").strip():
        return "error"

    return "continue"


def route_after_citations(state: RAGState) -> Literal["continue", "error"]:
    """Check that citation construction succeeded before validation."""
    if state.get("status") == "failed":
        return "error"

    if not state.get("answer", "").strip():
        return "error"

    if "citations" not in state:
        return "error"

    return "continue"


def route_after_validation(state: RAGState) -> ValidationRoute:
    """Only allow structurally valid answers into the success path."""
    if state.get("status") == "failed":
        return "error"

    if state.get("is_valid", False):
        return "finalize"

    return "error"
