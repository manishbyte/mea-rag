
"""Run the MEA RAG evaluation benchmark and save a JSON report.

Run from the project root:
    python -m evaluation.run_evaluation

Optional:
    python -m evaluation.run_evaluation --limit 5
    python -m evaluation.run_evaluation --no-answer-judge
    python -m evaluation.run_evaluation --top-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from evaluation.answer_metrics import AnswerMetrics
from evaluation.citation_metrics import CitationMetrics
from evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    EvaluationQuestion,
    load_dataset,
    validate_dataset,
)
from evaluation.retrieval_metrics import RetrievalMetrics


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_report.json"
)

logger = logging.getLogger("mea_rag_evaluation")

_EVALUATION_RESOURCES: list[Any] = []


async def _get_graph(factory_path: str) -> Any:
    """Initialize the application and return only its graph."""
    module_name, separator, attribute_name = factory_path.partition(":")

    if not separator:
        raise ValueError(
            "Use the factory format 'module:function', "
            "for example 'main:build_application'."
        )

    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)

    application = factory()

    if inspect.isawaitable(application):
        application = await application

    if (
        isinstance(application, tuple)
        and len(application) == 2
    ):
        graph, resources = application
        global _EVALUATION_RESOURCES
        _EVALUATION_RESOURCES = resources
    else:
        graph = application

    if not hasattr(graph, "ainvoke") and not hasattr(graph, "invoke"):
        raise TypeError(
            "build_application() must return a LangGraph-compatible "
            "graph, or a (graph, resources) tuple."
        )

    return graph




def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


def _document_text(document: Any) -> str:
    if isinstance(document, str):
        return document

    if isinstance(document, dict):
        return str(
            document.get("text")
            or document.get("page_content")
            or document.get("content")
            or ""
        )

    if hasattr(document, "page_content"):
        return str(document.page_content)

    return str(document)


def _document_metadata(document: Any) -> dict[str, Any]:
    if isinstance(document, dict):
        metadata = document.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    metadata = getattr(document, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _normalize_documents(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, dict):
        value = value.get("documents", value.get("results", [value]))

    if not isinstance(value, (list, tuple)):
        value = [value]

    documents: list[dict[str, Any]] = []

    for document in value:
        metadata = _document_metadata(document)
        text = _document_text(document)

        if isinstance(document, dict):
            normalized = dict(document)
            normalized["text"] = text
            normalized["metadata"] = metadata
        else:
            normalized = {
                "text": text,
                "metadata": metadata,
            }

        documents.append(normalized)

    return documents


def _find_value(
    result: dict[str, Any],
    names: tuple[str, ...],
) -> Any:
    """Find a value in the graph output, including a common nested state."""
    for name in names:
        value = result.get(name)
        if value is not None:
            return value

    for container_name in ("state", "result", "output", "data"):
        nested = result.get(container_name)

        if isinstance(nested, dict):
            for name in names:
                value = nested.get(name)
                if value is not None:
                    return value

    return None


def _extract_answer(result: dict[str, Any]) -> str:
    value = _find_value(
        result,
        ("answer", "final_answer", "response", "output_text"),
    )

    if isinstance(value, dict):
        value = (
            value.get("answer")
            or value.get("content")
            or value.get("text")
            or value.get("response")
            or ""
        )

    if isinstance(value, list):
        return "\n".join(
            str(item.get("text", item) if isinstance(item, dict) else item)
            for item in value
        )

    return str(value or "")


def _extract_documents(result: dict[str, Any]) -> list[dict[str, Any]]:
    value = _find_value(
        result,
        (
            "retrieved_documents",
            "retrieved_docs",
            "documents",
            "context_documents",
        ),
    )

    if value is None:
        metadata = result.get("retrieval_metadata")
        if isinstance(metadata, dict):
            value = (
                metadata.get("documents")
                or metadata.get("retrieved_documents")
            )

    return _normalize_documents(value)


def _extract_chunk_id(document: dict[str, Any]) -> str | None:
    metadata = document.get("metadata", {}) or {}

    chunk_id = (
        document.get("chunk_id")
        or metadata.get("chunk_id")
        or metadata.get("id")
        or document.get("id")
    )

    return str(chunk_id) if chunk_id is not None else None


def _extract_error_message(result: dict[str, Any]) -> str | None:
    error = _find_value(result, ("error", "error_message"))

    if error:
        return str(error)

    return None


async def _invoke_graph(
    graph: Any,
    question: str,
) -> dict[str, Any]:
    """Call the existing graph without carrying history between test cases."""
    initial_state = {
        "question": question,
        "conversation_history": [],
    }

    if hasattr(graph, "ainvoke"):
        result = await graph.ainvoke(initial_state)
    else:
        result = await asyncio.to_thread(graph.invoke, initial_state)

    if not isinstance(result, dict):
        result = _to_dict(result)

    if not result:
        raise TypeError(
            "Graph output is not a dictionary. "
            "Adapt _invoke_graph() to your graph's return type."
        )

    return result


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate_results(
    rows: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    failed = [row for row in rows if row.get("status") == "error"]

    retrieval_keys = [
        f"recall_at_{k}",
        f"precision_at_{k}",
        f"mrr_at_{k}",
        f"hit_rate_at_{k}",
        f"ndcg_at_{k}",
    ]

    retrieval: dict[str, Any] = {}

    for key in retrieval_keys:
        values = [
            row["retrieval_metrics"].get(key)
            for row in successful
            if isinstance(row.get("retrieval_metrics", {}).get(key),
                          (int, float))
        ]

        retrieval[key] = _average(values)
        retrieval[f"{key}_evaluated_questions"] = len(values)

    answer_metrics: dict[str, Any] = {}

    for key in ("token_f1", "token_jaccard"):
        values = [
            row["answer_metrics"].get(key)
            for row in successful
            if isinstance(row.get("answer_metrics", {}).get(key),
                          (int, float))
        ]

        answer_metrics[key] = _average(values)
        answer_metrics[f"{key}_evaluated_questions"] = len(values)

    judge_scores: dict[str, Any] = {}

    for key in ("correctness", "relevance", "groundedness", "completeness",
                "overall_score"):
        values = []

        for row in successful:
            judge = row["answer_metrics"].get("llm_judge", {})
            if (
                judge.get("available")
                and isinstance(judge.get(key), (int, float))
            ):
                values.append(float(judge[key]))

        judge_scores[key] = _average(values)
        judge_scores[f"{key}_evaluated_questions"] = len(values)

    citation_keys = (
        "citation_precision",
        "citation_recall",
        "citation_coverage",
    )
    citation: dict[str, Any] = {}

    for key in citation_keys:
        values = [
            row["citation_metrics"].get(key)
            for row in successful
            if isinstance(row.get("citation_metrics", {}).get(key),
                          (int, float))
        ]

        citation[key] = _average(values)
        citation[f"{key}_evaluated_questions"] = len(values)

    latencies = [
        float(row["latency_seconds"])
        for row in successful
        if isinstance(row.get("latency_seconds"), (int, float))
    ]

    return {
        "total_questions": len(rows),
        "successful_questions": len(successful),
        "failed_questions": len(failed),
        "success_rate": (
            len(successful) / len(rows) if rows else None
        ),
        "average_latency_seconds": _average(latencies),
        "min_latency_seconds": min(latencies) if latencies else None,
        "max_latency_seconds": max(latencies) if latencies else None,
        "retrieval": retrieval,
        "answer_overlap": answer_metrics,
        "llm_judge": judge_scores,
        "citations": citation,
    }


async def run_evaluation(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
    top_k: int = 5,
    use_answer_judge: bool = False,
    graph_factory: str = "main:build_application",
    fail_fast: bool = False,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    dataset = load_dataset(dataset_path)
    warnings = validate_dataset(dataset)

    for warning in warnings:
        logger.warning("Dataset warning: %s", warning)

    questions = dataset.questions[:limit] if limit else dataset.questions
    graph = await _get_graph(graph_factory)

    rows: list[dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        logger.info(
            "Evaluating %d/%d: %s",
            index,
            len(questions),
            item.question_id,
        )

        started = time.perf_counter()

        try:
            graph_result = await _invoke_graph(graph, item.question)
            latency = time.perf_counter() - started

            answer = _extract_answer(graph_result)
            documents = _extract_documents(graph_result)
            error_message = _extract_error_message(graph_result)

            if error_message:
                raise RuntimeError(error_message)

            retrieved_ids = [
                chunk_id
                for document in documents
                if (chunk_id := _extract_chunk_id(document)) is not None
            ]

            retrieval_metrics = RetrievalMetrics.evaluate_query(
                retrieved_ids,
                item.relevant_chunk_ids,
                k=top_k,
            )

            answer_metrics = await AnswerMetrics.evaluate(
                question=item.question,
                answer=answer,
                reference_answer=item.reference_answer,
                retrieved_documents=documents,
                use_llm_judge=use_answer_judge,
            )

            citation_metrics = CitationMetrics.evaluate(
                answer=answer,
                retrieved_documents=documents,
                reference_pages=item.reference_pages,
            )

            rows.append({
                "question_id": item.question_id,
                "question": item.question,
                "status": "success",
                "latency_seconds": round(latency, 4),
                "answer": answer,
                "reference_answer": item.reference_answer,
                "retrieved_documents": documents,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieval_metrics": retrieval_metrics,
                "answer_metrics": answer_metrics,
                "citation_metrics": citation_metrics,
            })

        except Exception as exc:
            latency = time.perf_counter() - started

            logger.exception(
                "Evaluation failed for question %s",
                item.question_id,
            )

            rows.append({
                "question_id": item.question_id,
                "question": item.question,
                "status": "error",
                "latency_seconds": round(latency, 4),
                "error": f"{type(exc).__name__}: {exc}",
            })

            if fail_fast:
                break

    report = {
        "report_version": "1.0",
        "dataset": dataset.summary(),
        "run": {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "graph_factory": graph_factory,
            "top_k": top_k,
            "answer_judge_enabled": use_answer_judge,
            "questions_requested": len(questions),
            "questions_evaluated": len(rows),
        },
        "summary": _aggregate_results(rows, top_k),
        "dataset_warnings": warnings,
        "results": rows,
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=str)

    logger.info("Evaluation report saved to %s", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the MEA Annual Report RAG chatbot."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-answer-judge",
        action="store_true",
        help="Disable additional Groq LLM-as-judge calls.",
    )
    parser.add_argument(
        "--graph-factory",
        default="main:build_application",
        help="Factory path in module:function format.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    report = asyncio.run(
        run_evaluation(
            dataset_path=args.dataset,
            output_path=args.output,
            limit=args.limit,
            top_k=args.top_k,
            use_answer_judge=not args.no_answer_judge,
            graph_factory=args.graph_factory,
            fail_fast=args.fail_fast,
        )
    )

    print("\n========== MEA RAG EVALUATION ==========")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nReport: {args.output.resolve()}")


if __name__ == "__main__":
    main()
