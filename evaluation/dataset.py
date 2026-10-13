
"""Load and validate the MEA RAG evaluation dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent
    / "datasets"
    / "mea_evaluation.json"
)


class DatasetError(ValueError):
    """Raised when the evaluation dataset is invalid."""


@dataclass
class EvaluationQuestion:
    question_id: str
    question: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    reference_answer: str = ""
    reference_pages: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "EvaluationQuestion":
        if not isinstance(item, dict):
            raise DatasetError("Every dataset question must be a JSON object.")

        question_id = str(item.get("question_id", "")).strip()
        question = str(item.get("question", "")).strip()

        if not question_id:
            raise DatasetError("A question is missing question_id.")

        if not question:
            raise DatasetError(
                f"Question {question_id!r} has an empty question."
            )

        raw_chunk_ids = item.get("relevant_chunk_ids", [])
        raw_pages = item.get("reference_pages", [])

        if not isinstance(raw_chunk_ids, list):
            raise DatasetError(
                f"{question_id}: relevant_chunk_ids must be a list."
            )

        if not isinstance(raw_pages, list):
            raise DatasetError(
                f"{question_id}: reference_pages must be a list."
            )

        try:
            pages = [int(page) for page in raw_pages]
        except (TypeError, ValueError) as exc:
            raise DatasetError(
                f"{question_id}: reference_pages must contain integers."
            ) from exc

        metadata = item.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {"value": metadata}

        return cls(
            question_id=question_id,
            question=question,
            relevant_chunk_ids=[
                str(chunk_id)
                for chunk_id in raw_chunk_ids
                if chunk_id is not None
            ],
            reference_answer=str(
                item.get("reference_answer", "") or ""
            ).strip(),
            reference_pages=pages,
            metadata=metadata,
        )


@dataclass
class EvaluationDataset:
    dataset_name: str
    version: str
    description: str
    questions: list[EvaluationQuestion]

    @property
    def size(self) -> int:
        return len(self.questions)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "description": self.description,
            "total_questions": self.size,
            "questions_with_relevant_chunks": sum(
                bool(q.relevant_chunk_ids) for q in self.questions
            ),
            "questions_with_reference_answers": sum(
                bool(q.reference_answer) for q in self.questions
            ),
            "questions_with_reference_pages": sum(
                bool(q.reference_pages) for q in self.questions
            ),
        }


def load_dataset(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> EvaluationDataset:
    """Load a JSON evaluation dataset without modifying the source file."""

    dataset_path = Path(path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    try:
        with dataset_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except json.JSONDecodeError as exc:
        raise DatasetError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc

    if isinstance(raw, list):
        # Also support a JSON file containing a bare list of questions.
        raw_questions = raw
        dataset_name = dataset_path.stem
        version = "1.0"
        description = ""
    elif isinstance(raw, dict):
        raw_questions = raw.get("questions")

        if raw_questions is None:
            raw_questions = raw.get("data")

        if not isinstance(raw_questions, list):
            raise DatasetError(
                "Dataset must contain a 'questions' list, "
                "a 'data' list, or be a JSON list itself."
            )

        dataset_name = str(
            raw.get("dataset_name", dataset_path.stem)
        )
        version = str(raw.get("version", "1.0"))
        description = str(raw.get("description", ""))
    else:
        raise DatasetError(
            "The top-level dataset JSON must be an object or a list."
        )

    questions = [
        EvaluationQuestion.from_dict(item)
        for item in raw_questions
    ]

    if not questions:
        raise DatasetError("The evaluation dataset contains no questions.")

    question_ids = [q.question_id for q in questions]

    if len(question_ids) != len(set(question_ids)):
        raise DatasetError("Duplicate question_id values were found.")

    return EvaluationDataset(
        dataset_name=dataset_name,
        version=version,
        description=description,
        questions=questions,
    )


def validate_dataset(
    dataset: EvaluationDataset,
    *,
    require_relevant_chunks: bool = False,
    require_reference_answers: bool = False,
) -> list[str]:
    """Return validation warnings; raise only for invalid required labels."""

    warnings: list[str] = []

    for question in dataset.questions:
        if not question.relevant_chunk_ids:
            message = (
                f"{question.question_id}: no relevant chunk IDs labelled."
            )
            if require_relevant_chunks:
                raise DatasetError(message)
            warnings.append(message)

        if not question.reference_answer:
            message = (
                f"{question.question_id}: no reference answer provided."
            )
            if require_reference_answers:
                raise DatasetError(message)
            warnings.append(message)

        if not question.reference_pages:
            warnings.append(
                f"{question.question_id}: no reference pages labelled."
            )

    return warnings
