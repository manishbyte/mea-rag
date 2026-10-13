
from __future__ import annotations

import asyncio
from typing import Any, Sequence

from llm.models import LLMModels
from rag.prompts import SYSTEM_PROMPT, build_user_prompt


class AnswerGenerator:
    """Generate evidence-grounded answers from retrieved chunks."""

    def __init__(
        self,
        llm: LLMModels | None = None,
        max_completion_tokens: int = 2048,
        temperature: float = 0.2,
        max_concurrent_requests: int = 4,
    ) -> None:
        if max_completion_tokens < 1:
            raise ValueError(
                "max_completion_tokens must be at least 1."
            )

        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")

        if max_concurrent_requests < 1:
            raise ValueError(
                "max_concurrent_requests must be at least 1."
            )

        self.llm = llm or LLMModels()
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(
            max_concurrent_requests
        )

    @staticmethod
    def _format_context(
        documents: Sequence[dict[str, Any]],
    ) -> str:
        """Format retrieved chunks with source and page metadata."""

        context_parts: list[str] = []

        for index, document in enumerate(documents, start=1):
            if not isinstance(document, dict):
                continue

            text = document.get("text")

            if not isinstance(text, str) or not text.strip():
                continue

            metadata = document.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            source = document.get("source") or metadata.get("source")
            page_number = document.get("page_number")

            if page_number is None:
                page_number = metadata.get("page_number")

            chunk_id = document.get("chunk_id", "unknown")

            context_parts.append(
                "\n".join(
                    [
                        f"[Context {index}]",
                        f"Chunk ID: {chunk_id}",
                        f"Source: {source or 'Unknown source'}",
                        f"Page: {page_number if page_number is not None else 'Unknown'}",
                        "Text:",
                        text.strip(),
                    ]
                )
            )

        return "\n\n---\n\n".join(context_parts)

    async def generate(
        self,
        question: str,
        retrieved_documents: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate an answer and return its supporting context."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must not be empty.")

        if not retrieved_documents:
            return {
                "question": question.strip(),
                "answer": (
                    "I couldn't find relevant evidence in the "
                    "MEA Annual Report 2025 to answer this question."
                ),
                "context_count": 0,
                "context": "",
                "model": self.llm.model_name,
            }

        context = self._format_context(retrieved_documents)

        if not context:
            return {
                "question": question.strip(),
                "answer": (
                    "I couldn't find usable text in the retrieved "
                    "document chunks to answer this question."
                ),
                "context_count": 0,
                "context": "",
                "model": self.llm.model_name,
            }

        user_prompt = build_user_prompt(
            question=question,
            context=context,
        )

        async with self._semaphore:
            response = await self.llm.client.chat.completions.create(
                model=self.llm.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
            )

        if not response.choices:
            raise RuntimeError("The LLM returned no completion choices.")

        answer = response.choices[0].message.content

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The LLM returned an empty answer.")

        return {
            "question": question.strip(),
            "answer": answer.strip(),
            "context_count": len(retrieved_documents),
            "context": context,
            "model": self.llm.model_name,
        }

    async def close(self) -> None:
        """Close the LLM client owned by this generator."""
        await self.llm.close()

    async def __aenter__(self) -> AnswerGenerator:
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        await self.close()
