
"""Answer quality metrics: lexical diagnostics and optional LLM judging."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _f1_from_tokens(
    prediction: str,
    reference: str,
) -> float | None:
    predicted_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)

    if not predicted_tokens or not reference_tokens:
        return None

    predicted_counts = Counter(predicted_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum((predicted_counts & reference_counts).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


class AnswerMetrics:
    """Measure answer overlap and optionally request an LLM evaluation."""

    @staticmethod
    def reference_similarity(
        answer: str,
        reference_answer: str,
    ) -> dict[str, Any]:
        """Calculate token F1 and token-set Jaccard similarity."""
        answer_tokens = _tokens(answer)
        reference_tokens = _tokens(reference_answer)

        f1 = _f1_from_tokens(answer, reference_answer)

        answer_set = set(answer_tokens)
        reference_set = set(reference_tokens)
        union = answer_set | reference_set

        jaccard = (
            len(answer_set & reference_set) / len(union)
            if union
            else None
        )

        return {
            "token_f1": f1,
            "token_jaccard": jaccard,
            "answer_token_count": len(answer_tokens),
            "reference_token_count": len(reference_tokens),
        }

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        """Parse JSON from a judge response, including fenced JSON."""
        content = content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)

            if not match:
                raise ValueError("Judge did not return a JSON object.")

            result = json.loads(match.group(0))

        if not isinstance(result, dict):
            raise ValueError("Judge output must be a JSON object.")

        return result

    @classmethod
    async def llm_judge(
        cls,
        *,
        question: str,
        answer: str,
        reference_answer: str,
        retrieved_documents: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Use Groq to judge answer quality; requires GROQ_API_KEY."""
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            return {
                "available": False,
                "error": "GROQ_API_KEY is not configured.",
            }

        try:
            from groq import AsyncGroq
        except ImportError:
            return {
                "available": False,
                "error": "Install the groq package to use the LLM judge.",
            }

        model_name = (
            model
            or os.getenv("GROQ_EVALUATION_MODEL")
            or os.getenv("GROQ_MODEL")
        )

        if not model_name:
            return {
                "available": False,
                "error": "Set GROQ_EVALUATION_MODEL or GROQ_MODEL.",
            }

        documents = retrieved_documents or []
        context = "\n\n".join(
            str(doc.get("text", ""))[:2500]
            for doc in documents[:5]
        )

        prompt = f"""
Evaluate this RAG answer against the reference answer and retrieved
evidence. Do not reward unsupported claims. A reference answer is a
guide, not a reason to accept claims that contradict the evidence.

Return only a JSON object with:
- correctness: integer from 1 to 5
- relevance: integer from 1 to 5
- groundedness: integer from 1 to 5
- completeness: integer from 1 to 5
- reasoning: brief explanation

Question:
{question}

Reference answer:
{reference_answer}

Retrieved evidence:
{context or "(No retrieved evidence provided)"}

Generated answer:
{answer}
""".strip()

        client = AsyncGroq(api_key=api_key)

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evaluator of retrieval-"
                            "augmented generation systems. Return valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )

            content = response.choices[0].message.content or ""
            scores = cls._extract_json(content)

            normalized: dict[str, Any] = {
                "available": True,
                "judge_model": model_name,
            }

            for name in (
                "correctness",
                "relevance",
                "groundedness",
                "completeness",
            ):
                try:
                    score = int(scores[name])
                except (KeyError, TypeError, ValueError):
                    raise ValueError(
                        f"Judge output is missing a valid {name} score."
                    )

                if not 1 <= score <= 5:
                    raise ValueError(
                        f"Judge {name} score must be between 1 and 5."
                    )

                normalized[name] = score

            normalized["overall_score"] = sum(
                normalized[name]
                for name in (
                    "correctness",
                    "relevance",
                    "groundedness",
                    "completeness",
                )
            ) / 4

            normalized["reasoning"] = str(
                scores.get("reasoning", "")
            )

            return normalized

        except Exception as exc:
            return {
                "available": False,
                "judge_model": model_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            await client.close()

    @classmethod
    async def evaluate(
        cls,
        *,
        question: str,
        answer: str,
        reference_answer: str = "",
        retrieved_documents: list[dict[str, Any]] | None = None,
        use_llm_judge: bool = False,
        judge_model: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "answer_present": bool(answer.strip()),
            "answer_length_characters": len(answer),
            "reference_available": bool(reference_answer.strip()),
        }

        if reference_answer.strip():
            result.update(
                cls.reference_similarity(answer, reference_answer)
            )
        else:
            result.update({
                "token_f1": None,
                "token_jaccard": None,
                "answer_token_count": len(_tokens(answer)),
                "reference_token_count": 0,
            })

        if use_llm_judge:
            result["llm_judge"] = await cls.llm_judge(
                question=question,
                answer=answer,
                reference_answer=reference_answer,
                retrieved_documents=retrieved_documents,
                model=judge_model,
            )

        return result
