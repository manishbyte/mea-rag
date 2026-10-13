
import asyncio

from evaluation.answer_metrics import AnswerMetrics
from llm.models import LLMModels


async def main():
    llm = LLMModels()

    try:
        metrics = AnswerMetrics(llm=llm)

        result = await metrics.evaluate(
            question="What does the report say about India's foreign policy?",
            answer=(
                "The retrieved passage describes India's foreign policy "
                "priorities as supporting international cooperation."
            ),
            retrieved_documents=[
                {
                    "chunk_id": "test_chunk",
                    "text": (
                        "The passage discusses India's foreign policy "
                        "priorities and international cooperation."
                    ),
                    "metadata": {
                        "source": "mea_annual_report.pdf",
                        "page_number": 10,
                    },
                }
            ],
            reference_answer=(
                "The passage discusses India's foreign policy priorities "
                "and international cooperation."
            ),
            use_llm_judge=True,
        )

        print(result)

    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
