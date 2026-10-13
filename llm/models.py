
from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()


class LLMModels:
    """Configure and provide the asynchronous Groq LLM client."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Configure it in your .env file."
            )

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        self.client = AsyncGroq(
            api_key=self.api_key,
            timeout=timeout,
            max_retries=2,
        )

    async def close(self) -> None:
        """Close the asynchronous Groq client."""
        await self.client.close()

    async def __aenter__(self) -> LLMModels:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.close()
