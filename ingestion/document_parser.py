
from __future__ import annotations

import asyncio
import re
from typing import Any


class DocumentParser:
    """Clean and normalize page-level PDF extraction results."""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace without merging separate paragraphs."""

        text = text.replace("\x00", "")
        text = text.replace("\u00ad", "")  # Soft hyphen

        # Normalize line endings and horizontal whitespace.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)

        # Remove spaces around line breaks.
        text = re.sub(r" *\n *", "\n", text)

        # Limit excessive blank lines while preserving paragraphs.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def _parse_sync(
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate and clean extracted pages."""

        parsed_pages: list[dict[str, Any]] = []

        for page in pages:
            page_number = page.get("page_number")
            text = page.get("text")
            source = page.get("source")

            if not isinstance(page_number, int) or page_number < 1:
                raise ValueError(
                    f"Invalid page number: {page_number!r}"
                )

            if not isinstance(text, str):
                raise TypeError(
                    f"Page {page_number}: text must be a string."
                )

            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    f"Page {page_number}: source is missing."
                )

            cleaned_text = DocumentParser._clean_text(text)

            parsed_pages.append(
                {
                    "page_number": page_number,
                    "text": cleaned_text,
                    "source": source,
                    "char_count": len(cleaned_text),
                    "is_empty": not bool(cleaned_text),
                }
            )

        return parsed_pages

    async def parse(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Parse pages without blocking the asyncio event loop."""

        return await asyncio.to_thread(self._parse_sync, pages)
