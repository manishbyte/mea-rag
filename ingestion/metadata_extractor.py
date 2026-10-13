
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import fitz


class MetadataExtractor:
    """Extract PDF metadata and attach it to parsed pages."""

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)

    def _extract_pdf_metadata_sync(self) -> dict[str, Any]:
        """Read PDF metadata without blocking the event loop."""

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.pdf_path}"
            )

        with fitz.open(self.pdf_path) as pdf:
            if pdf.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            pdf_info = pdf.metadata or {}

            # Stable identifier based on the PDF file contents.
            digest = hashlib.sha256()

            with self.pdf_path.open("rb") as file:
                for block in iter(
                    lambda: file.read(1024 * 1024), b""
                ):
                    digest.update(block)

            return {
                "document_id": digest.hexdigest(),
                "source": self.pdf_path.name,
                "file_path": str(self.pdf_path.resolve()),
                "title": (
                    pdf_info.get("title") or ""
                ).strip() or self.pdf_path.stem,
                "author": (pdf_info.get("author") or "").strip(),
                "subject": (pdf_info.get("subject") or "").strip(),
                "creator": (pdf_info.get("creator") or "").strip(),
                "producer": (pdf_info.get("producer") or "").strip(),
                "creation_date": (
                    pdf_info.get("creationDate") or ""
                ).strip(),
                "total_pages": len(pdf),
                "document_type": "MEA Annual Report",
            }

    async def extract_pdf_metadata(self) -> dict[str, Any]:
        """Extract PDF metadata asynchronously."""

        return await asyncio.to_thread(
            self._extract_pdf_metadata_sync
        )

    async def enrich_pages(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach document metadata to every parsed page."""

        metadata = await self.extract_pdf_metadata()

        enriched_pages: list[dict[str, Any]] = []

        for page in pages:
            enriched_page = {
                **page,
                **metadata,
                # Explicitly preserve each page's own number.
                "page_number": page["page_number"],
            }

            enriched_pages.append(enriched_page)

        return enriched_pages
