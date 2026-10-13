from __future__ import annotations

import asyncio
from pathlib import Path

import fitz  # PyMuPDF


class PDFLoader:
    """Asynchronously load PDF text while preserving page numbers."""

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)

    def _extract_pages_sync(self) -> list[dict[str, object]]:
        """Extract text from each PDF page. Runs in a worker thread."""

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.pdf_path}"
            )

        pages: list[dict[str, object]] = []

        with fitz.open(self.pdf_path) as pdf:
            if pdf.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()

                pages.append(
                    {
                        "page_number": page_index + 1,
                        "text": text,
                        "source": self.pdf_path.name,
                    }
                )

        if not pages:
            raise ValueError("The PDF contains no pages.")

        return pages

    async def load(self) -> list[dict[str, object]]:
        """Load the PDF without blocking the asyncio event loop."""

        return await asyncio.to_thread(self._extract_pages_sync)


async def main() -> None:
    pdf_path = Path("data/raw/mea_annual_report.pdf")

    loader = PDFLoader(pdf_path)
    pages = await loader.load()

    print(f"PDF: {pdf_path.name}")
    print(f"Total pages: {len(pages)}")

    for page in pages[:3]:
        print(f"\n--- Page {page['page_number']} ---")
        print(str(page["text"])[:500])


if __name__ == "__main__":
    asyncio.run(main())
