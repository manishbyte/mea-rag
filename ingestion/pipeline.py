
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ingestion.pdf_loader import PDFLoader
from ingestion.document_parser import DocumentParser
from ingestion.metadata_extractor import MetadataExtractor
from ingestion.table_extractor import TableExtractor
from ingestion.chunker import DocumentChunker


logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Run the MEA Annual Report ingestion workflow."""

    def __init__(
        self,
        pdf_path: str | Path = "data/raw/mea_annual_report.pdf",
        output_dir: str | Path = "data/processed",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)

        self.loader = PDFLoader(self.pdf_path)
        self.parser = DocumentParser()
        self.metadata_extractor = MetadataExtractor(self.pdf_path)
        self.table_extractor = TableExtractor(self.pdf_path)

        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

    @staticmethod
    def _write_json_sync(
        output_path: Path,
        data: Any,
    ) -> None:
        """Write JSON from a worker thread."""

        output_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        try:
            with temporary_path.open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            temporary_path.replace(output_path)

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    async def _save_json(
        self,
        filename: str,
        data: Any,
    ) -> Path:
        output_path = self.output_dir / filename

        await asyncio.to_thread(
            self._write_json_sync,
            output_path,
            data,
        )

        return output_path

    async def run(self) -> dict[str, Any]:
        """Execute the complete ingestion workflow."""

        if not self.pdf_path.is_file():
            raise FileNotFoundError(
                f"MEA Annual Report PDF not found: {self.pdf_path}"
            )

        logger.info("Loading PDF: %s", self.pdf_path)

        # Step 1: Extract text page by page.
        raw_pages = await self.loader.load()

        # Step 2: Clean and validate extracted page text.
        parsed_pages = await self.parser.parse(raw_pages)

        # Step 3: Attach document metadata to every page.
        enriched_pages = await self.metadata_extractor.enrich_pages(
            parsed_pages
        )

        # Step 4: Extract tables independently of text processing.
        # Both tasks use worker threads internally.
        table_task = asyncio.create_task(
            self.table_extractor.extract_tables()
        )

        # Step 5: Chunk the page-level text.
        chunks = await self.chunker.chunk_pages(enriched_pages)

        # Wait for table extraction and propagate any failure.
        tables = await table_task

        # Step 6: Save processed data.
        pages_path, chunks_path, tables_path = await asyncio.gather(
            self._save_json("mea_pages.json", enriched_pages),
            self._save_json("mea_chunks.json", chunks),
            self._save_json("mea_tables.json", tables),
        )

        total_characters = sum(
            len(str(page["text"])) for page in enriched_pages
        )

        summary = {
            "document": self.pdf_path.name,
            "total_pages": len(enriched_pages),
            "non_empty_pages": sum(
                not page.get("is_empty", False)
                for page in enriched_pages
            ),
            "total_characters": total_characters,
            "total_chunks": len(chunks),
            "total_tables": len(tables),
            "pages_output": str(pages_path),
            "chunks_output": str(chunks_path),
            "tables_output": str(tables_path),
        }

        logger.info("Ingestion completed: %s", summary)

        return summary


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    pipeline = IngestionPipeline()
    summary = await pipeline.run()

    print("\n=== MEA Ingestion Summary ===")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
