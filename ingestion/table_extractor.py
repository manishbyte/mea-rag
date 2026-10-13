
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import fitz


class TableExtractor:
    """Extract PDF tables and preserve their source page numbers."""

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)

    def _extract_tables_sync(self) -> list[dict[str, Any]]:
        """Extract tables synchronously inside a worker thread."""

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.pdf_path}"
            )

        tables: list[dict[str, Any]] = []

        with fitz.open(self.pdf_path) as pdf:
            if pdf.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            # Same document ID approach used by MetadataExtractor.
            digest = hashlib.sha256()

            with self.pdf_path.open("rb") as file:
                for block in iter(
                    lambda: file.read(1024 * 1024), b""
                ):
                    digest.update(block)

            document_id = digest.hexdigest()

            for page_index, page in enumerate(pdf):
                try:
                    page_tables = page.find_tables()
                except Exception as exc:
                    print(
                        f"Warning: table detection failed on "
                        f"page {page_index + 1}: {exc}"
                    )
                    continue

                for table_index, table in enumerate(
                    page_tables.tables, start=1
                ):
                    raw_rows = table.extract()

                    if not raw_rows:
                        continue

                    # Normalize cell values and discard completely
                    # empty rows.
                    rows = [
                        [
                            str(cell).strip() if cell is not None else ""
                            for cell in row
                        ]
                        for row in raw_rows
                    ]

                    rows = [
                        row for row in rows
                        if any(cell for cell in row)
                    ]

                    if not rows:
                        continue

                    tables.append(
                        {
                            "table_id": (
                                f"{document_id[:12]}_"
                                f"p{page_index + 1}_t{table_index}"
                            ),
                            "document_id": document_id,
                            "source": self.pdf_path.name,
                            "page_number": page_index + 1,
                            "table_number_on_page": table_index,
                            "rows": rows,
                            "row_count": len(rows),
                            "column_count": max(
                                len(row) for row in rows
                            ),
                            "text": self._rows_to_text(rows),
                        }
                    )

        return tables

    @staticmethod
    def _rows_to_text(rows: list[list[str]]) -> str:
        """Convert table rows into readable text for later indexing."""

        lines: list[str] = []

        for row_index, row in enumerate(rows):
            cells = [
                f"Column {column_index + 1}: {cell}"
                for column_index, cell in enumerate(row)
                if cell
            ]

            if cells:
                lines.append(
                    f"Row {row_index + 1}: " + "; ".join(cells)
                )

        return "\n".join(lines)

    async def extract_tables(self) -> list[dict[str, Any]]:
        """Extract tables without blocking the asyncio event loop."""

        return await asyncio.to_thread(self._extract_tables_sync)

    async def save_tables(
        self,
        tables: list[dict[str, Any]],
        output_path: str | Path = "data/processed/mea_tables.json",
    ) -> Path:
        """Save extracted tables as JSON without blocking the event loop."""

        destination = Path(output_path)

        def write_json() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)

            with destination.open("w", encoding="utf-8") as file:
                json.dump(
                    tables,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        await asyncio.to_thread(write_json)
        return destination


async def main() -> None:
    pdf_path = Path("data/raw/mea_annual_report.pdf")

    extractor = TableExtractor(pdf_path)
    tables = await extractor.extract_tables()

    output_path = await extractor.save_tables(tables)

    print(f"PDF: {pdf_path.name}")
    print(f"Tables extracted: {len(tables)}")
    print(f"Saved to: {output_path}")

    for table in tables[:3]:
        print(
            f"\nPage {table['page_number']} | "
            f"Table {table['table_number_on_page']} | "
            f"Rows: {table['row_count']} | "
            f"Columns: {table['column_count']}"
        )
        print(table["text"][:800])


if __name__ == "__main__":
    asyncio.run(main())
