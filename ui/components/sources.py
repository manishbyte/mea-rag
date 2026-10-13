
from __future__ import annotations

from typing import Any

import streamlit as st


def _first_value(
    source: dict[str, Any],
    *keys: str,
) -> Any:
    """Return the first non-empty value among supported field names."""
    for key in keys:
        value = source.get(key)
        if value is not None and value != "":
            return value
    return None


def _normalize_citation(
    citation: Any,
    index: int,
) -> dict[str, Any] | None:
    """
    Normalize a citation from the RAG pipeline.

    Supports common citation formats, including dictionaries and strings.
    Missing metadata is left unknown rather than fabricated.
    """
    if isinstance(citation, str):
        text = citation.strip()
        if not text:
            return None

        return {
            "label": f"Source {index}",
            "source": None,
            "page": None,
            "excerpt": text,
            "url": None,
        }

    if not isinstance(citation, dict):
        return None

    metadata = citation.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    document = citation.get("document")
    if not isinstance(document, dict):
        document = {}

    source = _first_value(
        citation,
        "source",
        "source_name",
        "document_name",
        "filename",
        "file_name",
    )

    if source is None:
        source = _first_value(
            metadata,
            "source",
            "source_name",
            "document_name",
            "filename",
        )

    if source is None:
        source = _first_value(
            document,
            "source",
            "source_name",
            "document_name",
            "filename",
        )

    page = _first_value(
        citation,
        "page_number",
        "page",
        "page_number_1based",
    )

    if page is None:
        page = _first_value(
            metadata,
            "page_number",
            "page",
            "page_number_1based",
        )

    if page is None:
        page = _first_value(
            document,
            "page_number",
            "page",
        )

    excerpt = _first_value(
        citation,
        "excerpt",
        "text",
        "content",
        "chunk_text",
    )

    if excerpt is None:
        excerpt = _first_value(
            document,
            "page_content",
            "text",
            "content",
        )

    url = _first_value(citation, "url", "source_url")
    if url is None:
        url = _first_value(metadata, "url", "source_url")

    return {
        "label": str(
            _first_value(citation, "label", "title")
            or source
            or f"Source {index}"
        ),
        "source": str(source) if source is not None else None,
        "page": page,
        "excerpt": str(excerpt) if excerpt is not None else None,
        "url": str(url) if url is not None else None,
    }


def render_sources(
    citations: list[Any] | None,
    *,
    expanded: bool = False,
) -> None:
    """
    Render source citations for one assistant response.

    Args:
        citations: Citations returned by the backend.
        expanded: Whether citation panels should initially be expanded.
    """
    if not citations:
        st.caption("No citations were provided for this response.")
        return

    normalized: list[dict[str, Any]] = []

    for index, citation in enumerate(citations, start=1):
        item = _normalize_citation(citation, index)
        if item is not None:
            normalized.append(item)

    if not normalized:
        st.caption("No displayable source details were returned.")
        return

    st.markdown("**Sources**")

    for index, source in enumerate(normalized, start=1):
        title = source["label"]

        details: list[str] = []

        if source["source"]:
            details.append(source["source"])

        if source["page"] is not None:
            details.append(f"Page {source['page']}")

        if details:
            title = f"{title} — {' · '.join(details)}"

        with st.expander(title, expanded=expanded):
            excerpt = source["excerpt"]

            if excerpt:
                st.markdown("**Relevant excerpt**")
                st.write(excerpt)
            else:
                st.caption("No excerpt was included in the citation.")

            url = source["url"]

            if url:
                if url.startswith(("https://", "http://")):
                    st.link_button(
                        "Open source",
                        url,
                        use_container_width=False,
                    )
                else:
                    st.caption("A source link was omitted because it is not an HTTP(S) URL.")
