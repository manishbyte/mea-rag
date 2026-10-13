
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx
import streamlit as st

from ui.components.sources import render_sources

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = httpx.Timeout(
    timeout=120.0,
    connect=10.0,
)


def _headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _api_error_message(response: httpx.Response) -> str:
    """Return a safe, readable error from a FastAPI response."""
    try:
        payload = response.json()
        detail = payload.get("detail")

        if isinstance(detail, str) and detail.strip():
            return detail
    except ValueError:
        pass

    if response.status_code == 401:
        return "Your session has expired. Please log in again."

    if response.status_code == 503:
        return "The RAG application is not ready yet. Please try again shortly."

    return f"The API request failed with status {response.status_code}."


def _load_messages(
    api_base_url: str,
    access_token: str,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """Load a conversation's persisted messages from the backend."""
    url = (
        f"{api_base_url.rstrip('/')}/api/chat/conversations/"
        f"{quote(conversation_id, safe='')}/messages"
    )

    try:
        response = httpx.get(
            url,
            headers=_headers(access_token),
            params={"limit": 500, "offset": 0},
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 401:
            st.session_state["auth_expired"] = True
            return []

        response.raise_for_status()
        payload = response.json()

        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            logger.warning("Unexpected conversation messages format.")
            return []

        normalized: list[dict[str, Any]] = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content")

            if role not in ("user", "assistant") or not isinstance(content, str):
                continue

            normalized.append(
                {
                    "role": role,
                    "content": content,
                    "citations": message.get("citations") or [],
                    "context_count": message.get("context_count", 0),
                    "latency_ms": message.get("latency_ms"),
                }
            )

        return normalized

    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Failed to load conversation history: HTTP %s",
            exc.response.status_code,
        )
        st.error(_api_error_message(exc.response))
    except httpx.RequestError:
        logger.exception("Could not load conversation history.")
        st.error("Could not connect to the backend API.")
    except ValueError:
        logger.exception("Invalid conversation history response.")
        st.error("The server returned an invalid conversation response.")

    return []


def _send_question(
    api_base_url: str,
    access_token: str,
    question: str,
    conversation_id: str | None,
) -> dict[str, Any] | None:
    """Send a question to the authenticated chat endpoint."""
    payload: dict[str, Any] = {"question": question}

    if conversation_id:
        payload["conversation_id"] = conversation_id

    try:
        response = httpx.post(
            f"{api_base_url.rstrip('/')}/api/chat",
            headers=_headers(access_token),
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 401:
            st.session_state["auth_expired"] = True
            return None

        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
            logger.error("The chat API returned an unexpected response.")
            st.error("The server returned an invalid answer.")
            return None

        return data

    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Chat request failed with HTTP %s",
            exc.response.status_code,
        )
        st.error(_api_error_message(exc.response))
    except httpx.TimeoutException:
        st.error(
            "The request took too long. Please check your connection "
            "and whether the backend is still processing."
        )
    except httpx.RequestError:
        logger.exception("Could not connect to the chat API.")
        st.error("Could not connect to the backend API.")
    except ValueError:
        logger.exception("Invalid chat API response.")
        st.error("The server returned an invalid response.")

    return None


def _render_message(message: dict[str, Any]) -> None:
    """Render one message and any citations belonging to it."""
    role = message.get("role", "assistant")
    content = message.get("content", "")

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant":
            citations = message.get("citations") or []

            if citations:
                render_sources(citations)
            else:
                st.caption("No citations were returned for this response.")

            latency_ms = message.get("latency_ms")
            context_count = message.get("context_count")

            metadata: list[str] = []

            if isinstance(context_count, int):
                metadata.append(f"Retrieved chunks: {context_count}")

            if isinstance(latency_ms, (int, float)):
                metadata.append(f"Response time: {latency_ms / 1000:.2f}s")

            if metadata:
                st.caption(" · ".join(metadata))


def render_chat(
    api_base_url: str,
    access_token: str,
) -> None:
    """
    Render the chat history and input.

    Expected session-state keys:
        conversation_id
        messages
        load_conversation_id
    """
    st.title("MEA Annual Report 2025")
    st.caption(
        "Ask questions about the report. Answers depend on the "
        "available document context and retrieved sources."
    )

    if st.session_state.pop("auth_expired", False):
        for key in (
            "access_token",
            "user",
            "authenticated",
            "conversation_id",
            "messages",
            "active_conversation_title",
            "load_conversation_id",
        ):
            st.session_state.pop(key, None)

        st.warning("Your session has expired. Please log in again.")
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Load history when a conversation is selected in the sidebar.
    load_id = st.session_state.get("load_conversation_id")
    current_id = st.session_state.get("conversation_id")

    if load_id and str(load_id) == str(current_id):
        with st.spinner("Loading conversation history..."):
            st.session_state["messages"] = _load_messages(
                api_base_url=api_base_url,
                access_token=access_token,
                conversation_id=str(load_id),
            )

        st.session_state.pop("load_conversation_id", None)

        if st.session_state.get("auth_expired"):
            st.rerun()

    messages = st.session_state.get("messages", [])

    if not messages:
        st.info(
            "Start by asking a question about the Ministry of External "
            "Affairs Annual Report 2025."
        )

    for message in messages:
        _render_message(message)

    question = st.chat_input(
        "Ask a question about the annual report...",
        max_chars=10_000,
    )

    if not question or not question.strip():
        return

    question = question.strip()

    # Display the user's message immediately.
    user_message = {
        "role": "user",
        "content": question,
        "citations": [],
    }

    st.session_state["messages"].append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the report and generating an answer..."):
            result = _send_question(
                api_base_url=api_base_url,
                access_token=access_token,
                question=question,
                conversation_id=st.session_state.get("conversation_id"),
            )

        if st.session_state.get("auth_expired"):
            for key in (
                "access_token",
                "user",
                "authenticated",
                "conversation_id",
                "messages",
                "active_conversation_title",
                "load_conversation_id",
            ):
                st.session_state.pop(key, None)

            st.warning("Your session has expired. Please log in again.")
            st.rerun()

        if result is None:
            # Keep the question visible. The persisted history will be
            # reloaded when the user opens this conversation again.
            st.session_state["messages"] = _load_messages(
                api_base_url=api_base_url,
                access_token=access_token,
                conversation_id=str(
                    st.session_state.get("conversation_id") or ""
                ),
            ) if st.session_state.get("conversation_id") else [
                message
                for message in st.session_state["messages"]
                if message is not user_message
            ]
            return

        answer = result["answer"]
        citations = result.get("citations") or []

        # The first successful question creates a conversation on the server.
        new_conversation_id = result.get("conversation_id")
        if new_conversation_id:
            st.session_state["conversation_id"] = str(
                new_conversation_id
            )

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "context_count": result.get("context_count", 0),
            "latency_ms": result.get("latency_ms"),
        }

        st.markdown(answer)

        if citations:
            render_sources(citations)
        else:
            st.caption("No citations were returned for this response.")

        st.session_state["messages"].append(assistant_message)
