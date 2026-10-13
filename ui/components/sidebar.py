
from __future__ import annotations

import logging
from typing import Any

import httpx
import streamlit as st

logger = logging.getLogger(__name__)


def _api_headers(access_token: str) -> dict[str, str]:
    """Build authentication headers for protected API requests."""
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _request_conversations(
    api_base_url: str,
    access_token: str,
) -> list[dict[str, Any]]:
    """Fetch the current user's conversations from FastAPI."""
    try:
        response = httpx.get(
            f"{api_base_url.rstrip('/')}/api/chat/conversations",
            headers=_api_headers(access_token),
            params={"limit": 100, "offset": 0},
            timeout=20.0,
        )

        if response.status_code == 401:
            st.session_state.pop("access_token", None)
            st.session_state.pop("user", None)
            st.session_state["authenticated"] = False
            st.rerun()

        response.raise_for_status()

        data = response.json()
        if not isinstance(data, list):
            logger.warning("Unexpected conversations response format.")
            return []

        return data

    except httpx.HTTPStatusError:
        logger.exception("The API rejected the conversations request.")
        st.error("Could not load your conversations from the server.")
    except httpx.RequestError:
        logger.exception("Could not connect to the API.")
        st.error(
            "Cannot connect to the backend API. "
            "Check that FastAPI is running."
        )
    except ValueError:
        logger.exception("The API returned invalid JSON.")
        st.error("The server returned an invalid response.")

    return []


def render_sidebar(
    api_base_url: str,
    access_token: str,
) -> str | None:
    """
    Render the conversation sidebar.

    Returns:
        The selected conversation ID, or None for a new conversation.
    """
    with st.sidebar:
        st.title("MEA Research Assistant")
        st.caption("Annual Report 2025")

        st.divider()

        user = st.session_state.get("user") or {}
        email = user.get("email", "Logged-in user")
        full_name = user.get("full_name")

        st.markdown("### Account")
        if full_name:
            st.write(full_name)
        st.caption(email)

        if st.button(
            "＋ New conversation",
            use_container_width=True,
            type="primary",
        ):
            st.session_state["conversation_id"] = None
            st.session_state["messages"] = []
            st.session_state["active_conversation_title"] = None
            st.rerun()

        st.divider()
        st.markdown("### Recent conversations")

        conversations = _request_conversations(
            api_base_url=api_base_url,
            access_token=access_token,
        )

        if not conversations:
            st.caption("No previous conversations yet.")
        else:
            selected_id = st.session_state.get("conversation_id")

            for conversation in conversations:
                conversation_id = str(conversation.get("id", ""))
                if not conversation_id:
                    continue

                title = str(
                    conversation.get("title") or "New conversation"
                ).strip()

                # Keep sidebar labels compact.
                if len(title) > 38:
                    title = title[:35] + "..."

                is_selected = conversation_id == str(selected_id)

                if st.button(
                    f"{'● ' if is_selected else '　'}{title}",
                    key=f"conversation_{conversation_id}",
                    use_container_width=True,
                    help=conversation.get("title") or "Open conversation",
                ):
                    st.session_state["conversation_id"] = conversation_id
                    st.session_state["active_conversation_title"] = (
                        conversation.get("title") or "Conversation"
                    )
                    st.session_state["messages"] = []
                    st.session_state["load_conversation_id"] = (
                        conversation_id
                    )
                    st.rerun()

        st.divider()

        if st.button("Log out", use_container_width=True):
            # Clear client-side authentication and conversation state.
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

            st.rerun()

    return st.session_state.get("conversation_id")
