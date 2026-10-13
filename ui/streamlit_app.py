
from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

from ui.components.chat import render_chat
from ui.components.sidebar import render_sidebar

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_TITLE = "MEA Research Assistant"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
AUTH_TIMEOUT = httpx.Timeout(timeout=30.0, connect=10.0)


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

def initialize_session_state() -> None:
    """Initialize Streamlit state without overwriting existing values."""
    defaults: dict[str, Any] = {
        "access_token": None,
        "user": None,
        "authenticated": False,
        "conversation_id": None,
        "messages": [],
        "active_conversation_title": None,
        "load_conversation_id": None,
        "auth_mode": "login",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -------------------------------------------------------------------
# API helpers
# -------------------------------------------------------------------

def get_api_base_url() -> str:
    """
    Get the backend URL from the environment.

    Set API_BASE_URL in the deployment environment when the frontend
    and backend run as separate services.
    """
    return os.getenv(
        "API_BASE_URL",
        DEFAULT_API_BASE_URL,
    ).rstrip("/")


def api_error_detail(response: httpx.Response) -> str:
    """Extract a safe, readable error message from the API."""
    try:
        data = response.json()
        detail = data.get("detail")

        if isinstance(detail, str) and detail.strip():
            return detail

        if isinstance(detail, list):
            return "; ".join(
                str(item.get("msg", item))
                if isinstance(item, dict)
                else str(item)
                for item in detail
            )
    except ValueError:
        pass

    if response.status_code == 409:
        return "An account with this email may already exist."

    if response.status_code == 401:
        return "The email or password is incorrect."

    return f"Request failed with status {response.status_code}."


def call_auth_api(
    endpoint: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Call a public authentication endpoint."""
    api_base_url = get_api_base_url()

    try:
        response = httpx.post(
            f"{api_base_url}/api/auth/{endpoint}",
            json=payload,
            timeout=AUTH_TIMEOUT,
        )

        if not response.is_success:
            return None, api_error_detail(response)

        data = response.json()
        if not isinstance(data, dict):
            return None, "The server returned an unexpected response."

        return data, None

    except httpx.TimeoutException:
        return None, "The request timed out. Please try again."
    except httpx.RequestError:
        logger.exception("Could not connect to the authentication API.")
        return None, (
            "Cannot connect to the backend. Check that FastAPI is running "
            "and API_BASE_URL is correct."
        )
    except ValueError:
        logger.exception("Invalid authentication response.")
        return None, "The server returned an invalid response."


def fetch_current_user(
    api_base_url: str,
    access_token: str,
) -> dict[str, Any] | None:
    """Validate the token by requesting the authenticated user."""
    try:
        response = httpx.get(
            f"{api_base_url.rstrip('/')}/api/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=AUTH_TIMEOUT,
        )

        if not response.is_success:
            return None

        data = response.json()
        return data if isinstance(data, dict) else None

    except (httpx.RequestError, ValueError):
        logger.exception("Could not validate the current user session.")
        return None


# -------------------------------------------------------------------
# Authentication screens
# -------------------------------------------------------------------

def render_login() -> None:
    st.title("MEA Research Assistant")
    st.caption("Ask questions about the Ministry of External Affairs Annual Report 2025.")

    _, center, _ = st.columns([1, 1.3, 1])

    with center:
        st.subheader("Welcome back")
        st.write("Log in to continue your research conversations.")

        with st.form("login_form", clear_on_submit=False):
            email = st.text_input(
                "Email address",
                placeholder="you@example.com",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )
            submitted = st.form_submit_button(
                "Log in",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            if not email.strip() or not password:
                st.error("Enter your email address and password.")
            else:
                with st.spinner("Signing in..."):
                    data, error = call_auth_api(
                        "login",
                        {
                            "email": email.strip(),
                            "password": password,
                        },
                    )

                if error:
                    st.error(error)
                elif data:
                    token = data.get("access_token")

                    if not isinstance(token, str) or not token:
                        st.error("The API did not return an access token.")
                    else:
                        user = data.get("user")

                        if not isinstance(user, dict):
                            user = fetch_current_user(
                                get_api_base_url(),
                                token,
                            )

                        if not user:
                            st.error(
                                "Login succeeded, but the user profile "
                                "could not be loaded."
                            )
                        else:
                            st.session_state["access_token"] = token
                            st.session_state["user"] = user
                            st.session_state["authenticated"] = True
                            st.session_state["conversation_id"] = None
                            st.session_state["messages"] = []
                            st.rerun()

        st.divider()

        if st.button(
            "Create a new account",
            use_container_width=True,
        ):
            st.session_state["auth_mode"] = "register"
            st.rerun()


def render_registration() -> None:
    st.title("MEA Research Assistant")
    st.caption("Create an account to save your research conversations.")

    _, center, _ = st.columns([1, 1.3, 1])

    with center:
        st.subheader("Create account")

        with st.form("registration_form", clear_on_submit=False):
            full_name = st.text_input(
                "Full name",
                placeholder="Your full name",
            )
            email = st.text_input(
                "Email address",
                placeholder="you@example.com",
            )
            password = st.text_input(
                "Password",
                type="password",
                help="Use at least 12 characters.",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
            )

            submitted = st.form_submit_button(
                "Create account",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            if not full_name.strip() or not email.strip():
                st.error("Enter your name and email address.")
            elif len(password) < 12:
                st.error("Your password must contain at least 12 characters.")
            elif len(password) > 128:
                st.error("Your password must not exceed 128 characters.")
            elif password != confirm_password:
                st.error("The passwords do not match.")
            else:
                payload = {
                    "full_name": full_name.strip(),
                    "email": email.strip(),
                    "password": password,
                }

                with st.spinner("Creating your account..."):
                    data, error = call_auth_api("register", payload)

                if error:
                    st.error(error)
                elif data:
                    st.success("Account created. You can now log in.")
                    st.session_state["auth_mode"] = "login"
                    st.rerun()

        st.divider()

        if st.button("Back to login", use_container_width=True):
            st.session_state["auth_mode"] = "login"
            st.rerun()


# -------------------------------------------------------------------
# Authenticated application
# -------------------------------------------------------------------

def render_authenticated_app() -> None:
    api_base_url = get_api_base_url()
    access_token = st.session_state.get("access_token")

    if not isinstance(access_token, str) or not access_token:
        st.session_state["authenticated"] = False
        st.rerun()

    # Validate the token on app entry and after a browser rerun that
    # has not already established a valid user profile.
    if not st.session_state.get("user"):
        with st.spinner("Checking your session..."):
            user = fetch_current_user(api_base_url, access_token)

        if not user:
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

            st.session_state["authenticated"] = False
            st.warning("Your session has expired. Please log in again.")
            st.rerun()

        st.session_state["user"] = user

    render_sidebar(
        api_base_url=api_base_url,
        access_token=access_token,
    )

    render_chat(
        api_base_url=api_base_url,
        access_token=access_token,
    )


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def main() -> None:
    initialize_session_state()

    if st.session_state.get("authenticated"):
        render_authenticated_app()
    elif st.session_state.get("auth_mode") == "register":
        render_registration()
    else:
        render_login()


if __name__ == "__main__":
    main()
