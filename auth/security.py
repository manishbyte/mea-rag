
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import (
    HashingError,
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
from dotenv import load_dotenv
from jwt.exceptions import InvalidTokenError

load_dotenv()

JWT_ALGORITHM = "HS256"
JWT_ISSUER = os.getenv("JWT_ISSUER", "mea-rag-api")
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip()

if len(JWT_SECRET_KEY) < 32:
    raise RuntimeError(
        "JWT_SECRET_KEY must contain at least 32 characters. "
        "Generate a cryptographically random secret and add it to .env."
    )

if ACCESS_TOKEN_EXPIRE_MINUTES < 1:
    raise RuntimeError(
        "ACCESS_TOKEN_EXPIRE_MINUTES must be at least 1."
    )

password_hasher = PasswordHasher()


class AuthenticationError(Exception):
    """Raised when a credential or token is invalid."""


def hash_password(password: str) -> str:
    """Return an Argon2id password hash."""
    if not password:
        raise ValueError("Password cannot be empty.")

    try:
        return password_hasher.hash(password)
    except HashingError as exc:
        raise RuntimeError("Password hashing failed.") from exc


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """Verify a password without exposing hashing internals."""
    if not plain_password or not hashed_password:
        return False

    try:
        return password_hasher.verify(
            hashed_password,
            plain_password,
        )
    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHashError,
    ):
        return False


def password_needs_rehash(hashed_password: str) -> bool:
    """Check whether a stored password hash needs updating."""
    try:
        return password_hasher.check_needs_rehash(hashed_password)
    except (VerificationError, InvalidHashError):
        return True


def create_access_token(
    user_id: uuid.UUID | str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token for a user."""
    now = datetime.now(timezone.utc)

    if expires_delta is None:
        expires_delta = timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    if expires_delta.total_seconds() <= 0:
        raise ValueError("Token lifetime must be positive.")

    expires_at = now + expires_delta

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "iat": now,
        "exp": expires_at,
        "iss": JWT_ISSUER,
        "type": "access",
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Verify the JWT signature, issuer, and expiration.

    Raises AuthenticationError for invalid, expired, or
    incorrectly configured tokens.
    """
    if not token:
        raise AuthenticationError("Invalid access token.")

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            options={
                "require": ["sub", "exp", "iat", "iss", "type"],
            },
        )

        if payload.get("type") != "access":
            raise AuthenticationError("Invalid access token.")

        subject = payload.get("sub")

        if not isinstance(subject, str):
            raise AuthenticationError("Invalid access token.")

        # Validate the subject as a UUID because user IDs use UUIDs.
        uuid.UUID(subject)

        return payload

    except (InvalidTokenError, ValueError) as exc:
        raise AuthenticationError(
            "Invalid or expired access token."
        ) from exc
