
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.security import (
    AuthenticationError,
    decode_access_token,
)
from database.models import User
from database.session import get_db


# Clients should send:
# Authorization: Bearer <access_token>
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
    auto_error=False,
)


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    Validate the access token and load its active user.

    Raises HTTP 401 if the token is missing, invalid, expired,
    or its user no longer exists or is inactive.
    """
    authentication_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise authentication_error

    try:
        payload = decode_access_token(token)
        user_id = uuid.UUID(payload["sub"])
    except (AuthenticationError, ValueError, KeyError, TypeError):
        raise authentication_error from None

    result = await session.execute(
        select(User).where(
            User.id == user_id,
            User.is_active.is_(True),
        )
    )

    user = result.scalar_one_or_none()

    if user is None:
        raise authentication_error

    return user


# Reusable type aliases for FastAPI route signatures.
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]
AuthenticatedUser = Annotated[User, Depends(get_current_user)]
