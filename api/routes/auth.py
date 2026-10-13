
from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from api.dependencies import AuthenticatedUser, DatabaseSession
from auth.schemas import (
    AuthMessageResponse,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from auth.security import (
    create_access_token,
    hash_password,
    password_needs_rehash,
    verify_password,
)
from database.models import User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


def _user_response(user: User) -> UserResponse:
    """Convert a database user into the public response schema."""
    return UserResponse.model_validate(user)


def _create_token_response(user: User) -> TokenResponse:
    """Create an access token and return the user's public profile."""
    from auth.security import ACCESS_TOKEN_EXPIRE_MINUTES

    token = create_access_token(
        user_id=str(user.id),
        expires_delta=timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        ),
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=_user_response(user),
    )


# -------------------------------------------------------------------
# POST /api/auth/register
# -------------------------------------------------------------------

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: UserRegisterRequest,
    session: DatabaseSession,
) -> UserResponse:
    email = str(payload.email).strip().lower()

    existing_result = await session.execute(
        select(User.id).where(User.email == email)
    )

    if existing_result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    full_name = (
        payload.full_name.strip()
        if payload.full_name is not None
        else None
    )

    user = User(
        email=email,
        password_hash=hash_password(
            payload.password.get_secret_value()
        ),
        full_name=full_name,
        is_active=True,
        is_verified=False,
    )

    session.add(user)

    try:
        await session.commit()
        await session.refresh(user)
    except IntegrityError:
        await session.rollback()
        # The unique database constraint also protects against
        # simultaneous registration attempts using the same email.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from None
    except Exception:
        await session.rollback()
        logger.exception("Failed to register a user.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again.",
        ) from None

    logger.info("A user account was registered.")
    return _user_response(user)


# -------------------------------------------------------------------
# POST /api/auth/login
# -------------------------------------------------------------------

@router.post(
    "/login",
    response_model=TokenResponse,
)
async def login(
    payload: UserLoginRequest,
    session: DatabaseSession,
) -> TokenResponse:
    email = str(payload.email).strip().lower()

    result = await session.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()

    # Use the same public error for an unknown email and a bad password.
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    password = payload.password.get_secret_value()

    try:
        password_valid = verify_password(
            password,
            user.password_hash,
        )
    except Exception:
        logger.exception("Password verification failed.")
        password_valid = False

    if not password_valid or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Upgrade the password hash if the Argon2 configuration changes.
    try:
        if password_needs_rehash(user.password_hash):
            user.password_hash = hash_password(password)
            await session.commit()
    except Exception:
        await session.rollback()
        logger.exception("Could not upgrade a user's password hash.")
        # A successful password check should not be converted into
        # a login failure solely because a hash upgrade failed.

    return _create_token_response(user)


# -------------------------------------------------------------------
# GET /api/auth/me
# -------------------------------------------------------------------

@router.get(
    "/me",
    response_model=UserResponse,
)
async def get_me(
    current_user: AuthenticatedUser,
) -> UserResponse:
    """Return the profile belonging to the authenticated user."""
    return _user_response(current_user)


# -------------------------------------------------------------------
# POST /api/auth/logout
# -------------------------------------------------------------------

@router.post(
    "/logout",
    response_model=AuthMessageResponse,
)
async def logout(
    current_user: AuthenticatedUser,
) -> AuthMessageResponse:
    """
    Acknowledge logout.

    This API uses stateless JWT access tokens, so this endpoint does
    not revoke a token already issued. The client should discard it.
    """
    return AuthMessageResponse(
        message="Logged out. Please discard your access token."
    )
