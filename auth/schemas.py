
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
)


class UserRegisterRequest(BaseModel):
    """Request body for creating a user account."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    email: EmailStr

    password: SecretStr = Field(
        ...,
        min_length=12,
        max_length=128,
        description="Password must contain at least 12 characters.",
    )

    full_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Full name cannot be blank.")

        return value


class UserLoginRequest(BaseModel):
    """Request body for logging into an existing account."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    email: EmailStr

    password: SecretStr = Field(
        ...,
        min_length=1,
        max_length=128,
    )


class UserResponse(BaseModel):
    """Safe public representation of an authenticated user."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    full_name: str | None = None
    is_active: bool
    is_verified: bool
    created_at: datetime


class TokenResponse(BaseModel):
    """Response returned after successful authentication."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(
        ...,
        gt=0,
        description="Access-token lifetime in seconds.",
    )
    user: UserResponse


class AuthMessageResponse(BaseModel):
    """Generic response for authentication-related operations."""

    message: str


class TokenPayload(BaseModel):
    """Validated claims extracted from a JWT."""

    sub: str
    exp: int
    iat: int | None = None
    token_type: str = "access"
