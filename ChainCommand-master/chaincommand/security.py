"""Password hashing (Argon2) and JWT access tokens (report section 18)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from .config import settings

_hasher = PasswordHasher()
_ALGORITHM = "HS256"
_DEFAULT_SECRET = "dev-jwt-secret-change-me-0123456789abcdef"  # noqa: S105


def hash_password(password: str) -> str:
    """Return a salted Argon2id hash. The plain password is never stored."""
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def _secret() -> str:
    secret = settings.jwt_secret.get_secret_value()
    if secret == _DEFAULT_SECRET and settings.env in ("production", "staging"):
        raise RuntimeError("CC_JWT_SECRET must be set to a strong secret outside development.")
    return secret


def create_access_token(user_id: int, role: str, expires_minutes: Optional[int] = None) -> tuple[str, int]:
    """Return (token, lifetime_in_seconds)."""
    minutes = settings.jwt_expire_minutes if expires_minutes is None else expires_minutes
    now = datetime.now(UTC)
    payload = {"sub": str(user_id), "role": role, "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(payload, _secret(), algorithm=_ALGORITHM), minutes * 60


def decode_access_token(token: str) -> dict[str, Any]:
    """Raises ``jwt.PyJWTError`` if the token is invalid, tampered with or expired."""
    return jwt.decode(token, _secret(), algorithms=[_ALGORITHM], options={"require": ["exp", "sub"]})
