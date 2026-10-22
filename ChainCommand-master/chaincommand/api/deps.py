"""Authentication and role-based access control dependencies."""

from __future__ import annotations

from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..db import get_db
from ..db.models import Role, User
from ..security import decode_access_token

_bearer = HTTPBearer(auto_error=False)

STAFF_UP = (Role.STAFF, Role.MANAGER, Role.ADMIN)
MANAGER_UP = (Role.MANAGER, Role.ADMIN)
ADMIN_ONLY = (Role.ADMIN,)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"})


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise _unauthorized("Not authenticated")
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, ValueError, KeyError):
        raise _unauthorized("Invalid or expired token") from None
    user = db.get(User, user_id)
    if user is None:
        raise _unauthorized("User no longer exists")
    return user


def require_roles(*roles: str):
    """Dependency factory: ``Depends(require_roles(Role.MANAGER, Role.ADMIN))``."""

    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires one of the roles: {', '.join(roles)}")
        return user

    return checker


require_staff = require_roles(*STAFF_UP)
require_manager = require_roles(*MANAGER_UP)
require_admin = require_roles(*ADMIN_ONLY)
