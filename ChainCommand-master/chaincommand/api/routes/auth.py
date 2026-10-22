"""Login and user management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...db import get_db
from ...db.models import User
from ...security import create_access_token, hash_password, verify_password
from ..deps import get_current_user, require_admin
from ..schemas import LoginRequest, TokenResponse, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

# Verified against when the e-mail is unknown, so response time does not reveal which e-mails exist.
_DUMMY_HASH = hash_password("not-a-real-password")


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(func.lower(User.email) == body.email.strip().lower()))
    password_ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not password_ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    token, lifetime = create_access_token(user.user_id, user.role)
    return TokenResponse(access_token=token, expires_in=lifetime, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user


@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    return db.scalars(select(User).order_by(User.user_id)).all()


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(body: UserCreate, db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    if db.scalar(select(User).where(func.lower(User.email) == body.email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "A user with this email already exists")
    user = User(name=body.name, email=body.email, role=body.role, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return user
