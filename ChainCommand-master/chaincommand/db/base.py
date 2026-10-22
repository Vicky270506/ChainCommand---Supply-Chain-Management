"""Database engine / session helpers.

The database is chosen with the CC_DATABASE_URL setting:
    sqlite:///./inventory.db                                   (default, zero setup)
    postgresql+psycopg://user:password@localhost:5432/dbname   (PostgreSQL)
"""

from __future__ import annotations

from typing import Iterator, Optional

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings
from .models import Base

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:  # noqa: ARG001
    """SQLite ignores foreign keys unless asked; PostgreSQL always enforces them."""
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def make_engine(url: Optional[str] = None) -> Engine:
    url = url or settings.database_url
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


def get_db() -> Iterator[Session]:
    """FastAPI dependency: ``db: Session = Depends(get_db)``."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def init_db(engine: Optional[Engine] = None) -> None:
    """Create all tables that do not exist yet."""
    Base.metadata.create_all(engine or get_engine())


def reset_db(engine: Optional[Engine] = None) -> None:
    """Drop and recreate all tables (destroys data)."""
    engine = engine or get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
