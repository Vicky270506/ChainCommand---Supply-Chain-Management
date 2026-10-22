"""Persistence backend abstraction (Strategy Pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..config import settings
from ..data.schemas import KPISnapshot
from ..utils.logging_config import get_logger

log = get_logger(__name__)


class PersistenceBackend(ABC):
    """Abstract base class for persistence backends."""

    @abstractmethod
    async def setup(self) -> None:
        """One-time initialization (create tables, external tables, etc.)."""

    @abstractmethod
    async def teardown(self) -> None:
        """Clean shutdown (close connections, etc.)."""

    @abstractmethod
    async def persist_cycle(
        self,
        cycle: int,
        kpi: KPISnapshot,
        events: list,
        pos: list,
        products: list,
        suppliers: list,
    ) -> None:
        """Persist data from one decision cycle."""

    @abstractmethod
    async def persist_demand_history(self, df: pd.DataFrame) -> None:
        """Persist demand history DataFrame."""

    @abstractmethod
    async def query_kpi_trend(self, metric: str, days: int) -> list:
        """Query KPI trend from persistent storage."""

    @abstractmethod
    async def query_events(self, event_type: str, limit: int) -> list:
        """Query events from persistent storage."""


class NullBackend(PersistenceBackend):
    """No-op backend used when AWS is disabled."""

    async def setup(self) -> None:
        log.debug("null_backend_setup")

    async def teardown(self) -> None:
        log.debug("null_backend_teardown")

    async def persist_cycle(
        self,
        cycle: int,
        kpi: KPISnapshot,
        events: list,
        pos: list,
        products: list,
        suppliers: list,
    ) -> None:
        pass

    async def persist_demand_history(self, df: pd.DataFrame) -> None:
        pass

    async def query_kpi_trend(self, metric: str, days: int) -> list:
        return []

    async def query_events(self, event_type: str, limit: int) -> list:
        return []


def get_backend() -> PersistenceBackend:
    """Factory: return AWSBackend if enabled, else NullBackend."""
    if settings.aws_enabled:
        from .aws_backend import AWSBackend

        log.info("aws_backend_selected")
        return AWSBackend()
    log.debug("null_backend_selected")
    return NullBackend()
