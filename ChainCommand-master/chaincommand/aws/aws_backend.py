"""AWSBackend — assembles S3, Redshift, Athena, and QuickSight clients."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import pandas as pd

from ..config import settings
from ..data.schemas import KPISnapshot
from ..kpi.engine import ALLOWED_KPI_METRICS
from ..utils.logging_config import get_logger
from .athena_client import AthenaClient
from .backend import PersistenceBackend
from .quicksight_client import QuickSightClient
from .redshift_client import RedshiftClient
from .s3_client import S3Client

log = get_logger(__name__)


class AWSBackend(PersistenceBackend):
    """Full AWS persistence backend using S3, Redshift, Athena, and QuickSight."""

    def __init__(self) -> None:
        self._s3: S3Client | None = None
        self._redshift: RedshiftClient | None = None
        self._athena: AthenaClient | None = None
        self._quicksight: QuickSightClient | None = None

    async def setup(self) -> None:
        """Initialize all AWS clients and create required tables."""
        log.info("aws_backend_setup_start")
        try:
            self._s3 = S3Client()
            self._redshift = RedshiftClient()
            self._athena = AthenaClient()
            self._quicksight = QuickSightClient()

            # Redshift: create tables (sync I/O — run off event loop)
            await asyncio.to_thread(self._redshift.create_tables)

            # Athena: create database + external tables
            await self._athena.create_database()
            await self._athena.create_external_tables()

            log.info("aws_backend_setup_complete")
        except Exception:
            log.exception("aws_backend_setup_failed")
            raise

    async def teardown(self) -> None:
        """Close all AWS client connections."""
        if self._redshift:
            await asyncio.to_thread(self._redshift.close)
        # boto3 clients (S3, Athena, QuickSight) don't hold persistent
        # connections, but nullify references for cleanliness.
        self._s3 = None
        self._athena = None
        self._quicksight = None
        log.info("aws_backend_teardown")

    async def persist_cycle(
        self,
        cycle: int,
        kpi: KPISnapshot,
        events: list,
        pos: list,
        products: list,
        suppliers: list,
    ) -> None:
        """Persist one cycle's data to S3 and Redshift."""
        if self._s3 is None or self._redshift is None:
            raise RuntimeError(
                "AWSBackend.persist_cycle called before setup() — "
                "S3 or Redshift client is None"
            )

        now = datetime.now(timezone.utc)
        date_path = f"{now.year:04d}/{now.month:02d}/{now.day:02d}"
        prefix = settings.aws_s3_prefix.rstrip("/")

        # ── S3: upload JSONL (sync boto3 calls — run off event loop) ──
        # KPI snapshot
        kpi_key = f"{prefix}/kpi_snapshots/{date_path}/cycle_{cycle}.jsonl"
        kpi_data = kpi.model_dump()
        kpi_data["cycle"] = cycle
        await asyncio.to_thread(self._s3.upload_jsonl, [kpi_data], kpi_key)

        # Events
        if events:
            events_key = f"{prefix}/events/{date_path}/cycle_{cycle}.jsonl"
            event_records = [
                e.model_dump() if hasattr(e, "model_dump") else e for e in events
            ]
            await asyncio.to_thread(self._s3.upload_jsonl, event_records, events_key)

        # Purchase orders
        if pos:
            pos_key = f"{prefix}/purchase_orders/{date_path}/cycle_{cycle}.jsonl"
            po_records = [
                p.model_dump() if hasattr(p, "model_dump") else p for p in pos
            ]
            await asyncio.to_thread(self._s3.upload_jsonl, po_records, pos_key)

        # Log when products/suppliers are provided but not persisted
        if products:
            log.warning(
                "aws_persist_cycle_products_not_persisted",
                cycle=cycle, count=len(products),
            )
        if suppliers:
            log.warning(
                "aws_persist_cycle_suppliers_not_persisted",
                cycle=cycle, count=len(suppliers),
            )

        # ── Redshift: direct INSERT for KPI (fast, single row) ──
        await asyncio.to_thread(self._redshift.insert_kpi_snapshot, cycle, kpi)

        log.info("aws_persist_cycle", cycle=cycle)

    async def persist_demand_history(self, df: pd.DataFrame) -> None:
        """Upload demand history DataFrame to S3 as Parquet."""
        if self._s3 is None:
            raise RuntimeError(
                "AWSBackend.persist_demand_history called before setup() — "
                "S3 client is None"
            )
        prefix = settings.aws_s3_prefix.rstrip("/")
        key = f"{prefix}/demand_history/full_history.parquet"
        await asyncio.to_thread(self._s3.upload_dataframe, df, key)
        log.info("aws_persist_demand_history", rows=len(df))

    async def query_kpi_trend(self, metric: str, days: int) -> list:
        """Query KPI trend from Redshift."""
        if self._redshift is None:
            raise RuntimeError(
                "AWSBackend.query_kpi_trend called before setup() — "
                "Redshift client is None"
            )
        if metric not in ALLOWED_KPI_METRICS:
            return []

        # Belt-and-suspenders: regex check even after allowlist
        if not re.match(r'^[a-z_]+$', metric):
            log.warning("query_kpi_trend_rejected_metric", metric=metric)
            return []

        safe_days = max(1, min(int(days), 365))
        sql = (
            f"SELECT cycle, timestamp, {metric} "  # noqa: S608
            f"FROM kpi_snapshots "
            f"WHERE timestamp >= DATEADD(day, -{safe_days}, GETDATE()) "
            f"ORDER BY cycle"
        )
        return await asyncio.to_thread(self._redshift.query, sql)

    async def query_events(self, event_type: str, limit: int) -> list:
        """Query events from Athena (ad-hoc on S3).

        Uses parameterized query to prevent SQL injection.
        """
        if self._athena is None:
            raise RuntimeError(
                "AWSBackend.query_events called before setup() — "
                "Athena client is None"
            )
        safe_limit = max(1, min(int(limit), 500))
        sql = (
            "SELECT event_id, timestamp, event_type, severity, source_agent, description "
            "FROM events "
            "WHERE event_type = ? "
            "ORDER BY timestamp DESC "
            "LIMIT ?"
        )
        return await self._athena.run_query(sql, params=[event_type, safe_limit])
