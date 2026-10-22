"""Redshift operations for ChainCommand data persistence."""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from ..data.schemas import KPISnapshot
from ..utils.logging_config import get_logger

log = get_logger(__name__)

# ── Table DDL ────────────────────────────────────────────

CREATE_KPI_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS kpi_snapshots (
    id              BIGINT IDENTITY(1,1) PRIMARY KEY,
    cycle           INT NOT NULL,
    timestamp       TIMESTAMP NOT NULL DEFAULT GETDATE(),
    otif            FLOAT,
    fill_rate       FLOAT,
    mape            FLOAT,
    dsi             FLOAT,
    stockout_count  INT,
    total_inventory_value FLOAT,
    carrying_cost   FLOAT,
    order_cycle_time FLOAT,
    perfect_order_rate FLOAT,
    inventory_turnover FLOAT,
    backorder_rate  FLOAT,
    supplier_defect_rate FLOAT
)
SORTKEY(timestamp);
"""

CREATE_PURCHASE_ORDERS = """
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id           VARCHAR(64) PRIMARY KEY,
    supplier_id     VARCHAR(64),
    product_id      VARCHAR(64) DISTKEY,
    quantity        FLOAT,
    unit_cost       FLOAT,
    total_cost      FLOAT,
    status          VARCHAR(32),
    created_at      TIMESTAMP,
    expected_delivery TIMESTAMP
)
SORTKEY(created_at);
"""

CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    event_id        VARCHAR(64) PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL DEFAULT GETDATE(),
    event_type      VARCHAR(128),
    severity        VARCHAR(32),
    source_agent    VARCHAR(128),
    description     VARCHAR(65535),
    data            VARCHAR(65535),
    resolved        BOOLEAN DEFAULT FALSE,
    resolution      VARCHAR(65535)
);
"""

CREATE_PRODUCTS = """
CREATE TABLE IF NOT EXISTS products (
    product_id      VARCHAR(64),
    name            VARCHAR(256),
    category        VARCHAR(64),
    unit_cost       FLOAT,
    selling_price   FLOAT,
    lead_time_days  INT,
    min_order_qty   INT,
    current_stock   FLOAT,
    reorder_point   FLOAT,
    safety_stock    FLOAT,
    daily_demand_avg FLOAT,
    snapshot_timestamp TIMESTAMP NOT NULL DEFAULT GETDATE()
);
"""

CREATE_SUPPLIERS = """
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id     VARCHAR(64),
    name            VARCHAR(256),
    reliability_score FLOAT,
    lead_time_mean  FLOAT,
    lead_time_std   FLOAT,
    cost_multiplier FLOAT,
    capacity        FLOAT,
    defect_rate     FLOAT,
    on_time_rate    FLOAT,
    is_active       BOOLEAN,
    snapshot_timestamp TIMESTAMP NOT NULL DEFAULT GETDATE()
);
"""

ALL_CREATE_STATEMENTS = [
    CREATE_KPI_SNAPSHOTS,
    CREATE_PURCHASE_ORDERS,
    CREATE_EVENTS,
    CREATE_PRODUCTS,
    CREATE_SUPPLIERS,
]


class RedshiftClient:
    """Encapsulates Redshift connection and query operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        iam_role: Optional[str] = None,
    ) -> None:
        self._host = host or settings.aws_redshift_host
        self._port = port or settings.aws_redshift_port
        self._database = database or settings.aws_redshift_db
        self._user = user or settings.aws_redshift_user
        self._password = password or settings.aws_redshift_password.get_secret_value()
        self._iam_role = iam_role or settings.aws_redshift_iam_role
        if self._iam_role and not re.match(r'^arn:aws:iam::\d+:role/[\w+=,.@\-/]+$', self._iam_role):
            raise ValueError(f"Invalid IAM role ARN format: {self._iam_role}")
        self._conn: Any = None
        self._conn_lock = threading.Lock()

    def _connect(self) -> Any:
        """Establish (or return cached) Redshift connection with health check.

        Thread-safe: uses a lock to prevent concurrent connection mutations
        when called from multiple asyncio.to_thread workers.
        """
        with self._conn_lock:
            return self._connect_unlocked()

    def _connect_unlocked(self) -> Any:
        """Inner connection logic (caller must hold _conn_lock)."""
        if self._conn is not None:
            # Verify the cached connection is still alive
            try:
                cur = self._conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
            except Exception:
                log.warning("redshift_connection_stale, reconnecting")
                try:
                    self._conn.close()
                except Exception:  # noqa: S110
                    pass
                self._conn = None

        if self._conn is None:
            import redshift_connector

            self._conn = redshift_connector.connect(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password,
            )
        return self._conn

    def close(self) -> None:
        """Close the connection."""
        with self._conn_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def create_tables(self) -> None:
        """Create all required tables if they don't exist."""
        conn = self._connect()
        cursor = conn.cursor()
        try:
            for ddl in ALL_CREATE_STATEMENTS:
                cursor.execute(ddl)
            conn.commit()
            log.info("redshift_tables_created")
        finally:
            cursor.close()

    _ALLOWED_TABLES = {"kpi_snapshots", "purchase_orders", "events", "products", "suppliers"}
    _ALLOWED_FORMATS = {"JSON", "PARQUET", "CSV"}

    def copy_from_s3(self, table: str, s3_key: str, file_format: str = "JSON") -> None:
        """Execute a COPY command to load data from S3."""
        if table not in self._ALLOWED_TABLES:
            raise ValueError(f"Invalid table: {table}")
        if file_format.upper() not in self._ALLOWED_FORMATS:
            raise ValueError(f"Invalid format: {file_format}")
        if not re.match(r'^[\w./\-]+$', s3_key):
            raise ValueError(f"Invalid S3 key: {s3_key}")

        format_clause = file_format.upper()
        if format_clause == "JSON":
            format_clause = "JSON 'auto'"

        sql = f"""
            COPY {table}
            FROM 's3://{settings.aws_s3_bucket}/{s3_key}'
            IAM_ROLE '{self._iam_role}'
            FORMAT AS {format_clause}
            TIMEFORMAT 'auto'
            TRUNCATECOLUMNS
            BLANKSASNULL
            EMPTYASNULL;
        """
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            conn.commit()
            log.info("redshift_copy", table=table, s3_key=s3_key)
        finally:
            cursor.close()

    def query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results as list of dicts."""
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params or ())
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return [dict(zip(columns, row, strict=False)) for row in rows]
        finally:
            cursor.close()

    def insert_kpi_snapshot(self, cycle: int, snapshot: KPISnapshot) -> None:
        """Direct INSERT of a KPI snapshot row."""
        sql = """
            INSERT INTO kpi_snapshots (
                cycle, timestamp, otif, fill_rate, mape, dsi,
                stockout_count, total_inventory_value, carrying_cost,
                order_cycle_time, perfect_order_rate, inventory_turnover,
                backorder_rate, supplier_defect_rate
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, (
                cycle,
                snapshot.timestamp.isoformat(),
                snapshot.otif,
                snapshot.fill_rate,
                snapshot.mape,
                snapshot.dsi,
                snapshot.stockout_count,
                snapshot.total_inventory_value,
                snapshot.carrying_cost,
                snapshot.order_cycle_time,
                snapshot.perfect_order_rate,
                snapshot.inventory_turnover,
                snapshot.backorder_rate,
                snapshot.supplier_defect_rate,
            ))
            conn.commit()
            log.info("redshift_insert_kpi", cycle=cycle)
        finally:
            cursor.close()
