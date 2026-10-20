"""Tests for RedshiftClient — all redshift_connector calls are mocked."""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from chaincommand.data.schemas import KPISnapshot


@pytest.fixture
def mock_connector():
    mock = MagicMock()
    mock_conn = MagicMock()
    mock.connect.return_value = mock_conn
    with patch.dict(sys.modules, {"redshift_connector": mock}):
        yield mock, mock_conn


@pytest.fixture
def redshift_client(mock_connector):
    from chaincommand.aws.redshift_client import RedshiftClient

    client = RedshiftClient(
        host="test-host",
        port=5439,
        database="testdb",
        user="testuser",
        password="testpass",
        iam_role="arn:aws:iam::123456:role/test",
    )
    return client


class TestCreateTables:
    def test_executes_all_ddl_statements(self, redshift_client, mock_connector):
        from chaincommand.aws.redshift_client import ALL_CREATE_STATEMENTS

        _, mock_conn = mock_connector
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        redshift_client.create_tables()

        assert mock_cursor.execute.call_count == len(ALL_CREATE_STATEMENTS)
        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once()

    def test_ddl_contains_required_tables(self):
        from chaincommand.aws.redshift_client import ALL_CREATE_STATEMENTS

        combined = "\n".join(ALL_CREATE_STATEMENTS)
        assert "kpi_snapshots" in combined
        assert "purchase_orders" in combined
        assert "events" in combined
        assert "products" in combined
        assert "suppliers" in combined


class TestCopyFromS3:
    def test_copy_command_format(self, redshift_client, mock_connector):
        _, mock_conn = mock_connector
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        redshift_client.copy_from_s3("events", "supply-chain/events/2025/01/01/events.jsonl")

        sql_executed = mock_cursor.execute.call_args[0][0]
        assert "COPY events" in sql_executed
        assert "FROM 's3://" in sql_executed
        assert "IAM_ROLE" in sql_executed
        assert "FORMAT AS JSON 'auto'" in sql_executed
        mock_conn.commit.assert_called_once()

    def test_copy_parquet_format_unchanged(self, redshift_client, mock_connector):
        _, mock_conn = mock_connector
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        redshift_client.copy_from_s3("products", "supply-chain/products/file.parquet", file_format="PARQUET")

        sql_executed = mock_cursor.execute.call_args[0][0]
        assert "FORMAT AS PARQUET" in sql_executed


class TestQuery:
    def test_returns_list_of_dicts(self, redshift_client, mock_connector):
        _, mock_conn = mock_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("cycle",), ("otif",), ("fill_rate",)]
        mock_cursor.fetchall.return_value = [
            (1, 0.95, 0.97),
            (2, 0.93, 0.96),
        ]
        mock_conn.cursor.return_value = mock_cursor

        results = redshift_client.query("SELECT cycle, otif, fill_rate FROM kpi_snapshots")

        assert len(results) == 2
        assert results[0] == {"cycle": 1, "otif": 0.95, "fill_rate": 0.97}
        assert results[1]["cycle"] == 2

    def test_empty_result(self, redshift_client, mock_connector):
        _, mock_conn = mock_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("cycle",)]
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        results = redshift_client.query("SELECT cycle FROM kpi_snapshots WHERE 1=0")
        assert results == []


class TestInsertKpiSnapshot:
    def test_inserts_all_fields(self, redshift_client, mock_connector):
        _, mock_conn = mock_connector
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        snapshot = KPISnapshot(
            timestamp=datetime(2025, 1, 15, 12, 0, 0),
            otif=0.95,
            fill_rate=0.97,
            mape=12.5,
            dsi=35.0,
            stockout_count=2,
            total_inventory_value=500000.0,
            carrying_cost=342.47,
            order_cycle_time=7.5,
            perfect_order_rate=0.88,
            inventory_turnover=8.2,
            backorder_rate=0.03,
            supplier_defect_rate=0.02,
        )

        redshift_client.insert_kpi_snapshot(cycle=5, snapshot=snapshot)

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        params = mock_cursor.execute.call_args[0][1]
        assert "INSERT INTO kpi_snapshots" in sql
        assert params[0] == 5  # cycle
        assert params[2] == 0.95  # otif
        assert len(params) == 14
        mock_conn.commit.assert_called_once()


class TestClose:
    def test_close_connection(self, redshift_client, mock_connector):
        _, mock_conn = mock_connector
        # Force connection to be created
        redshift_client._connect()

        redshift_client.close()
        mock_conn.close.assert_called_once()
        assert redshift_client._conn is None

    def test_close_no_connection(self):
        from chaincommand.aws.redshift_client import RedshiftClient

        client = RedshiftClient(host="x", port=5439, database="x", user="x", password="x")
        client._conn = None
        client.close()  # Should not raise
