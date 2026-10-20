"""Tests for AthenaClient — all boto3 calls are mocked."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_boto3():
    mock = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock}):
        yield mock


@pytest.fixture
def athena_client(mock_boto3):
    import importlib

    import chaincommand.aws.athena_client as mod

    importlib.reload(mod)
    client = mod.AthenaClient(
        database="testdb",
        output_location="s3://test-bucket/athena-results/",
        region="us-east-1",
    )
    return client


def _mock_query_success(athena_client, execution_id="test-exec-123"):
    """Set up mocks for a successful query execution."""
    athena = athena_client._client
    athena.start_query_execution.return_value = {"QueryExecutionId": execution_id}
    athena.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
    }
    return athena


class TestCreateDatabase:
    @pytest.mark.asyncio
    async def test_creates_database(self, athena_client):
        athena = _mock_query_success(athena_client)

        result = await athena_client.create_database()

        athena.start_query_execution.assert_called_once()
        sql = athena.start_query_execution.call_args[1]["QueryString"]
        assert "CREATE DATABASE IF NOT EXISTS testdb" in sql
        assert result == "test-exec-123"


class TestCreateExternalTables:
    @pytest.mark.asyncio
    async def test_creates_all_tables(self, athena_client):
        from chaincommand.aws.athena_client import ALL_EXTERNAL_TABLES

        athena = _mock_query_success(athena_client)

        result = await athena_client.create_external_tables()

        assert len(result) == len(ALL_EXTERNAL_TABLES)
        assert athena.start_query_execution.call_count == len(ALL_EXTERNAL_TABLES)

    @pytest.mark.asyncio
    async def test_external_table_ddl_contains_required_tables(self, athena_client):
        athena = _mock_query_success(athena_client)

        await athena_client.create_external_tables()

        calls = athena.start_query_execution.call_args_list
        all_sql = " ".join(c[1]["QueryString"] for c in calls)
        assert "demand_history" in all_sql
        assert "kpi_snapshots" in all_sql
        assert "events" in all_sql
        assert "STORED AS PARQUET" in all_sql
        assert "ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'" in all_sql
        assert "STORED AS TEXTFILE" in all_sql


class TestRunQuery:
    @pytest.mark.asyncio
    async def test_returns_parsed_results(self, athena_client):
        athena = _mock_query_success(athena_client)
        athena.get_query_results.return_value = {
            "ResultSet": {
                "Rows": [
                    {"Data": [{"VarCharValue": "event_type"}, {"VarCharValue": "count"}]},
                    {"Data": [{"VarCharValue": "low_stock"}, {"VarCharValue": "15"}]},
                    {"Data": [{"VarCharValue": "anomaly"}, {"VarCharValue": "3"}]},
                ]
            }
        }

        results = await athena_client.run_query("SELECT event_type, count(*) FROM events GROUP BY 1")

        assert len(results) == 2
        assert results[0] == {"event_type": "low_stock", "count": "15"}
        assert results[1] == {"event_type": "anomaly", "count": "3"}

    @pytest.mark.asyncio
    async def test_empty_results(self, athena_client):
        athena = _mock_query_success(athena_client)
        athena.get_query_results.return_value = {"ResultSet": {"Rows": []}}

        results = await athena_client.run_query("SELECT * FROM empty_table")
        assert results == []


class TestWaitForQuery:
    @pytest.mark.asyncio
    async def test_polling_until_succeeded(self, athena_client):
        athena = athena_client._client
        athena.get_query_execution.side_effect = [
            {"QueryExecution": {"Status": {"State": "RUNNING"}}},
            {"QueryExecution": {"Status": {"State": "RUNNING"}}},
            {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}},
        ]

        with patch("chaincommand.aws.athena_client.asyncio.sleep", new_callable=AsyncMock):
            state = await athena_client._wait_for_query("poll-test")

        assert state == "SUCCEEDED"
        assert athena.get_query_execution.call_count == 3

    @pytest.mark.asyncio
    async def test_query_failure_raises(self, athena_client):
        athena = athena_client._client
        athena.get_query_execution.return_value = {
            "QueryExecution": {
                "Status": {
                    "State": "FAILED",
                    "StateChangeReason": "Syntax error",
                }
            }
        }

        with pytest.raises(RuntimeError, match="FAILED"):
            await athena_client._wait_for_query("fail-test")

    @pytest.mark.asyncio
    async def test_timeout_raises(self, athena_client):
        athena = athena_client._client
        athena.get_query_execution.return_value = {
            "QueryExecution": {"Status": {"State": "RUNNING"}}
        }

        with patch("chaincommand.aws.athena_client.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError):
                await athena_client._wait_for_query("timeout-test", max_wait=0.01, interval=0.005)


class TestGetQueryResults:
    @pytest.mark.asyncio
    async def test_header_row_excluded(self, athena_client):
        athena = athena_client._client
        athena.get_query_results.return_value = {
            "ResultSet": {
                "Rows": [
                    {"Data": [{"VarCharValue": "col1"}, {"VarCharValue": "col2"}]},
                    {"Data": [{"VarCharValue": "a"}, {"VarCharValue": "b"}]},
                ]
            }
        }

        results = await athena_client.get_query_results("exec-123")
        assert len(results) == 1
        assert results[0] == {"col1": "a", "col2": "b"}

    @pytest.mark.asyncio
    async def test_pagination_fetches_all_pages(self, athena_client):
        """Test that get_query_results follows NextToken pagination."""
        athena = athena_client._client
        athena.get_query_results.side_effect = [
            {
                "ResultSet": {
                    "Rows": [
                        {"Data": [{"VarCharValue": "col1"}]},
                        {"Data": [{"VarCharValue": "page1_row1"}]},
                    ]
                },
                "NextToken": "token-page-2",
            },
            {
                "ResultSet": {
                    "Rows": [
                        {"Data": [{"VarCharValue": "page2_row1"}]},
                        {"Data": [{"VarCharValue": "page2_row2"}]},
                    ]
                },
            },
        ]

        results = await athena_client.get_query_results("paginated-exec")
        assert len(results) == 3
        assert results[0] == {"col1": "page1_row1"}
        assert results[1] == {"col1": "page2_row1"}
        assert results[2] == {"col1": "page2_row2"}
        assert athena.get_query_results.call_count == 2
