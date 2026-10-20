"""Tests for QuickSightClient — all boto3 calls are mocked."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_boto3():
    mock = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock}):
        yield mock


@pytest.fixture
def qs_client(mock_boto3):
    import importlib

    import chaincommand.aws.quicksight_client as mod

    importlib.reload(mod)
    client = mod.QuickSightClient(account_id="123456789012", region="us-east-1")
    return client


class TestCreateDataSource:
    def test_athena_source(self, qs_client):
        qs_client._client.create_data_source.return_value = {
            "Arn": "arn:aws:quicksight:us-east-1:123456789012:datasource/test",
            "CreationStatus": "CREATION_SUCCESSFUL",
        }

        result = qs_client.create_data_source(
            name="Athena Source",
            source_type="athena",
            config={"workgroup": "primary"},
        )

        qs_client._client.create_data_source.assert_called_once()
        call_kwargs = qs_client._client.create_data_source.call_args[1]
        assert call_kwargs["AwsAccountId"] == "123456789012"
        assert call_kwargs["Type"] == "ATHENA"
        assert "AthenaParameters" in call_kwargs["DataSourceParameters"]
        assert result["status"] == "CREATION_SUCCESSFUL"

    def test_redshift_source(self, qs_client):
        qs_client._client.create_data_source.return_value = {
            "Arn": "arn:aws:quicksight:us-east-1:123456789012:datasource/test",
            "CreationStatus": "CREATION_SUCCESSFUL",
        }

        qs_client.create_data_source(
            name="Redshift Source",
            source_type="redshift",
            config={"host": "cluster.abc.redshift.amazonaws.com", "port": 5439, "database": "testdb"},
        )

        call_kwargs = qs_client._client.create_data_source.call_args[1]
        assert call_kwargs["Type"] == "REDSHIFT"
        assert "RedshiftParameters" in call_kwargs["DataSourceParameters"]
        params = call_kwargs["DataSourceParameters"]["RedshiftParameters"]
        assert params["Host"] == "cluster.abc.redshift.amazonaws.com"


class TestCreateDataset:
    def test_creates_dataset_with_custom_sql(self, qs_client):
        qs_client._client.create_data_set.return_value = {
            "Arn": "arn:aws:quicksight:us-east-1:123456789012:dataset/test",
            "Status": "CREATION_SUCCESSFUL",
        }

        result = qs_client.create_dataset(
            name="KPI Trends",
            source_id="arn:aws:quicksight:us-east-1:123456789012:datasource/redshift",
            sql="SELECT * FROM kpi_snapshots",
        )

        qs_client._client.create_data_set.assert_called_once()
        call_kwargs = qs_client._client.create_data_set.call_args[1]
        assert call_kwargs["ImportMode"] == "DIRECT_QUERY"
        custom_sql = call_kwargs["PhysicalTableMap"]["main"]["CustomSql"]
        assert custom_sql["SqlQuery"] == "SELECT * FROM kpi_snapshots"
        assert "dataset_id" in result

    def test_creates_dataset_accepting_data_source_id(self, qs_client):
        qs_client._client.create_data_set.return_value = {
            "Arn": "arn:aws:quicksight:us-east-1:123456789012:dataset/test",
            "Status": "CREATION_SUCCESSFUL",
        }

        qs_client.create_dataset(
            name="KPI Trends",
            source_id="redshift-source-id",
            sql="SELECT * FROM kpi_snapshots",
        )

        custom_sql = qs_client._client.create_data_set.call_args[1]["PhysicalTableMap"]["main"]["CustomSql"]
        assert custom_sql["DataSourceArn"] == (
            "arn:aws:quicksight:us-east-1:123456789012:datasource/redshift-source-id"
        )


class TestCreateDashboard:
    def test_creates_dashboard_with_dataset_refs(self, qs_client):
        qs_client._client.create_dashboard.return_value = {
            "Arn": "arn:aws:quicksight:us-east-1:123456789012:dashboard/test",
            "CreationStatus": "CREATION_SUCCESSFUL",
        }

        result = qs_client.create_dashboard(
            name="Supply Chain Overview",
            dataset_ids=["arn:ds-1", "arn:ds-2"],
        )

        qs_client._client.create_dashboard.assert_called_once()
        call_kwargs = qs_client._client.create_dashboard.call_args[1]
        refs = call_kwargs["SourceEntity"]["SourceTemplate"]["DataSetReferences"]
        assert len(refs) == 2
        assert refs[0]["DataSetArn"] == "arn:ds-1"
        assert "dashboard_id" in result

    def test_creates_dashboard_accepting_dataset_ids(self, qs_client):
        qs_client._client.create_dashboard.return_value = {
            "Arn": "arn:aws:quicksight:us-east-1:123456789012:dashboard/test",
            "CreationStatus": "CREATION_SUCCESSFUL",
        }

        qs_client.create_dashboard(
            name="Supply Chain Overview",
            dataset_ids=["dataset-one", "dataset-two"],
        )

        refs = qs_client._client.create_dashboard.call_args[1]["SourceEntity"]["SourceTemplate"]["DataSetReferences"]
        assert refs[0]["DataSetArn"] == "arn:aws:quicksight:us-east-1:123456789012:dataset/dataset-one"
        assert refs[1]["DataSetArn"] == "arn:aws:quicksight:us-east-1:123456789012:dataset/dataset-two"


class TestListDashboards:
    def test_returns_dashboard_summaries(self, qs_client):
        qs_client._client.list_dashboards.return_value = {
            "DashboardSummaryList": [
                {
                    "DashboardId": "dash-001",
                    "Name": "KPI Overview",
                    "Arn": "arn:aws:quicksight:us-east-1:123456789012:dashboard/dash-001",
                    "PublishedVersionNumber": 3,
                    "LastUpdatedTime": "2025-01-15T10:00:00Z",
                },
                {
                    "DashboardId": "dash-002",
                    "Name": "Event Analytics",
                    "Arn": "arn:aws:quicksight:us-east-1:123456789012:dashboard/dash-002",
                    "PublishedVersionNumber": 1,
                    "LastUpdatedTime": "2025-01-16T10:00:00Z",
                },
            ]
        }

        result = qs_client.list_dashboards()

        assert len(result) == 2
        assert result[0]["dashboard_id"] == "dash-001"
        assert result[0]["name"] == "KPI Overview"
        assert result[1]["published_version"] == 1

    def test_empty_list(self, qs_client):
        qs_client._client.list_dashboards.return_value = {"DashboardSummaryList": []}

        result = qs_client.list_dashboards()
        assert result == []
