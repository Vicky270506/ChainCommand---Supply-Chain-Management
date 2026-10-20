"""Tests for S3Client — all boto3 calls are mocked."""

from __future__ import annotations

import json
import sys
from datetime import timezone  # noqa: F401
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def mock_boto3():
    mock = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock}):
        yield mock


@pytest.fixture
def s3_client(mock_boto3):
    # Force re-import so the lazy `import boto3` inside __init__ picks up our mock
    import importlib

    import chaincommand.aws.s3_client as mod

    importlib.reload(mod)
    client = mod.S3Client(bucket="test-bucket", prefix="test-prefix", region="us-east-1")
    return client


class TestUploadDataframe:
    def test_uploads_parquet_to_s3(self, s3_client):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        key = "test-prefix/demand/2025/01/01/data.parquet"

        result = s3_client.upload_dataframe(df, key)

        s3_client._client.put_object.assert_called_once()
        call_kwargs = s3_client._client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == key
        assert call_kwargs["ContentType"] == "application/octet-stream"
        assert result == f"s3://test-bucket/{key}"


class TestUploadJsonl:
    def test_uploads_jsonl_to_s3(self, s3_client):
        records = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
        key = "test-prefix/events/2025/01/01/events.jsonl"

        result = s3_client.upload_jsonl(records, key)

        s3_client._client.put_object.assert_called_once()
        call_kwargs = s3_client._client.put_object.call_args[1]
        body = call_kwargs["Body"].decode("utf-8")
        lines = body.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1, "name": "A"}
        assert result == f"s3://test-bucket/{key}"


class TestUploadJson:
    def test_uploads_json_to_s3(self, s3_client):
        data = {"status": "ok", "count": 42}
        key = "test-prefix/meta/snapshot.json"

        result = s3_client.upload_json(data, key)

        s3_client._client.put_object.assert_called_once()
        call_kwargs = s3_client._client.put_object.call_args[1]
        parsed = json.loads(call_kwargs["Body"].decode("utf-8"))
        assert parsed == data
        assert result == f"s3://test-bucket/{key}"


class TestListObjects:
    def test_lists_objects(self, s3_client):
        s3_client._client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "test-prefix/events/file1.jsonl", "Size": 100, "LastModified": "2025-01-01"},
                {"Key": "test-prefix/events/file2.jsonl", "Size": 200, "LastModified": "2025-01-02"},
            ]
        }

        result = s3_client.list_objects("events")
        assert len(result) == 2
        assert result[0]["key"] == "test-prefix/events/file1.jsonl"
        assert result[1]["size"] == 200

    def test_empty_prefix(self, s3_client):
        s3_client._client.list_objects_v2.return_value = {}

        result = s3_client.list_objects("nonexistent")
        assert result == []


class TestDownloadJson:
    def test_downloads_and_parses_json(self, s3_client):
        body_mock = MagicMock()
        body_mock.read.return_value = b'{"key": "value"}'
        s3_client._client.get_object.return_value = {"Body": body_mock}

        result = s3_client.download_json("test-prefix/meta/snapshot.json")
        assert result == {"key": "value"}


class TestBuildKey:
    def test_date_partitioned_key(self, s3_client):
        """_build_key was removed as dead code; verify it no longer exists."""
        assert not hasattr(s3_client, "_build_key")
