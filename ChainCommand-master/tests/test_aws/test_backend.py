"""Tests for PersistenceBackend, NullBackend, get_backend(), and AWSBackend."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from chaincommand.aws.backend import NullBackend, PersistenceBackend, get_backend
from chaincommand.data.schemas import KPISnapshot

# ── NullBackend tests ────────────────────────────────────

class TestNullBackend:
    @pytest.mark.asyncio
    async def test_setup_is_noop(self):
        backend = NullBackend()
        await backend.setup()  # should not raise

    @pytest.mark.asyncio
    async def test_teardown_is_noop(self):
        backend = NullBackend()
        await backend.teardown()

    @pytest.mark.asyncio
    async def test_persist_cycle_is_noop(self):
        backend = NullBackend()
        snapshot = KPISnapshot(timestamp=datetime(2025, 1, 1))
        await backend.persist_cycle(
            cycle=1, kpi=snapshot, events=[], pos=[], products=[], suppliers=[]
        )

    @pytest.mark.asyncio
    async def test_persist_demand_history_is_noop(self):
        backend = NullBackend()
        df = pd.DataFrame({"a": [1, 2, 3]})
        await backend.persist_demand_history(df)

    @pytest.mark.asyncio
    async def test_query_kpi_trend_returns_empty(self):
        backend = NullBackend()
        result = await backend.query_kpi_trend("otif", 30)
        assert result == []

    @pytest.mark.asyncio
    async def test_query_events_returns_empty(self):
        backend = NullBackend()
        result = await backend.query_events("low_stock_alert", 50)
        assert result == []

    def test_is_persistence_backend(self):
        backend = NullBackend()
        assert isinstance(backend, PersistenceBackend)


# ── get_backend() factory tests ──────────────────────────

class TestGetBackend:
    def test_returns_null_backend_when_disabled(self):
        with patch("chaincommand.aws.backend.settings") as mock_settings:
            mock_settings.aws_enabled = False
            backend = get_backend()
            assert isinstance(backend, NullBackend)

    def test_returns_aws_backend_when_enabled(self):
        with patch("chaincommand.aws.backend.settings") as mock_settings:
            mock_settings.aws_enabled = True
            with patch("chaincommand.aws.aws_backend.S3Client"):
                with patch("chaincommand.aws.aws_backend.RedshiftClient"):
                    with patch("chaincommand.aws.aws_backend.AthenaClient"):
                        with patch("chaincommand.aws.aws_backend.QuickSightClient"):
                            backend = get_backend()
                            from chaincommand.aws.aws_backend import AWSBackend
                            assert isinstance(backend, AWSBackend)


# ── AWSBackend integration tests (mocked sub-clients) ───

class TestAWSBackend:
    @pytest.fixture
    def aws_backend(self):
        with patch("chaincommand.aws.aws_backend.S3Client") as MockS3, \
             patch("chaincommand.aws.aws_backend.RedshiftClient") as MockRS, \
             patch("chaincommand.aws.aws_backend.AthenaClient") as MockAthena, \
             patch("chaincommand.aws.aws_backend.QuickSightClient") as MockQS:
            from chaincommand.aws.aws_backend import AWSBackend

            # Athena methods are now async — use AsyncMock
            athena_instance = MockAthena.return_value
            athena_instance.create_database = AsyncMock()
            athena_instance.create_external_tables = AsyncMock()
            athena_instance.run_query = AsyncMock(return_value=[])

            backend = AWSBackend()
            # Store mocked classes for assertions
            backend._mock_s3_cls = MockS3
            backend._mock_rs_cls = MockRS
            backend._mock_athena_cls = MockAthena
            backend._mock_qs_cls = MockQS
            yield backend

    @pytest.mark.asyncio
    async def test_setup_initializes_all_clients(self, aws_backend):
        await aws_backend.setup()

        assert aws_backend._s3 is not None
        assert aws_backend._redshift is not None
        assert aws_backend._athena is not None
        assert aws_backend._quicksight is not None

        aws_backend._redshift.create_tables.assert_called_once()
        aws_backend._athena.create_database.assert_called_once()
        aws_backend._athena.create_external_tables.assert_called_once()

    @pytest.mark.asyncio
    async def test_teardown_closes_redshift(self, aws_backend):
        await aws_backend.setup()
        await aws_backend.teardown()

        aws_backend._redshift.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_cycle_uploads_to_s3_and_redshift(self, aws_backend):
        await aws_backend.setup()

        snapshot = KPISnapshot(
            timestamp=datetime(2025, 1, 15),
            otif=0.95,
            fill_rate=0.97,
        )

        # Create mock events and POs with model_dump
        mock_event = MagicMock()
        mock_event.model_dump.return_value = {"event_id": "EVT-1", "event_type": "test"}

        mock_po = MagicMock()
        mock_po.model_dump.return_value = {"po_id": "PO-1", "quantity": 100}

        await aws_backend.persist_cycle(
            cycle=1,
            kpi=snapshot,
            events=[mock_event],
            pos=[mock_po],
            products=[],
            suppliers=[],
        )

        # S3: should upload KPI JSONL, events JSONL, POs JSONL
        assert aws_backend._s3.upload_json.call_count == 0
        assert aws_backend._s3.upload_jsonl.call_count == 3

        # Redshift: should insert KPI snapshot
        aws_backend._redshift.insert_kpi_snapshot.assert_called_once_with(1, snapshot)

    @pytest.mark.asyncio
    async def test_persist_cycle_skips_empty_events(self, aws_backend):
        await aws_backend.setup()

        snapshot = KPISnapshot(timestamp=datetime(2025, 1, 15))
        await aws_backend.persist_cycle(
            cycle=1, kpi=snapshot, events=[], pos=[], products=[], suppliers=[]
        )

        # Only KPI JSONL uploaded, no other JSONL batches
        assert aws_backend._s3.upload_json.call_count == 0
        assert aws_backend._s3.upload_jsonl.call_count == 1

    @pytest.mark.asyncio
    async def test_persist_demand_history(self, aws_backend):
        await aws_backend.setup()

        df = pd.DataFrame({"date": ["2025-01-01"], "product_id": ["PRD-1"], "quantity": [100]})
        await aws_backend.persist_demand_history(df)

        aws_backend._s3.upload_dataframe.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_kpi_trend(self, aws_backend):
        await aws_backend.setup()
        aws_backend._redshift.query.return_value = [
            {"cycle": 1, "otif": 0.95},
            {"cycle": 2, "otif": 0.93},
        ]

        result = await aws_backend.query_kpi_trend("otif", 7)
        assert len(result) == 2
        aws_backend._redshift.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_kpi_trend_invalid_metric(self, aws_backend):
        await aws_backend.setup()
        result = await aws_backend.query_kpi_trend("invalid_metric", 7)
        assert result == []

    @pytest.mark.asyncio
    async def test_query_events(self, aws_backend):
        await aws_backend.setup()
        aws_backend._athena.run_query.return_value = [
            {"event_id": "EVT-1", "event_type": "low_stock"}
        ]

        result = await aws_backend.query_events("low_stock", 10)
        assert len(result) == 1
        aws_backend._athena.run_query.assert_called_once()
