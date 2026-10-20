"""Dashboard endpoint tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestKPIEndpoints:
    @pytest.mark.asyncio
    async def test_kpi_current_no_data(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get("/api/kpi/current", headers=auth_headers)
            assert resp.status_code == 503
            assert resp.json()["detail"] == "No KPI data available yet"

    @pytest.mark.asyncio
    async def test_kpi_history_no_data(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get("/api/kpi/history", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 0


class TestInventoryEndpoints:
    @pytest.mark.asyncio
    async def test_inventory_status(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get("/api/inventory/status", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 1
            assert data["products"][0]["product_id"] == "PRD-test01"

    @pytest.mark.asyncio
    async def test_inventory_status_filter(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get(
                "/api/inventory/status?product_id=PRD-test01",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_inventory_status_missing_product(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get(
                "/api/inventory/status?product_id=NONEXISTENT",
                headers=auth_headers,
            )
            assert resp.status_code == 404


class TestApprovalEndpoints:
    @pytest.mark.asyncio
    async def test_pending_approvals_empty(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get("/api/approvals/pending", headers=auth_headers)
            assert resp.status_code == 200
            assert resp.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_decide_approval_not_found(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.post(
                "/api/approval/FAKE-ID/decide?approved=true",
                headers=auth_headers,
            )
            assert resp.status_code == 404
            assert resp.json()["detail"] == "Approval request FAKE-ID not found"


class TestAWSEndpoints:
    @pytest.mark.asyncio
    async def test_aws_status(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get("/api/aws/status", headers=auth_headers)
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_aws_kpi_trend_disabled(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get(
                "/api/aws/kpi-trend/otif",
                headers=auth_headers,
            )
            assert resp.status_code == 503
            assert resp.json()["detail"] == "AWS backend not enabled"
