"""Shared fixtures for API test modules."""

from __future__ import annotations

import pytest


@pytest.fixture
def api_key():
    from chaincommand.config import settings
    return settings.api_key.get_secret_value()


@pytest.fixture
def auth_headers(api_key):
    return {"X-API-Key": api_key}
