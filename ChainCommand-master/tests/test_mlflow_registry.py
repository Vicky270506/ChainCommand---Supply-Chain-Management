"""Tests for MLflow model registry wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chaincommand.mlflow_registry import ModelRegistry


class TestModelRegistryStages:
    def test_transition_stage_normalizes_case(self):
        registry = ModelRegistry.__new__(ModelRegistry)
        registry._enabled = True
        registry._client = SimpleNamespace()
        calls = []

        def transition(model_name, version, stage):
            calls.append((model_name, version, stage))

        registry._client.transition_model_version_stage = transition

        assert registry.transition_stage("forecast", 3, "production") is True
        assert calls == [("forecast", "3", "Production")]

    def test_transition_stage_rejects_unknown_stage(self):
        registry = ModelRegistry.__new__(ModelRegistry)
        registry._enabled = True
        registry._client = SimpleNamespace()

        with pytest.raises(ValueError):
            registry.transition_stage("forecast", 3, "live")

    def test_get_production_model_uses_mlflow_stage_name(self):
        registry = ModelRegistry.__new__(ModelRegistry)
        registry._enabled = True
        captured = {}

        class Client:
            def get_latest_versions(self, model_name, stages):
                captured["model_name"] = model_name
                captured["stages"] = stages
                return [
                    SimpleNamespace(
                        version="7",
                        current_stage="Production",
                        run_id="run-123",
                    )
                ]

        registry._client = Client()

        result = registry.get_production_model("forecast")

        assert captured["stages"] == ["Production"]
        assert result == {
            "name": "forecast",
            "version": 7,
            "stage": "Production",
            "run_id": "run-123",
        }
