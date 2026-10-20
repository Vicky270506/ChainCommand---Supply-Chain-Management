"""MLflow Model Registry integration for ChainCommand."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import mlflow.pyfunc  # noqa: I001
    from mlflow.tracking import MlflowClient

    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class ModelRegistry:
    """Manages model versioning and stage transitions via MLflow."""

    STAGES = ("Staging", "Production", "Archived")
    _STAGE_LOOKUP = {stage.lower(): stage for stage in STAGES}

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "chaincommand",
    ) -> None:
        if not HAS_MLFLOW:
            logger.warning("mlflow not installed — registry disabled")
            self._enabled = False
            return
        self._enabled = True
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._client = MlflowClient(tracking_uri)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log_model_run(
        self,
        model_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Log a training run and register the model."""
        if not self._enabled:
            return None

        class _PlaceholderModel(mlflow.pyfunc.PythonModel):
            """Minimal PythonModel wrapper so log_model always has a valid model object."""

            def predict(self, context: Any, model_input: Any) -> Any:  # noqa: ARG002
                """Placeholder predict — real inference uses the registered artifact."""
                return None

        with mlflow.start_run(
            run_name=f"{model_name}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        ) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            if artifacts:
                for name, path in artifacts.items():
                    if Path(path).exists():
                        mlflow.log_artifact(path, name)
            # Log a pyfunc model so that register_model() can find an
            # artifact at the "model" path.  We use a minimal wrapper since
            # the real model object may not be available at logging time.
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=_PlaceholderModel(),
                registered_model_name=None,  # registered separately via register_model()
            )
            # Log model metadata
            mlflow.log_dict(
                {
                    "model_name": model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "params": params,
                    "metrics": metrics,
                },
                "model_metadata.json",
            )
            logger.info(
                "logged mlflow run %s for model %s", run.info.run_id, model_name
            )
            return run.info.run_id

    def register_model(self, run_id: str, model_name: str) -> int | None:
        """Register a model version from a run."""
        if not self._enabled:
            return None
        model_uri = f"runs:/{run_id}/model"
        try:
            result = mlflow.register_model(model_uri, model_name)
            logger.info(
                "registered model %s version %s", model_name, result.version
            )
            return int(result.version)
        except mlflow.exceptions.MlflowException as e:
            logger.error("failed to register model %s: %s", model_name, e)
            return None
        except Exception:
            logger.exception("unexpected error registering model %s", model_name)
            raise

    def transition_stage(
        self, model_name: str, version: int, stage: str
    ) -> bool:
        """Transition a model version to a new stage."""
        if not self._enabled:
            return False
        normalized_stage = self._STAGE_LOOKUP.get(stage.lower())
        if normalized_stage is None:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of {self.STAGES}"
            )
        try:
            # transition_model_version_stage deprecated in MLflow 2.9+;
            # fall back to set_registered_model_alias if unavailable.
            try:
                self._client.transition_model_version_stage(
                    model_name, str(version), normalized_stage
                )
            except AttributeError:
                # MLflow 2.9+: use alias-based workflow instead
                alias = normalized_stage.lower()
                self._client.set_registered_model_alias(
                    model_name, alias, str(version)
                )
            logger.info(
                "transitioned %s v%d to %s", model_name, version, normalized_stage
            )
            return True
        except mlflow.exceptions.MlflowException as e:
            logger.error(
                "failed to transition %s v%d: %s", model_name, version, e
            )
            return False
        except Exception:
            logger.exception(
                "unexpected error transitioning %s v%d", model_name, version
            )
            raise

    def get_production_model(self, model_name: str) -> dict[str, Any] | None:
        """Get the current production model version."""
        if not self._enabled:
            return None
        try:
            # get_latest_versions deprecated in MLflow 2.9+;
            # fall back to get_model_version_by_alias if unavailable.
            try:
                versions = self._client.get_latest_versions(
                    model_name, stages=["Production"]
                )
                if versions:
                    v = versions[0]
                    return {
                        "name": model_name,
                        "version": int(v.version),
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                    }
            except AttributeError:
                # MLflow 2.9+: alias-based lookup
                v = self._client.get_model_version_by_alias(
                    model_name, "production"
                )
                return {
                    "name": model_name,
                    "version": int(v.version),
                    "stage": "Production",
                    "run_id": v.run_id,
                }
        except mlflow.exceptions.MlflowException as e:
            logger.error(
                "failed to get production model %s: %s", model_name, e
            )
        except Exception:
            logger.exception(
                "unexpected error getting production model %s", model_name
            )
            raise
        return None

    def list_models(self, max_results: int = 100) -> list[dict[str, Any]]:
        """List all registered models.

        Args:
            max_results: Maximum number of models to return (default 100).
        """
        if not self._enabled:
            return []
        try:
            models = self._client.search_registered_models(
                max_results=max_results,
            )
            return [
                {
                    "name": m.name,
                    "latest_versions": [
                        {"version": v.version, "stage": v.current_stage}
                        for v in (m.latest_versions or [])
                    ],
                }
                for m in models
            ]
        except mlflow.exceptions.MlflowException as e:
            logger.error("failed to list models: %s", e)
            return []
        except Exception:
            logger.exception("unexpected error listing models")
            raise
