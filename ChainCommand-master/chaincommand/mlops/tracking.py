"""MLflow connection and small helpers shared by the pipelines."""
from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request

import pandas as pd
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dataset_source_registry import register_dataset_source

import mlflow

from ..config import settings

MONITOR_EXPERIMENT = "chaincommand-monitoring"


def configure(experiment: str | None = None) -> None:
    """Point MLflow at the configured tracking server and select the experiment."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment or settings.mlflow_experiment)


def wait_for_server(timeout: float = 120.0) -> None:
    """Block until an http(s) tracking server answers (no-op for file/sqlite URIs)."""
    uri = settings.mlflow_tracking_uri
    if not uri.startswith("http"):
        return
    deadline = time.time() + timeout
    while True:
        try:
            urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=5)  # noqa: S310
            return
        except (urllib.error.URLError, OSError):
            if time.time() > deadline:
                raise RuntimeError(f"MLflow server at {uri} did not become ready in {timeout:.0f}s") from None
            time.sleep(2)


def fingerprint(df: pd.DataFrame) -> dict:
    """Small description of a training table, logged so a run can be traced to its data."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()[:16]
    out = {"rows": int(len(df)), "hash": digest}
    if "date" in df:
        out["date_min"] = str(pd.to_datetime(df["date"]).min().date())
        out["date_max"] = str(pd.to_datetime(df["date"]).max().date())
    if "product_id" in df:
        out["products"] = int(df["product_id"].nunique())
    return out


class DatabaseSource(DatasetSource):
    """Dataset source that records which database table(s) a dataset was read from."""

    def __init__(self, uri: str):
        self._uri = uri

    @staticmethod
    def _get_source_type() -> str:
        return "chaincommand_database"

    def load(self, **kwargs):
        raise NotImplementedError("Database sources are for lineage only; rebuild with build_dataset().")

    @staticmethod
    def _can_resolve(raw_source) -> bool:
        return isinstance(raw_source, str) and raw_source.startswith("db://")

    @classmethod
    def _resolve(cls, raw_source):
        return cls(raw_source)

    def to_dict(self) -> dict:
        return {"uri": self._uri}

    @classmethod
    def from_dict(cls, source_dict: dict):
        return cls(source_dict["uri"])


register_dataset_source(DatabaseSource)


def log_dataset(df: pd.DataFrame, name: str, source: str, context: str) -> None:
    """Attach the table to the active run (shows up under 'Datasets' in the MLflow UI)."""
    dataset = mlflow.data.from_pandas(df, source="db://" + source, name=name)
    mlflow.log_input(dataset, context=context)


def safe_metrics(d: dict) -> dict[str, float]:
    """Keep only finite numbers (MLflow rejects NaN/None)."""
    out = {}
    for k, v in d.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and abs(f) != float("inf"):
            out[k] = f
    return out
