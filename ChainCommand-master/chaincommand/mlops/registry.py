"""Model registry: versions, aliases and champion/challenger promotion.

MLflow 3 replaces the old Staging/Production stages with *aliases*:
    champion    the model that serves forecasts
    challenger  a newly trained model while it is being compared
    previous_champion  the model that was replaced (for rollback)
The decision and the metrics are also written to the ``mlflow_metadata`` table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from sqlalchemy import update
from sqlalchemy.orm import Session

import mlflow
from mlflow import MlflowClient

from ..config import settings
from ..db.models import MLflowMetadata
from .tracking import safe_metrics

CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS = "previous_champion"
STOCKOUT_MODEL = "stockout-classifier"


def forecaster_name(horizon: int) -> str:
    return f"demand-forecaster-h{horizon}"


@dataclass
class Decision:
    promote: bool
    reason: str


def decide(challenger: float, champion: Optional[float], higher_is_better: bool = False,
           min_improvement: float = 0.0) -> Decision:
    """Promote when the challenger beats the champion by at least ``min_improvement`` (a fraction)."""
    if champion is None:
        return Decision(True, "no champion yet")
    if higher_is_better:
        needed = champion * (1 + min_improvement)
        ok = challenger > needed
    else:
        needed = champion * (1 - min_improvement)
        ok = challenger < needed
    word = "beats" if ok else "does not beat"
    return Decision(ok, f"challenger {challenger:.4f} {word} champion {champion:.4f} (needed {needed:.4f})")


def _client() -> MlflowClient:
    """A client for the configured server (the URI is set here, so callers need not call configure())."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def get_by_alias(name: str, alias: str):
    try:
        return _client().get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def load_champion(name: str):
    """Return (pyfunc model, version number) of the current champion."""
    mv = get_by_alias(name, CHAMPION)          # also points MLflow at the configured server
    if mv is None:
        raise LookupError(f"No champion for '{name}'. Train one: python -m chaincommand.mlops train-all")
    return mlflow.pyfunc.load_model(f"models:/{name}@{CHAMPION}"), int(mv.version)


def record(engine, name: str, version: int, run_id: str, metrics: dict, status: str) -> None:
    with Session(engine) as s:
        s.add(MLflowMetadata(model_name=name, model_version=str(version), run_id=run_id,
                             validation_metrics=safe_metrics(metrics), deployment_status=status))
        s.commit()


def promote_or_reject(engine, name: str, version: int, run_id: str, metrics: dict,
                      decision: Decision, tags: Optional[dict] = None) -> str:
    """Apply a decision to a freshly registered version. Returns 'champion' or 'rejected'."""
    client = _client()
    client.set_registered_model_alias(name, CHALLENGER, str(version))
    for k, v in {**(tags or {}), "decision_reason": decision.reason}.items():
        client.set_model_version_tag(name, str(version), k, str(v)[:250])

    old = get_by_alias(name, CHAMPION)
    if decision.promote:
        if old is not None:
            client.set_registered_model_alias(name, PREVIOUS, old.version)
            with Session(engine) as s:
                s.execute(update(MLflowMetadata)
                          .where(MLflowMetadata.model_name == name, MLflowMetadata.deployment_status == "champion")
                          .values(deployment_status="archived"))
                s.commit()
        client.set_registered_model_alias(name, CHAMPION, str(version))
        client.set_model_version_tag(name, str(version), "deployment_status", "champion")
        status = "champion"
    else:
        client.set_model_version_tag(name, str(version), "deployment_status", "rejected")
        status = "rejected"
    client.delete_registered_model_alias(name, CHALLENGER)
    record(engine, name, version, run_id, metrics, status)
    return status
