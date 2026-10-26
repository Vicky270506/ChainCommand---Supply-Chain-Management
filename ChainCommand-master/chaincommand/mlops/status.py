"""Registry status read from the ``mlflow_metadata`` table (no MLflow server needed)."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.base import get_engine
from ..db.models import MLflowMetadata

KEY_METRICS = ("test_mae", "test_rmse", "test_smape", "test_bias", "test_coverage",
               "test_precision", "test_recall", "test_f1", "test_pr_auc")


def registry_status(engine=None, limit: int = 50, session: Session | None = None) -> list[dict]:
    query = select(MLflowMetadata).order_by(MLflowMetadata.created_at.desc(), MLflowMetadata.id.desc()).limit(limit)
    if session is not None:
        rows = session.scalars(query).all()
    else:
        with Session(engine or get_engine()) as s:
            rows = s.scalars(query).all()
    return [{"model_name": r.model_name, "model_version": r.model_version, "run_id": r.run_id,
             "deployment_status": r.deployment_status, "created_at": r.created_at.isoformat(),
             "metrics": {k: v for k, v in (r.validation_metrics or {}).items() if k in KEY_METRICS}}
            for r in rows]
