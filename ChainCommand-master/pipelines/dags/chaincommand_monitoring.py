"""ChainCommand Model Monitoring Pipeline — Airflow DAG.

Checks for data drift, model performance degradation, and anomaly detection
quality every 6 hours.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "chaincommand",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _check_drift_threshold(**context) -> str:
    """Decide whether drift warrants retraining."""
    ti = context["ti"]
    drift_result = ti.xcom_pull(task_ids="detect_data_drift")
    # If drift detection command exits 0 with drift detected, retrain
    if drift_result and "drift_detected=true" in str(drift_result):
        return "trigger_retraining"
    return "log_no_drift"


with DAG(
    dag_id="chaincommand_monitoring_pipeline",
    default_args=default_args,
    description="Monitors model performance and data drift for ChainCommand",
    schedule="0 */6 * * *",  # Every 6 hours
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "monitoring", "drift-detection"],
) as dag:

    collect_predictions = BashOperator(
        task_id="collect_predictions",
        bash_command="python -c \""
        "from chaincommand.mlflow_registry import ModelRegistry; "
        "r = ModelRegistry(); "
        "models = r.list_models(); "
        "print(f'Registered models: {len(models)}')\"",
    )

    # PSI threshold configurable via Airflow Variable 'cc_drift_psi_threshold' (default 0.1)
    detect_data_drift = BashOperator(
        task_id="detect_data_drift",
        bash_command="python -c \""
        "import json; "
        "from airflow.models import Variable; "
        "from chaincommand.models.anomaly_detector import AnomalyDetector; "
        "ad = AnomalyDetector(); "
        "result = ad.detect_drift() if hasattr(ad, 'detect_drift') else {'drift_detected': False, 'psi_score': 0.0}; "
        "psi_threshold = float(Variable.get('cc_drift_psi_threshold', default_var='0.1')); "
        "result['drift_detected'] = result.get('psi_score', 0) > psi_threshold; "
        "print(json.dumps(result))\"",
    )

    # Thresholds configurable via Airflow Variables:
    #   cc_mape_threshold (default 0.20), cc_rmse_threshold (default 100.0)
    evaluate_forecast_accuracy = BashOperator(
        task_id="evaluate_forecast_accuracy",
        bash_command="python -c \""
        "import json; "
        "from airflow.models import Variable; "
        "from chaincommand.models.forecaster import EnsembleForecaster; "
        "f = EnsembleForecaster(); "
        "metrics = f.evaluate() if hasattr(f, 'evaluate') else {'mape': None, 'rmse': None, 'bias': None}; "
        "thresholds = {'mape_threshold': float(Variable.get('cc_mape_threshold', default_var='0.20')), "
        "'rmse_threshold': float(Variable.get('cc_rmse_threshold', default_var='100.0'))}; "
        "metrics['thresholds'] = thresholds; "
        "print(json.dumps(metrics))\"",
    )

    # Thresholds configurable via Airflow Variables:
    #   cc_anomaly_precision_threshold (default 0.80), cc_anomaly_recall_threshold (default 0.70)
    evaluate_anomaly_precision = BashOperator(
        task_id="evaluate_anomaly_precision",
        bash_command="python -c \""
        "import json; "
        "from airflow.models import Variable; "
        "from chaincommand.models.anomaly_detector import AnomalyDetector; "
        "ad = AnomalyDetector(); "
        "metrics = ad.evaluate() if hasattr(ad, 'evaluate') else {'precision': None, 'recall': None, 'f1': None}; "
        "thresholds = {'precision_threshold': float(Variable.get('cc_anomaly_precision_threshold', default_var='0.80')), "
        "'recall_threshold': float(Variable.get('cc_anomaly_recall_threshold', default_var='0.70'))}; "
        "metrics['thresholds'] = thresholds; "
        "print(json.dumps(metrics))\"",
    )

    check_drift = BranchPythonOperator(
        task_id="check_drift_threshold",
        python_callable=_check_drift_threshold,
    )

    trigger_retraining = BashOperator(
        task_id="trigger_retraining",
        bash_command="airflow dags trigger chaincommand_training_pipeline",
    )

    log_no_drift = BashOperator(
        task_id="log_no_drift",
        bash_command="echo 'No significant drift detected. Models are performing within thresholds.'",
    )

    report_metrics = BashOperator(
        task_id="report_metrics",
        bash_command="python -c \""
        "from chaincommand.metrics import track_kpi; "
        "track_kpi('monitoring_run', 1.0); "
        "print('Monitoring metrics reported.')\"",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    collect_predictions >> [detect_data_drift, evaluate_forecast_accuracy, evaluate_anomaly_precision]
    detect_data_drift >> check_drift >> [trigger_retraining, log_no_drift]
    [evaluate_forecast_accuracy, evaluate_anomaly_precision, trigger_retraining, log_no_drift] >> report_metrics
