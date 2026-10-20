"""ChainCommand ML Training Pipeline — Airflow DAG."""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "chaincommand",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="chaincommand_training_pipeline",
    default_args=default_args,
    description="ML + RL training pipeline for ChainCommand supply chain optimization",
    schedule="0 2 * * 1",  # Every Monday at 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    dagrun_timeout=timedelta(hours=4),
    tags=["ml", "rl", "training", "supply-chain"],
) as dag:

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command="python -m chaincommand.data.validator",
    )

    generate_features = BashOperator(
        task_id="generate_features",
        bash_command="python -m chaincommand.data.generator --mode=features",
    )

    train_forecaster = BashOperator(
        task_id="train_forecaster",
        bash_command="python -m chaincommand.models.forecaster --train",
    )

    train_anomaly_detector = BashOperator(
        task_id="train_anomaly_detector",
        bash_command="python -m chaincommand.models.anomaly_detector --train",
    )

    train_optimizer = BashOperator(
        task_id="train_optimizer",
        bash_command="python -m chaincommand.models.optimizer --train",
    )

    train_rl_policy = BashOperator(
        task_id="train_rl_policy",
        bash_command="python -c 'from chaincommand.rl import RLInventoryPolicy; p = RLInventoryPolicy(); p.train()'",
    )

    evaluate_models = BashOperator(
        task_id="evaluate_models",
        bash_command="python -m chaincommand.models.evaluate --all",
    )

    register_models = BashOperator(
        task_id="register_models",
        bash_command="python -m chaincommand.mlflow_registry --register",
    )

    promote_champion = BashOperator(
        task_id="promote_champion",
        bash_command="python -m chaincommand.mlflow_registry --promote",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    validate_data >> generate_features
    generate_features >> [train_forecaster, train_anomaly_detector, train_optimizer, train_rl_policy]
    [train_forecaster, train_anomaly_detector, train_optimizer, train_rl_policy] >> evaluate_models
    evaluate_models >> register_models >> promote_champion
