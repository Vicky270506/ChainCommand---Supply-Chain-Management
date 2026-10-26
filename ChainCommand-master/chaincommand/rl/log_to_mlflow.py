"""Log artifacts/policy_comparison.csv to MLflow."""
import os

import mlflow
import pandas as pd

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("rl-policy-comparison")
df = pd.read_csv("artifacts/policy_comparison.csv")

for row in df.to_dict("records"):
    key = ("rule_based" if row["policy"].startswith("rule")
           else "ppo" if row["policy"].startswith("PPO") else "ss_tuned")
    with mlflow.start_run(run_name=key):
        mlflow.log_param("policy", row["policy"])
        for name, value in row.items():
            if name != "policy":
                mlflow.log_metric(name.replace("%", "pct"), float(value))
print("logged", len(df), "runs")