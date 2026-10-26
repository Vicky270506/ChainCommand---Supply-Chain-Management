"""MLflow pyfunc wrappers, so models can be loaded as ``models:/<name>@champion``
without knowing which library trained them.

The ``chaincommand`` package must be installed wherever these models are loaded
(it is, in the Docker image).
"""
from __future__ import annotations

import joblib
import mlflow.pyfunc
import pandas as pd


class _Bundle(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = joblib.load(context.artifacts["bundle"])


class ForecastModel(_Bundle):
    """Input: the feature columns of the forecasting table. Output: forecast, lower, upper."""

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)
        lo, hi = self.bundle["interval"].predict_interval(X)
        return pd.DataFrame(
            {"forecast": self.bundle["point"].predict(X), "lower": lo, "upper": hi}, index=X.index
        )


class StockoutModel(_Bundle):
    """Input: the stockout feature columns. Output: probability and the 0/1 flag."""

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)
        clf = self.bundle["classifier"]
        prob = clf.predict_proba(X)
        return pd.DataFrame(
            {"stockout_probability": prob, "stockout_flag": (prob >= clf.threshold).astype(int)}, index=X.index
        )
