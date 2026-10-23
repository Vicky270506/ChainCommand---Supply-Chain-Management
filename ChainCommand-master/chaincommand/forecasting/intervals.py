"""Prediction intervals from two quantile LightGBM models."""
import numpy as np

from chaincommand.forecasting.gbm import GBMForecaster


class IntervalForecaster:
    def __init__(self, feature_cols, level=0.8):
        self.level = level
        tail = (1 - level) / 2
        self.lower = GBMForecaster(
            feature_cols, params={"objective": "quantile", "alpha": tail}
        )
        self.upper = GBMForecaster(
            feature_cols, params={"objective": "quantile", "alpha": 1 - tail}
        )

    def fit(self, X, y, X_valid=None, y_valid=None):
        self.lower.fit(X, y, X_valid, y_valid)
        self.upper.fit(X, y, X_valid, y_valid)
        return self

    def predict_interval(self, X):
        lo = self.lower.predict(X)
        hi = self.upper.predict(X)
        # Two separate models can occasionally cross; keep lower <= upper.
        return np.minimum(lo, hi), np.maximum(lo, hi)