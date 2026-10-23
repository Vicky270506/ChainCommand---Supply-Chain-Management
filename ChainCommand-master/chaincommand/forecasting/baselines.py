"""Baseline forecasters. Both take the Step 4 feature table (a DataFrame)."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SeasonalNaive:
    """Forecast = demand on the same weekday last week (lag_7).

    Falls back to the 28-day rolling mean when lag_7 is missing, then to 0.
    """

    def __init__(self, col="lag_7", fallback="roll_mean_28"):
        self.col = col
        self.fallback = fallback

    def fit(self, X, y=None):
        return self  # nothing to learn

    def predict(self, X):
        p = X[self.col].fillna(X[self.fallback])
        return p.fillna(0).to_numpy(dtype=float)


class LinearBaseline:
    """Linear regression on numeric features.

    Median-imputes gaps (lag_364 is empty in year one), scales, fits, and
    clips predictions at 0 because demand cannot be negative.
    """

    def __init__(self, feature_cols):
        self.feature_cols = list(feature_cols)
        self.model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearRegression(),
        )

    def fit(self, X, y):
        self.model.fit(X[self.feature_cols], y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X[self.feature_cols]), 0, None)