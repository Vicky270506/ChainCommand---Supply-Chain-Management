"""Feature engineering for forecasting and stockout-risk models.

Database loading lives in ``chaincommand.features.dataset`` (``build_dataset``).
"""

from .engineering import (
    CATEGORICAL,
    TARGET,
    describe_features,
    feature_columns,
    make_features,
    rolling_origin_folds,
    time_split,
)

__all__ = [
    "CATEGORICAL",
    "TARGET",
    "describe_features",
    "feature_columns",
    "make_features",
    "rolling_origin_folds",
    "time_split",
]
