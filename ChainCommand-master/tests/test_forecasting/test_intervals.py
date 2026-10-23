import numpy as np
import pandas as pd

from chaincommand.features.engineering import feature_columns
from chaincommand.forecasting.intervals import IntervalForecaster
from chaincommand.forecasting.metrics import coverage


def _noisy_frame(n=1500):
    rng = np.random.default_rng(2)
    cols = feature_columns(1)
    df = pd.DataFrame(rng.random((n, len(cols))), columns=cols)
    df["product_id"] = rng.integers(0, 5, n)
    df["category_id"] = rng.integers(0, 2, n)
    df["is_promo"] = rng.integers(0, 2, n)
    df["demand"] = 20 + 20 * df["is_promo"] + rng.normal(0, 4, n)
    return df


def test_interval_is_ordered_and_covers_roughly_target():
    df = _noisy_frame()
    train, valid, test = df[:1000], df[1000:1250], df[1250:]
    m = IntervalForecaster(
        feature_cols=feature_columns(1), level=0.8
    )
    for g in (m.lower, m.upper):
        g.params.update({"min_data_in_leaf": 10, "learning_rate": 0.1})
    m.fit(train, train["demand"], valid, valid["demand"])
    lo, hi = m.predict_interval(test)
    assert (lo <= hi).all()
    cov = coverage(test["demand"], lo, hi)
    assert 0.6 < cov < 0.95