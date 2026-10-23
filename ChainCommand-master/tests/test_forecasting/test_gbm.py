import numpy as np
import pandas as pd

from chaincommand.features.engineering import feature_columns
from chaincommand.forecasting.gbm import GBMForecaster


def _frame(n=600):
    rng = np.random.default_rng(1)
    cols = feature_columns(1)
    df = pd.DataFrame(rng.random((n, len(cols))), columns=cols)
    df["product_id"] = rng.integers(0, 5, n)
    df["category_id"] = rng.integers(0, 2, n)
    df["is_promo"] = rng.integers(0, 2, n)
    df["demand"] = 5 + 20 * df["is_promo"]  # a clean rule the model should find
    return df


def test_gbm_learns_simple_rule_and_is_nonnegative():
    df = _frame()
    train, valid, test = df[:400], df[400:500], df[500:]
    params = {"min_data_in_leaf": 5, "learning_rate": 0.1}
    m = GBMForecaster(feature_columns(1), params=params)
    m.fit(train, train["demand"], valid, valid["demand"])
    pred = m.predict(test)
    assert pred.shape == (100,)
    assert (pred >= 0).all()
    assert np.abs(pred - test["demand"]).mean() < 2


def test_importance_lists_every_feature():
    df = _frame()
    m = GBMForecaster(feature_columns(1), params={"min_data_in_leaf": 5})
    m.fit(df, df["demand"])
    imp = m.importance()
    assert set(imp["feature"]) == set(feature_columns(1))
    assert imp.iloc[0]["feature"] == "is_promo"