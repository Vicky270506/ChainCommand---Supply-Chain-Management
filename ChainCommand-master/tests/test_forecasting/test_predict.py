import numpy as np
import pandas as pd

from chaincommand.features.engineering import feature_columns
from chaincommand.forecasting.predict import forecast_future, train_final


def _frame(n=900):
    rng = np.random.default_rng(3)
    cols = feature_columns(1)
    df = pd.DataFrame(rng.random((n, len(cols))), columns=cols)
    df["product_id"] = rng.integers(0, 3, n)
    df["category_id"] = rng.integers(0, 2, n)
    df["is_promo"] = rng.integers(0, 2, n)
    df["demand"] = 10 + 20 * df["is_promo"] + rng.normal(0, 3, n)
    df["date"] = pd.date_range("2024-01-01", periods=n)
    df["is_future"] = False
    df.loc[df.index[-30:], ["demand", "is_future"]] = [np.nan, True]
    return df


def test_final_model_forecasts_only_future_rows():
    df = _frame()
    
    point, interval = train_final(df)
    out = forecast_future(df, point, interval)
    assert len(out) == 30
    assert out["date"].min() > df.loc[~df["is_future"], "date"].max()
    assert (out["lower"] <= out["upper"]).all()
    assert (out["forecast"] >= 0).all()