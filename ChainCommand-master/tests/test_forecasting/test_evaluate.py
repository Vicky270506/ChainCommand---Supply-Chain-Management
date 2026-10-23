import numpy as np
import pandas as pd

from chaincommand.features.engineering import feature_columns
from chaincommand.forecasting.evaluate import evaluate_models, linear_columns


def _fake_frame(n=60):
    rng = np.random.default_rng(0)
    cols = feature_columns(1)
    df = pd.DataFrame(rng.random((n, len(cols))), columns=cols)
    df["demand"] = df["lag_7"]  # seasonal naive is exactly right here
    return df


def test_linear_columns_drop_ids():
    cols = linear_columns(1)
    assert "product_id" not in cols
    assert "category_id" not in cols
    assert "lag_7" in cols


def test_evaluate_models_shape_and_naive_perfect():
    df = _fake_frame()
    res = evaluate_models(df[:30], df[30:45], df[45:], horizon=1, include_gbm=False)
    assert len(res) == 4  # 2 models x 2 splits
    assert set(res["model"]) == {"seasonal_naive", "linear"}
    naive = res[res["model"] == "seasonal_naive"]
    assert (naive["MAE"] == 0).all()