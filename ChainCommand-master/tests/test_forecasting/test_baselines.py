import numpy as np
import pandas as pd
import pytest

from chaincommand.forecasting.baselines import SeasonalNaive, LinearBaseline


def test_seasonal_naive_uses_lag_7():
    X = pd.DataFrame({"lag_7": [5.0, 9.0], "roll_mean_28": [1.0, 1.0]})
    assert SeasonalNaive().fit(X).predict(X).tolist() == [5.0, 9.0]


def test_seasonal_naive_fallback_and_zero():
    X = pd.DataFrame({
        "lag_7": [np.nan, np.nan],
        "roll_mean_28": [4.0, np.nan],
    })
    assert SeasonalNaive().predict(X).tolist() == [4.0, 0.0]


def test_linear_recovers_simple_relation():
    X = pd.DataFrame({"lag_1": np.arange(50, dtype=float)})
    y = 2 * X["lag_1"] + 3
    m = LinearBaseline(["lag_1"]).fit(X, y)
    assert m.predict(X) == pytest.approx(y.to_numpy(), abs=1e-6)


def test_linear_handles_nan_and_never_negative():
    X = pd.DataFrame({
        "lag_1": np.arange(50, dtype=float),
        "lag_364": [np.nan] * 25 + list(range(25)),
    })
    y = X["lag_1"] - 20  # goes negative early on
    m = LinearBaseline(["lag_1", "lag_364"]).fit(X, y)
    assert (m.predict(X) >= 0).all()