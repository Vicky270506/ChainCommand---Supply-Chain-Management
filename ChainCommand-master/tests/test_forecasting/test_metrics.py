import numpy as np
import pytest

from chaincommand.forecasting.metrics import mae, rmse, smape, bias, coverage


def test_perfect_forecast():
    y = [5, 10, 15]
    assert mae(y, y) == 0
    assert rmse(y, y) == 0
    assert smape(y, y) == 0
    assert bias(y, y) == 0


def test_mae_and_bias_symmetric_errors():
    y, p = [10, 10], [12, 8]
    assert mae(y, p) == 2
    assert bias(y, p) == 0


def test_rmse_penalises_big_miss():
    y, p = [0, 0, 0, 0], [0, 0, 0, 8]
    assert mae(y, p) == 2
    assert rmse(y, p) == pytest.approx(4.0)


def test_bias_sign():
    assert bias([10], [15]) > 0   # over-forecast
    assert bias([10], [5]) < 0    # under-forecast


def test_smape_value():
    assert smape([100], [50]) == pytest.approx(200 / 3)


def test_smape_both_zero_is_zero():
    assert smape([0, 0], [0, 0]) == 0


def test_coverage_half():
    y = [1, 2, 3, 4]
    lower = [0, 0, 5, 5]
    upper = [2, 2, 6, 6]
    assert coverage(y, lower, upper) == 0.5


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2], [1])