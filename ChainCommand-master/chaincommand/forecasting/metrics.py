"""Forecast accuracy metrics. All inputs are array-like of equal length."""
import numpy as np


def _to_arrays(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    return y, p


def mae(y_true, y_pred):
    """Mean absolute error, in units of demand."""
    y, p = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def rmse(y_true, y_pred):
    """Root mean squared error. Penalises large misses more than MAE."""
    y, p = _to_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def smape(y_true, y_pred):
    """Symmetric MAPE in percent.

    When both actual and forecast are 0 the denominator is 0. That is a
    perfect forecast, so those terms count as 0 error (kept in the mean,
    not dropped).
    """
    y, p = _to_arrays(y_true, y_pred)
    denom = np.abs(y) + np.abs(p)
    terms = np.zeros_like(denom)
    nonzero = denom != 0
    terms[nonzero] = 2 * np.abs(y - p)[nonzero] / denom[nonzero]
    return float(np.mean(terms) * 100)


def bias(y_true, y_pred):
    """Mean signed error. Positive = over-forecasting, negative = under."""
    y, p = _to_arrays(y_true, y_pred)
    return float(np.mean(p - y))


def coverage(y_true, lower, upper):
    """Share of actuals that fall inside [lower, upper]."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))