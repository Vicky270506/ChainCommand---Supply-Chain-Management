import numpy as np
import pandas as pd

from chaincommand.stockout.labels import LABEL, add_stockout_label


def _frame(lost, pid=1):
    n = len(lost)
    return pd.DataFrame({
        "product_id": pid,
        "date": pd.date_range("2026-01-01", periods=n),
        "lost_units": lost,
        "demand": 10.0,
        "is_future": False,
    })


def test_window_and_incomplete_tail():
    out = add_stockout_label(_frame([0, 0, 0, 0, 5, 0, 0, 0, 0, 0]), 3)
    expected = pd.Series([0, 0, 1, 1, 1, 0, 0, 0, np.nan, np.nan])
    pd.testing.assert_series_equal(
        out[LABEL].reset_index(drop=True), expected, check_names=False
    )


def test_products_are_independent():
    df = pd.concat([_frame([0] * 6, pid=1), _frame([0, 0, 9, 0, 0, 0], pid=2)])
    out = add_stockout_label(df, 2)
    assert out[out["product_id"] == 1][LABEL].dropna().sum() == 0
    assert out[out["product_id"] == 2][LABEL].sum() == 2  # days 1 and 2


def test_future_rows_get_no_label():
    df = _frame([0, 0, 0, 0, 0, 0])
    df.loc[4:, ["is_future", "demand"]] = [True, np.nan]
    out = add_stockout_label(df, 2)
    assert out[LABEL].iloc[3:].isna().all()


def test_label_never_looks_backwards():
    base = _frame([0] * 10)
    changed = _frame([0] * 10)
    changed.loc[6, "lost_units"] = 5
    a = add_stockout_label(base, 3)[LABEL]
    b = add_stockout_label(changed, 3)[LABEL]
    # rows after the changed day cannot be affected
    assert a.iloc[7:].equals(b.iloc[7:]) or (a.iloc[7:].isna() == b.iloc[7:].isna()).all()
    assert b.iloc[4:7].eq(1).all()