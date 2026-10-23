import numpy as np
import pandas as pd

from chaincommand.stockout.features import add_stockout_features

from chaincommand.stockout.features import incoming_units


def test_incoming_units_point_in_time():
    dates = pd.date_range("2026-01-01", periods=8)
    df = pd.concat([
        pd.DataFrame({"product_id": 1, "date": dates}),
        pd.DataFrame({"product_id": 2, "date": dates}),
    ], ignore_index=True)
    orders = pd.DataFrame({
        "product_id": [1, 2],
        "date": pd.to_datetime(["2026-01-03", "2026-01-05"]),
        "received_date": pd.to_datetime(["2026-01-06", None]),
        "quantity": [100, 50],
    })
    got = incoming_units(df, orders).tolist()
    # product 1: on order from Jan 3 to Jan 5, received Jan 6 -> visible to rows Jan 4-6
    assert got[:8] == [0, 0, 0, 100, 100, 100, 0, 0]
    # product 2: still open at the end of the data
    assert got[8:] == [0, 0, 0, 0, 0, 50, 50, 50]
    
def test_cover_and_expected_demand():
    df = pd.DataFrame({
        "stock_at_origin": [70.0, 50.0], "roll_mean_7": [10.0, 0.0],
        "roll_mean_28": [10.0, 0.0], "roll_std_28": [5.0, 0.0],
        "supplier_lead_time_days": [5.0, 5.0],
    })
    out = add_stockout_features(df, 7)
    assert out["expected_demand_h"].tolist() == [70.0, 0.0]
    assert out["stock_cover_days"].iloc[0] == 7.0
    assert np.isnan(out["stock_cover_days"].iloc[1])  # zero demand, no divide error
    assert out["demand_cv"].iloc[0] == 0.5
    assert out["lead_time_gap"].iloc[0] == -2.0