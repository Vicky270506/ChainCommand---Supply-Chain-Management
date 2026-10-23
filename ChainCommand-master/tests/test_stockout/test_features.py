import numpy as np
import pandas as pd

from chaincommand.stockout.features import add_stockout_features


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