"""Point-in-time features for the stockout classifier (all known at forecast origin)."""
import numpy as np
import pandas as pd


def add_stockout_features(df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    out = df.copy()
    mean7 = out["roll_mean_7"].replace(0, np.nan)
    mean28 = out["roll_mean_28"].replace(0, np.nan)

    # Expected demand over the window, from the recent average (not a model
    # forecast: those were trained on these days and would leak).
    out["expected_demand_h"] = out["roll_mean_7"] * horizon_days
    out["stock_minus_expected"] = out["stock_at_origin"] - out["expected_demand_h"]
    out["stock_cover_days"] = out["stock_at_origin"] / mean7
    out["demand_cv"] = out["roll_std_28"] / mean28
    out["lead_time_gap"] = out["supplier_lead_time_days"] - out["stock_cover_days"]
    return out


STOCKOUT_FEATURES = [
    "stock_at_origin", "stockout_days_14", "expected_demand_h",
    "stock_minus_expected", "stock_cover_days", "demand_cv",
    "supplier_lead_time_days", "lead_time_gap",
]