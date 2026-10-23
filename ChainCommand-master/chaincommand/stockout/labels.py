"""Stockout label: will this product lose sales in the next H days?"""
import pandas as pd

LABEL = "stockout_next_h"


def add_stockout_label(df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """Add LABEL = 1 if any lost sale occurs on days d .. d+H-1, else 0.

    Rows whose full H-day window is not yet observed (the last H-1 days of
    history, and future rows) get NaN so they are never trained or scored on.
    """
    out = df.sort_values(["product_id", "date"]).copy()

    gaps = out.groupby("product_id")["date"].diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        raise ValueError("expected one row per product per day")

    observed = (~out["is_future"]) & out["demand"].notna()
    hit = (out["lost_units"] > 0).astype(float).where(observed)

    def forward_sum(s: pd.Series) -> pd.Series:
        # rolling looks backwards, so reverse, roll, reverse back
        return s[::-1].rolling(horizon_days, min_periods=horizon_days).sum()[::-1]

    n_hits = hit.groupby(out["product_id"]).transform(forward_sum)
    out[LABEL] = (n_hits > 0).astype(float).where(n_hits.notna())
    return out