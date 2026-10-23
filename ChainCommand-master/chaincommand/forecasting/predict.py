"""Train the final model, save it, and forecast the future rows."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from chaincommand.features.dataset import build_dataset
from chaincommand.features.engineering import TARGET, feature_columns
from chaincommand.forecasting.intervals import IntervalForecaster

ARTIFACT_DIR = Path("artifacts")


def train_final(df, horizon=1, level=0.8, stop_days=30):
    """Fit point + interval models on all labelled history.

    The last `stop_days` of history are held out only to choose when to
    stop adding trees, then the model is used as-is.
    """
    labelled = df[(~df["is_future"]) & df[TARGET].notna()]
    cut = labelled["date"].max() - pd.Timedelta(days=stop_days)
    fit_part = labelled[labelled["date"] <= cut]
    stop_part = labelled[labelled["date"] > cut]

    model = IntervalForecaster(feature_columns(horizon), level=level)
    model.fit(fit_part, fit_part[TARGET], stop_part, stop_part[TARGET])
    # Point forecast = the median of the two bounds would be crude, so we
    # train a proper point model too.
    from chaincommand.forecasting.gbm import GBMForecaster

    point = GBMForecaster(feature_columns(horizon))
    point.fit(fit_part, fit_part[TARGET], stop_part, stop_part[TARGET])
    return point, model


def forecast_future(df, point, interval):
    future = df[df["is_future"]].copy()
    lo, hi = interval.predict_interval(future)
    out = future[["date", "product_id"]].copy()
    out["forecast"] = point.predict(future)
    out["lower"] = lo
    out["upper"] = hi
    return out.sort_values(["product_id", "date"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--extend", type=int, default=7)
    args = parser.parse_args()

    df = build_dataset(horizon=args.horizon, extend_days=args.extend)
    point, interval = train_final(df, args.horizon)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"point": point, "interval": interval, "horizon": args.horizon},
                ARTIFACT_DIR / "forecaster.joblib")
    forecasts = forecast_future(df, point, interval)
    forecasts.to_csv(ARTIFACT_DIR / "forecasts.csv", index=False)
    print(f"saved model and {len(forecasts)} forecast rows")
    print(forecasts.head(8).to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()