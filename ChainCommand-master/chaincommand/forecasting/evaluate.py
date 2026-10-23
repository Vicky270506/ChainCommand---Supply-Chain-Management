"""Fit the baselines on a time split and print their accuracy."""
import argparse
import numpy as np
import pandas as pd
from chaincommand.forecasting.metrics import bias, coverage, mae, rmse, smape
from chaincommand.forecasting.intervals import IntervalForecaster
from chaincommand.features.dataset import build_dataset
from chaincommand.features.engineering import (
    CATEGORICAL,
    TARGET,
    feature_columns,
    time_split,
    rolling_origin_folds
)
from chaincommand.forecasting.baselines import LinearBaseline, SeasonalNaive
from chaincommand.forecasting.metrics import bias, mae, rmse, smape
from chaincommand.forecasting.gbm import GBMForecaster


def linear_columns(horizon=1):
    """Numeric features only. product_id and category_id are labels, not
    quantities, so a plain linear model would read them as 'bigger id = more
    demand'. Tree models in the next phase will handle them properly."""
    return [c for c in feature_columns(horizon) if c not in CATEGORICAL]


def score(y_true, y_pred):
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }


def evaluate_models(train, valid, test, horizon=1, include_gbm=True):
    models = {
        "seasonal_naive": SeasonalNaive(),
        "linear": LinearBaseline(linear_columns(horizon)),
    }
    if include_gbm:
        models["lightgbm"] = GBMForecaster(feature_columns(horizon))
    rows = []
    for name, model in models.items():
        if name == "lightgbm":
            # valid is used only to decide when to stop adding trees
            model.fit(train, train[TARGET], valid, valid[TARGET])
        else:
            model.fit(train, train[TARGET])
        for split_name, part in (("valid", valid), ("test", test)):
            rows.append(
                {"model": name, "split": split_name,
                 **score(part[TARGET], model.predict(part))}
            )
    return pd.DataFrame(rows)

def evaluate_intervals(train, valid, test, horizon=1, level=0.8):
    model = IntervalForecaster(feature_columns(horizon), level=level)
    model.fit(train, train[TARGET], valid, valid[TARGET])
    rows = []
    for split_name, part in (("valid", valid), ("test", test)):
        lo, hi = model.predict_interval(part)
        rows.append({
            "split": split_name,
            "target": level,
            "coverage": coverage(part[TARGET], lo, hi),
            "mean_width": float(np.mean(hi - lo)),
        })
    return pd.DataFrame(rows)

def backtest(df, horizon=1, n_folds=3, valid_days=30):
    """MAE per model on walk-forward folds."""
    rows = []
    for k, (train, valid) in enumerate(
        rolling_origin_folds(df, n_folds, valid_days), start=1
    ):
        # LightGBM needs a stopping set. Carve it from the END of train so the
        # fold's validation window is never used for fitting or stopping.
        cut = train["date"].max() - pd.Timedelta(days=30)
        fit_part = train[train["date"] <= cut]
        stop_part = train[train["date"] > cut]

        models = {
            "seasonal_naive": SeasonalNaive(),
            "linear": LinearBaseline(linear_columns(horizon)),
            "lightgbm": GBMForecaster(feature_columns(horizon)),
        }
        for name, model in models.items():
            if name == "lightgbm":
                model.fit(fit_part, fit_part[TARGET], stop_part, stop_part[TARGET])
            else:
                model.fit(train, train[TARGET])
            rows.append({"fold": k, "model": name,
                         "MAE": mae(valid[TARGET], model.predict(valid))})
    table = pd.DataFrame(rows).pivot(index="model", columns="fold", values="MAE")
    return table.assign(mean=lambda d: d.mean(axis=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args()

    df = build_dataset(horizon=args.horizon)
    if args.backtest:
        print("Walk-forward MAE (3 folds x 30 days)")
        print(backtest(df, args.horizon).to_string(float_format=lambda v: f"{v:.2f}"))
        return
    train, valid, test = time_split(df)
    print(f"train={len(train)} valid={len(valid)} test={len(test)} rows")
    results = evaluate_models(train, valid, test, args.horizon)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n80% prediction interval (LightGBM quantile models)")
    intervals = evaluate_intervals(train, valid, test, args.horizon)
    print(intervals.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()