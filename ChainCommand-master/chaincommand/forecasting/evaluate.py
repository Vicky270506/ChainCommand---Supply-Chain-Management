"""Fit the baselines on a time split and print their accuracy."""
import argparse

import pandas as pd

from chaincommand.features.dataset import build_dataset
from chaincommand.features.engineering import (
    CATEGORICAL,
    TARGET,
    feature_columns,
    time_split,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    df = build_dataset(horizon=args.horizon)
    train, valid, test = time_split(df)
    print(f"train={len(train)} valid={len(valid)} test={len(test)} rows")
    results = evaluate_models(train, valid, test, args.horizon)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()