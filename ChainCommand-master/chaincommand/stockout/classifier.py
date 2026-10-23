"""LightGBM stockout classifier. Threshold is chosen on validation, scored on test."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score,
)

from chaincommand.features.dataset import build_dataset, get_engine
from chaincommand.features.engineering import time_split
from chaincommand.stockout.features import (
    STOCKOUT_FEATURES_FULL, add_incoming_features, add_stockout_features, load_orders,
)
from chaincommand.stockout.labels import LABEL, add_stockout_label


def build_stockout_table(engine=None, horizon_days=7):
    engine = engine or get_engine()
    df = build_dataset(engine)
    df = add_stockout_label(df, horizon_days)
    df = add_stockout_features(df, horizon_days)
    return add_incoming_features(df, load_orders(engine))


def labelled_splits(df):
    # time_split filters on demand, not on the label, so drop unlabelled tails here
    return tuple(p.dropna(subset=[LABEL]) for p in time_split(df))


def best_threshold(y_true, prob):
    """Probability cut-off that maximises F1 (use validation data only)."""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


class StockoutClassifier:
    def __init__(self, feature_cols=STOCKOUT_FEATURES_FULL, params=None,
                 num_rounds=1000, early_stopping=50):
        self.feature_cols = list(feature_cols)
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping
        self.params = {
            "objective": "binary", "metric": "average_precision",
            "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 100,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "verbosity": -1, "seed": 42,
        }
        if params:
            self.params.update(params)
        self.threshold = 0.5
        self.model = None

    def fit(self, X, y, X_valid, y_valid):
        pos = float(np.sum(y))
        # Class imbalance: weight positives by neg/pos. This distorts the raw
        # probabilities, which is why the threshold is tuned afterwards.
        params = {**self.params, "scale_pos_weight": (len(y) - pos) / max(pos, 1.0)}
        train_set = lgb.Dataset(X[self.feature_cols], label=y)
        valid_set = lgb.Dataset(X_valid[self.feature_cols], label=y_valid, reference=train_set)
        self.model = lgb.train(
            params, train_set, num_boost_round=self.num_rounds, valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
        )
        self.threshold = best_threshold(y_valid, self.predict_proba(X_valid))
        return self

    def predict_proba(self, X):
        return self.model.predict(X[self.feature_cols],
                                  num_iteration=self.model.best_iteration or None)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)


def report(y_true, pred, prob=None):
    out = {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }
    if prob is not None:
        out["PR_AUC"] = average_precision_score(y_true, prob)
    return out


def main():
    df = build_stockout_table()
    train, valid, test = labelled_splits(df)
    print(f"train={len(train)} valid={len(valid)} test={len(test)} rows; "
          f"positive rate train={train[LABEL].mean():.3f} test={test[LABEL].mean():.3f}")

    clf = StockoutClassifier().fit(train, train[LABEL], valid, valid[LABEL])
    print(f"threshold chosen on valid: {clf.threshold:.3f}")

    rows = []
    for split, part in (("valid", valid), ("test", test)):
        y = part[LABEL]
        prob = clf.predict_proba(part)
        rows.append({"model": "lightgbm", "split": split, **report(y, clf.predict(part), prob)})
        rule_margin = (part["stock_plus_incoming_minus_expected"] < 0).astype(int)
        rows.append({"model": "rule_stock_lt_demand", "split": split, **report(y, rule_margin)})
        rule_cover = (part["lead_time_gap"] > 0).astype(int)
        rows.append({"model": "rule_cover_lt_leadtime", "split": split, **report(y, rule_cover)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()