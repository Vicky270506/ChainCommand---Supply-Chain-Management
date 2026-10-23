"""LightGBM point forecaster with native categorical handling."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from chaincommand.features.engineering import CATEGORICAL


class GBMForecaster:
    def __init__(self, feature_cols, params=None, num_rounds=2000, early_stopping=50):
        self.feature_cols = list(feature_cols)
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping
        self.params = {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 42,
        }
        if params:
            self.params.update(params)
        self.categories = {}
        self.model = None

    def _prepare(self, X, fitting=False):
        X = X[self.feature_cols].copy()
        for c in CATEGORICAL:
            if c in X.columns:
                if fitting:
                    # Remember the category set from train so valid/test
                    # are encoded identically.
                    self.categories[c] = sorted(X[c].unique())
                X[c] = pd.Categorical(X[c], categories=self.categories[c])
        return X

    def fit(self, X, y, X_valid=None, y_valid=None):
        train_set = lgb.Dataset(self._prepare(X, fitting=True), label=y)
        valid_sets, callbacks = [], []
        if X_valid is not None:
            valid_set = lgb.Dataset(
                self._prepare(X_valid), label=y_valid, reference=train_set
            )
            valid_sets = [valid_set]
            callbacks = [lgb.early_stopping(self.early_stopping, verbose=False)]
        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_rounds,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return self

    def predict(self, X):
        raw = self.model.predict(
            self._prepare(X), num_iteration=self.model.best_iteration or None
        )
        return np.clip(raw, 0, None)

    def importance(self):
        gain = self.model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self.feature_cols, "gain": gain})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )