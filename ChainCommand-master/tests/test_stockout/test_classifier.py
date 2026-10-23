import numpy as np
import pandas as pd

from chaincommand.stockout.classifier import StockoutClassifier, best_threshold
from chaincommand.stockout.features import STOCKOUT_FEATURES_FULL


def test_best_threshold_separates_clean_scores():
    t = best_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 < t <= 0.8


def test_classifier_learns_rule_with_rare_positives():
    rng = np.random.default_rng(4)
    n = 3000
    X = pd.DataFrame(rng.random((n, len(STOCKOUT_FEATURES_FULL))), columns=STOCKOUT_FEATURES_FULL)
    y = (X["stock_cover_days"] < 0.1).astype(int)  # about 10% positive
    tr, va, te = slice(0, 2000), slice(2000, 2500), slice(2500, n)
    clf = StockoutClassifier(params={"min_data_in_leaf": 10, "learning_rate": 0.1})
    clf.fit(X[tr], y[tr], X[va], y[va])
    pred = clf.predict(X[te])
    tp = int(((pred == 1) & (y[te] == 1)).sum())
    assert tp / max(int(pred.sum()), 1) > 0.8   # precision
    assert tp / int(y[te].sum()) > 0.8          # recall