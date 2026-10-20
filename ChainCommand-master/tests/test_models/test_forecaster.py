"""Tests for demand forecasting models (LSTM, XGB, Ensemble)."""

from __future__ import annotations

from chaincommand.models.forecaster import (
    EnsembleForecaster,
    ForecastModel,
    LSTMForecaster,
    XGBForecaster,
)


class TestLSTMForecaster:
    def test_untrained_predict_empty(self):
        lstm = LSTMForecaster()
        assert lstm.predict("PRD-0000") == []

    def test_is_trained_false_initially(self):
        lstm = LSTMForecaster()
        assert lstm.is_trained is False

    def test_train_and_predict(self, sample_demand_df):
        lstm = LSTMForecaster()
        lstm.train(sample_demand_df, "PRD-0000")
        assert lstm.is_trained is True
        preds = lstm.predict("PRD-0000", horizon=10)
        assert len(preds) == 10
        for p in preds:
            assert p.model_used == "lstm"
            assert p.predicted_demand >= 0

    def test_train_skip_insufficient_data(self, sample_demand_df):
        lstm = LSTMForecaster()
        lstm.train(sample_demand_df, "NONEXISTENT")
        assert "NONEXISTENT" not in lstm._trained


class TestXGBForecaster:
    def test_untrained_predict_empty(self):
        xgb = XGBForecaster()
        assert xgb.predict("PRD-0000") == []

    def test_train_and_predict(self, sample_demand_df):
        xgb = XGBForecaster()
        xgb.train(sample_demand_df, "PRD-0000")
        assert xgb.is_trained is True
        preds = xgb.predict("PRD-0000", horizon=10)
        assert len(preds) == 10
        for p in preds:
            assert p.model_used == "xgboost"
            assert p.predicted_demand >= 0


class TestEnsembleForecaster:
    def test_train_all(self, trained_forecaster):
        assert trained_forecaster._lstm.is_trained
        assert trained_forecaster._xgb.is_trained

    def test_predict_returns_ensemble(self, trained_forecaster):
        preds = trained_forecaster.predict("PRD-0000", horizon=15)
        assert len(preds) == 15
        for p in preds:
            assert p.model_used == "ensemble"

    def test_weights_sum_to_one(self, trained_forecaster):
        w = trained_forecaster._weights.get("PRD-0000", {"lstm": 0.5, "xgb": 0.5})
        assert abs(w["lstm"] + w["xgb"] - 1.0) < 0.01

    def test_get_accuracy(self, trained_forecaster):
        acc = trained_forecaster.get_accuracy("PRD-0000")
        assert "lstm_mape" in acc
        assert "xgb_mape" in acc
        assert "weights" in acc

    def test_compute_mape(self):
        import numpy as np

        actual = np.array([100.0, 200.0, 300.0])
        predicted = [110.0, 190.0, 310.0]
        mape = EnsembleForecaster._compute_mape(actual, predicted)
        assert 0 < mape < 100

    def test_compute_mape_empty(self):
        import numpy as np

        assert EnsembleForecaster._compute_mape(np.array([]), []) is None


class TestForecastModelProtocol:
    def test_lstm_satisfies_protocol(self):
        assert isinstance(LSTMForecaster(), ForecastModel)

    def test_xgb_satisfies_protocol(self):
        assert isinstance(XGBForecaster(), ForecastModel)

    def test_ensemble_satisfies_protocol(self):
        assert isinstance(EnsembleForecaster(), ForecastModel)
