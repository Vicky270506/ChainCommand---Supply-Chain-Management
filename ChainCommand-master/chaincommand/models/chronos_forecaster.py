"""Chronos-2 zero-shot forecaster (v2.0, Section 4.1).

Only activated when the chronos-forecasting package is installed
and settings.enable_chronos is True.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..data.schemas import ForecastResult, utc_now
from ..utils.logging_config import get_logger

log = get_logger(__name__)


class ChronosForecaster:
    """Zero-shot probabilistic time-series forecasting via Chronos-2.

    Conforms to the same train()/predict() interface as LSTM/XGB forecasters.
    """

    def __init__(self, model_name: str = "amazon/chronos-t5-small") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None
        self._history: Dict[str, np.ndarray] = {}
        self._available = False

        try:
            from chronos import ChronosPipeline  # noqa: F401
            self._available = True
        except ImportError:
            log.info("chronos_unavailable", msg="Install chronos-forecasting[torch] to use Chronos-2")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_trained(self) -> bool:
        return len(self._history) > 0

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self._available:
            return

        import torch
        from chronos import ChronosPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ChronosPipeline.from_pretrained(
            self._model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        log.info("chronos_loaded", model=self._model_name, device=device)

    def train(self, history: pd.DataFrame, product_id: str) -> None:
        """Store history for the product (Chronos is zero-shot, no training needed)."""
        series = history[history["product_id"] == product_id]["quantity"].values
        if len(series) < 10:
            return
        self._history[product_id] = series.astype(np.float32)
        log.debug("chronos_history_stored", product_id=product_id, points=len(series))

    def train_all(self, history: pd.DataFrame, product_ids: List[str]) -> None:
        for pid in product_ids:
            self.train(history, pid)

    def predict(self, product_id: str, horizon: int = 30) -> List[ForecastResult]:
        """Generate probabilistic forecasts using Chronos-2."""
        series = self._history.get(product_id)
        if series is None or len(series) == 0:
            return []

        if self._available:
            return self._predict_chronos(product_id, series, horizon)
        return self._predict_fallback(product_id, series, horizon)

    def _predict_chronos(
        self, product_id: str, series: np.ndarray, horizon: int
    ) -> List[ForecastResult]:
        """Use actual Chronos pipeline for prediction."""
        import torch

        self._load_model()
        if self._model is None:
            return self._predict_fallback(product_id, series, horizon)

        context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
        forecast = self._model.predict(context, prediction_length=horizon, num_samples=20)

        # forecast shape: (1, num_samples, horizon)
        samples = forecast.numpy()[0]  # (num_samples, horizon)
        median = np.median(samples, axis=0)
        lower = np.percentile(samples, 10, axis=0)
        upper = np.percentile(samples, 90, axis=0)

        results = []
        for i in range(horizon):
            results.append(ForecastResult(
                product_id=product_id,
                forecast_date=utc_now() + timedelta(days=i + 1),
                predicted_demand=round(float(max(0, median[i])), 1),
                confidence_lower=round(float(max(0, lower[i])), 1),
                confidence_upper=round(float(upper[i]), 1),
                model_used="chronos",
            ))
        return results

    def _predict_fallback(
        self, product_id: str, series: np.ndarray, horizon: int
    ) -> List[ForecastResult]:
        """Statistical fallback when Chronos is not installed."""
        mean = float(np.mean(series))
        std = float(np.std(series))
        trend = float(np.polyfit(range(len(series)), series, 1)[0]) if len(series) > 2 else 0.0

        results = []
        product_seed = hash(product_id) % (2**31)
        rng = np.random.RandomState(product_seed)
        for i in range(horizon):
            predicted = mean + trend * i + rng.normal(0, std * 0.2)
            predicted = max(0, predicted)
            results.append(ForecastResult(
                product_id=product_id,
                forecast_date=utc_now() + timedelta(days=i + 1),
                predicted_demand=round(predicted, 1),
                confidence_lower=round(max(0, predicted - 1.65 * std), 1),
                confidence_upper=round(predicted + 1.65 * std, 1),
                model_used="chronos_fallback",
            ))
        return results
