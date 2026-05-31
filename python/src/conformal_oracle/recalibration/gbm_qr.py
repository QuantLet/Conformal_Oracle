"""Gradient-boosted quantile regression recalibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


class GBMQuantileRegression:
    """Gradient-boosted quantile regression via LightGBM.

    Features: raw VaR forecast and lagged realised volatility
    (5-day and 20-day rolling standard deviation).

    Default hyperparameters are QS-optimal from the paper's
    grid search (Appendix F.1):
        n_estimators=100, max_depth=3, learning_rate=0.05
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.20,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._early_stopping_rounds = early_stopping_rounds
        self._val_fraction = val_fraction
        self._model: object | None = None
        self._alpha: float = 0.01

    @staticmethod
    def _make_features(
        raw_var: np.ndarray,
        realised: np.ndarray,
    ) -> np.ndarray:
        vol5 = (
            pd.Series(realised)
            .rolling(5, min_periods=1)
            .std()
            .fillna(0.0)
            .values
        )
        vol20 = (
            pd.Series(realised)
            .rolling(20, min_periods=1)
            .std()
            .fillna(0.0)
            .values
        )
        vol5_lag = np.concatenate([[0.0], vol5[:-1]])
        vol20_lag = np.concatenate([[0.0], vol20[:-1]])
        return np.column_stack([-raw_var, vol5_lag, vol20_lag])

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "GBMQuantileRegression requires lightgbm. "
                "Install with: pip install conformal-oracle[gbm]"
            ) from e

        self._alpha = alpha
        X = self._make_features(raw_var_forecasts, realised)
        y = realised

        n_val = max(int(len(y) * self._val_fraction), 30)
        X_tr, y_tr = X[:-n_val], y[:-n_val]
        X_vl, y_vl = X[-n_val:], y[-n_val:]

        params = {
            "objective": "quantile",
            "alpha": alpha,
            "learning_rate": self._learning_rate,
            "num_leaves": 2 ** self._max_depth - 1,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

        self._model = lgb.train(
            params,
            dtrain,
            num_boost_round=self._n_estimators,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(
                    self._early_stopping_rounds, verbose=False,
                ),
            ],
        )

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Must call fit() before apply()")

        n = len(raw_var_forecasts)
        dummy_realised = np.zeros(n)
        X = self._make_features(raw_var_forecasts, dummy_realised)
        pred = self._model.predict(
            X, num_iteration=self._model.best_iteration,
        )
        return -pred
