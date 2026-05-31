"""Test that TSFM submodule imports without optional deps."""

from __future__ import annotations


def test_tsfm_submodule_importable():
    """Importing the tsfm submodule must not raise."""
    from conformal_oracle.forecasters import tsfm

    assert hasattr(tsfm, "BaseTSFMForecaster")
    assert hasattr(tsfm, "TSFMPredictionCache")


def test_base_class_importable():
    from conformal_oracle.forecasters.tsfm._base import BaseTSFMForecaster

    assert BaseTSFMForecaster is not None


def test_cache_importable():
    from conformal_oracle.forecasters.tsfm._cache import TSFMPredictionCache

    assert TSFMPredictionCache is not None


def test_chronos_module_importable():
    from conformal_oracle.forecasters.tsfm import chronos

    assert hasattr(chronos, "ChronosForecaster")
    assert hasattr(chronos, "PAPER_MODELS")


def test_lag_llama_module_importable():
    from conformal_oracle.forecasters.tsfm import lag_llama

    assert hasattr(lag_llama, "LagLlamaForecaster")
    assert hasattr(lag_llama, "PAPER_MODEL")


def test_timesfm_module_importable():
    from conformal_oracle.forecasters.tsfm import timesfm

    assert hasattr(timesfm, "TimesFM25Forecaster")
    assert hasattr(timesfm, "PAPER_MODEL")
    assert hasattr(timesfm, "QUANTILE_LEVELS")


def test_moirai_module_importable():
    from conformal_oracle.forecasters.tsfm import moirai

    assert hasattr(moirai, "MoiraiForecaster")
    assert hasattr(moirai, "PAPER_MODELS")


def test_forecasters_init_no_crash():
    """Main forecasters __init__ should import without TSFM deps."""
    from conformal_oracle.forecasters import (
        HistoricalSimulationForecaster,
    )

    assert HistoricalSimulationForecaster is not None

    try:
        from conformal_oracle.forecasters import (
            GARCHNormalForecaster,
            GJRGARCHForecaster,
        )

        assert GJRGARCHForecaster is not None
        assert GARCHNormalForecaster is not None
    except (ImportError, AttributeError):
        pass
