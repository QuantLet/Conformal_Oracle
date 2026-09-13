"""Exercise R7 APIs on deterministic toy inputs, not manuscript simulations.

Run with conformal-oracle 0.4.0 installed. No model or external data is needed.
"""

import numpy as np

from conformal_oracle import (
    SeparatedSplitConformalVaR,
    proxy_separation_gap,
    recalibration_indication,
    selectively_recalibrate,
)


def main() -> None:
    t = np.arange(1500, dtype=float)
    returns = 0.01 * np.sin(t / 7) + 0.005 * np.cos(t / 3)
    quantiles = np.full(len(t), -0.003)
    n_cal = int(0.70 * len(t))
    cal_returns, cal_quantiles = returns[:n_cal], quantiles[:n_cal]
    gap = proxy_separation_gap(cal_quantiles - cal_returns, context_length=12)
    separated = SeparatedSplitConformalVaR(
        gap=gap.gap, minimum_evaluation_size=10,
    ).split(returns, quantiles)
    contiguous = SeparatedSplitConformalVaR(gap=0).split(returns, quantiles)
    assert separated.q_v_stat == contiguous.q_v_stat
    assert separated.evaluation_indices[0] == n_cal + gap.gap
    assert not gap.certified

    decision = recalibration_indication(
        calibration_returns=cal_returns, calibration_quantiles=cal_quantiles,
    )
    selected = selectively_recalibrate(
        quantiles[n_cal:],
        calibration_returns=cal_returns, calibration_quantiles=cal_quantiles,
        decision=decision, method="rolling", window=250,
        evaluation_returns=returns[n_cal:],
    )
    assert decision.apply
    assert selected.final_quantiles.shape == quantiles[n_cal:].shape
    assert np.isfinite(selected.final_quantiles).all()
    print(f"proxy gap={gap.gap}; certified={gap.certified}")
    print(f"same static shift={separated.q_v_stat == contiguous.q_v_stat}")
    print(f"apply={decision.apply}; reasons={decision.reasons}")

    # Two violations in 250 observations: both raw calibration checks pass.
    good_returns = np.ones(250)
    good_returns[:2] = -1
    good_quantiles = np.zeros(250)
    skip = recalibration_indication(
        calibration_returns=good_returns, calibration_quantiles=good_quantiles,
    )
    raw = np.array([-0.0, -0.01, -0.02], dtype=np.float32)
    unchanged = selectively_recalibrate(
        raw, calibration_returns=good_returns,
        calibration_quantiles=good_quantiles, decision=skip, method="rolling",
    )
    assert not skip.apply
    assert unchanged.final_quantiles.dtype == raw.dtype
    assert unchanged.final_quantiles.tobytes() == raw.tobytes()
    print("skip preserves raw dtype and bytes=True")


if __name__ == "__main__":
    main()
