# Direct-quantile supervised forecasting check

Specification fixed on 10 September 2026 before fitting this benchmark or
reading its forecast losses. This supplementary check answers the request
for a conventional machine-learning base forecaster; it does not change
the ten-model, 240-pair reporting family or the external protocol.

Use all 24 existing return series, their original calendars, the common
512-observation warm-up and the same 70/30 calibration/test split. Predict
the return's 1% conditional quantile directly with LightGBM 4.6.0:
objective=quantile, alpha=.01, 200 trees, learning_rate=.05, num_leaves=15,
max_depth=4, min_child_samples=20, min_child_weight=.001, max_bin=255,
subsample=colsample_bytree=1, subsample_freq=0, reg_alpha=reg_lambda=0,
random_state=20260910, n_jobs=1, deterministic=true, force_col_wise=true.
No early stopping, tuning, fitted distribution, tail extrapolation or
base-model forecast enters the model.

Features are return lags 1--20 and trailing means and sample standard
deviations at 5, 20, 60 and 250 observations, all ending at t-1. Refit at
the first forecast and on the first observed day of each new calendar
year. Use the most recent 1,250 complete feature/target rows strictly
before that date, expanding until that many are available. The initial
fit consequently has 262 rows. Annual refitting differs from the other
models' adaptation schedules and is reported explicitly.

Save every fitted model, tree dump, training interval, feature order,
configuration, input hash and every daily forecast. Repeat the complete
fit in a fresh process. Independently reconstruct features, traverse the
saved trees and recompute forecasts, calibration ranks and test metrics.
The supplementary comparison includes Raw, Static and Rolling250 on all
24 assets, regardless of results.

Use the existing paired calendar bootstrap draws (999 replicates, seeds
20260910+block length, circular blocks of 20 and 60 calendar days). A single
22-contrast family contains GBM Raw minus each of the ten existing Raw
forecasters, GBM Static minus each corresponding Static forecaster, and
GBM Static/ Rolling250 minus GBM Raw. Report centred max-absolute-
standardised simultaneous bands at both lengths in return-unit pinball
loss; retain all contrasts, including unfavourable and inconclusive ones.
Do not promote any favourable subset to the primary comparison.
