# conformal-oracle 0.3.1 — release notes

> **Are you affected?** You are affected if you called
> `audit(..., mode='rolling')`, `compute_qv_roll`, or
> `compute_qv_roll_from_scores` on **0.3.0 or earlier**. Check your version with
> `python -c "import conformal_oracle; print(conformal_oracle.__version__)"` —
> upgrade to `0.3.1` and re-run those audits. Static-mode users are unaffected
> to four decimals.
>
> **0.3.0 has been yanked on PyPI** (reason: *rolling-mode under-coverage bug,
> fixed in 0.3.1*). Yanked releases still install if pinned explicitly but are
> skipped by normal resolution; pin `conformal-oracle>=0.3.1`.

## Summary

**0.3.1 is a coverage bugfix release.** The split-conformal correction used the
plain empirical quantile `np.quantile(scores, 1 - alpha)` instead of the
finite-sample split-conformal quantile (the `ceil((n + 1)(1 - alpha))`-th order
statistic; Vovk, Gammerman & Shafer 2005; Lei et al. 2018). The plain quantile
is **one order statistic too small**, so the correction was too small and the
corrected forecast **under-covered** — realised violation rates ran **above**
the nominal level. Upgrading is recommended for any user of the rolling audit.

## Direction of the error

- **Under-coverage.** Corrected VaR was too tight; realised exceedances exceeded
  the target rate `alpha`. Reported Basel/traffic-light zones were therefore
  **optimistic** (too many Green pairs relative to true coverage).

## Where it is material (window sizes)

The error in the quantile *level* is `ceil((n+1)(1-alpha))/n - (1-alpha)`, of
order `1/n`, where `n` is the number of calibration scores the quantile is taken
over. Its practical impact scales inversely with that window:

| Path | Window `n` | Impact at alpha = 0.01 |
|------|-----------|------------------------|
| **Rolling** (`audit(mode='rolling')`, `compute_qv_roll`, `compute_qv_roll_from_scores`) | **250** (default) | **Material.** 249th vs plain ~247.5th order statistic. Illustrative on the 216-pair empirical panel of the companion study (test-split protocol, 250-day Basel window): mean violation **0.016 → 0.010** (target 0.010), Basel Green share **72% → 95%**. These are panel-specific magnitudes — the direction (under-coverage) is general; the size depends on window length, tail shape, and dependence. |
| Static (`audit(mode='static')`, `compute_qv_stat`, `ConformalShift`) | full calibration split (~10³–10⁴) | **Negligible** (< 0.0005 in the quantile level; mean violation moved 0.012 → 0.011, Green unchanged at 88%). Corrected here for consistency. |

Rule of thumb: material for windows up to a few hundred observations; negligible
for calibration sets in the thousands.

## Affected modes / APIs

- `conformal_oracle.conformal.rolling.compute_qv_roll` / `compute_qv_roll_from_scores` — **fixed (materially changes output)**
- `conformal_oracle.audit.audit(..., mode='rolling')` — **fixed (materially changes output)**
- `conformal_oracle.conformal.static.compute_qv_stat` — fixed (output changes negligibly)
- `conformal_oracle.recalibration.ConformalShift` — fixed (output changes negligibly)
- `conformal_oracle.audit.audit(..., mode='static')` — fixed (output changes negligibly)

**Not affected / unchanged:**
- `AdaptiveConformalInference`, `ACICalibrator` (ACI): calibrate on the full
  large-`n` set, and their adaptive miscoverage level is a separate design
  choice — deliberately left unchanged.
- Bootstrap CIs (`bootstrap_qv_ci`): resample the full score set, immaterial.
- Baseline recalibrators reading distributional quantiles (historical
  simulation, EVT-POT, FHS): unaffected — those are return/innovation quantiles,
  not conformal score thresholds.

## Fix

New single source of truth: `conformal_oracle.conformal.quantile.conformal_quantile(scores, alpha)`,
returning the `ceil((n + 1)(1 - alpha))`-th order statistic (falling back to the
maximum score in the conformal `+inf` case, `alpha < 1/(n+1)`). Used by the
rolling, static, and `ConformalShift` correction paths.

## Action for users

- Re-run any **rolling** audit produced with 0.3.0 or earlier; prior rolling
  coverage figures under-state the true violation rate.
- **Static** results are unchanged to four decimal places; no action needed.
- No API changes; drop-in upgrade.

## Tests added

- `tests/test_conformal/test_conformal_quantile.py`: pins the order statistic on
  known cases, guards "never less conservative than the plain quantile", and a
  **behavioral panel regression** — rolling coverage on a fixed 20-series
  synthetic panel must land in `[0.005, 0.012]` (the pre-fix code produced
  ~0.0135 and would fail this bound).
