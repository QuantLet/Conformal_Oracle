"""Render the research summary from validated numerical result files."""
import json
import pandas as pd
import engine as e


def main():
    data=pd.read_csv(e.OUT/'contrasts.csv')
    primary=data[(data.family=='primary') & data.primary].sort_values('h_factor')
    all_cells=data[data.family=='primary']
    relative=(100*(all_cells.mean_delta-all_cells.leading_prediction).abs()
              /all_cells.leading_prediction.abs())
    fin=pd.read_csv(e.ROOT/'artifacts/r8_shape_cost/financial/summary.csv')
    matched=fin[fin.population=='matched161'].set_index('state')
    high,other,total=(matched.loc[s] for s in ['high','other','all'])
    inference=json.loads((e.ROOT/'artifacts/r8_shape_cost/financial/inference_status.json').read_text())
    rows=[]
    for _,r in primary.iterrows():
        rows.append('| '+('Smaller error' if r.h_factor==.5 else 'Larger error')
          + f' | {r.mean_delta:.8f} | [{r.simultaneous_lower:.8f}, {r.simultaneous_upper:.8f}]'
          + f' | {r.leading_prediction:.8f} |')
    text='''# Shape, estimation cost and the value of tail recalibration

11 September 2026. Completed research extension authorised by the author's
“go”. The canonical R8 manuscript and preceding financial results remain
unchanged. The new mathematical candidates, simulation and financial
decomposition are archived separately for review and subsequent integration.

## Main result

The fixed simulation supports the predicted change in sign. When the local
forecast error is small, the volatility-scaled ERM correction has higher
expected loss than the constant ERM correction. With a larger error, the
ordering reverses. Both corrections estimate one coefficient from the same
observations using the same original-return pinball objective.

The registered primary conditions use Normal innovations, 1,000 calibration
observations and a static contiguous test horizon of 428 observations. Each
point averages 5,000 independent calibration histories, integrating future
loss over the known innovation and scale-transition laws. Differences below
are Vol-ERM minus Shift-ERM, in the simulation's return units; negative values
favour Vol-ERM.

| Primary condition | Estimated loss difference | Simultaneous 95% Monte Carlo band | First-order prediction |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

Both registered sign conditions pass the predeclared simultaneous criterion.
All twelve cells in the primary family have intervals on the predicted side
of zero; these cells share histories and are not twelve independent studies.

The quantitative approximation is less accurate. Absolute discrepancies
relative to the leading prediction range from {relative.min():.1f}% to
{relative.max():.1f}% across the twelve cells. The high-error primary prediction
lies outside its Monte Carlo interval. This supports a qualitative cost
mechanism, not quantitative equivalence or a ready-to-use financial cutoff.

![Predicted and observed loss differences](../../artifacts/r8_shape_cost/figures/shape_cost.png)

The figure scales differences by calibration size. Dashed predictions are
not fitted to the simulation. The plot has a transparent background and its
legend is outside, below the panels.

## What the result adds

For the bounded known-scale model, the proposed comparison separates the
additional error removed by the scaled correction from the additional cost
of its fitted coefficient. This makes the cost argument testable while
holding coefficient count fixed. The proof candidates specify continuous
innovation density, moments, predictability and joint geometric mixing;
the contiguous result keeps fitted coefficients fixed throughout the horizon.

The cost is estimator-specific. Return-loss Vol-ERM uses scale-weighted
quantiles; an unweighted quantile of standardised scores has a different
leading cost. The separate conformal/rank sensitivity confirms why the
primary result must not be transferred indiscriminately: Vol-CP's low-error
Normal n=1,000 band includes zero. It does not establish the same significant
sign crossing for every conformal estimator. Both sensitivity families and
all methods are retained in the numerical files.

This is a controlled study motivated by earlier development. It is not a new
external financial test, an estimated financial decision rule, or a general
claim that scaling is novel. The earlier two adverse quadratic-approximation
examples remain unchanged. The strongest defensible addition is an explicit
estimator-cost prediction with a transparent finite-sample test and limits.

## Financial context

The existing-path study compares Shift-ERM with Vol-ERM on the {int(total.pairs)} reference
pairs having both predefined volatility states. In high states, mean
violations move from {100*high.shift_pi:.3f}% to {100*high.vol_pi:.3f}%; in other
states they move from {100*other.shift_pi:.3f}% to {100*other.vol_pi:.3f}%.
Normalised QS improves by {-high.delta_normalized:.8f} in high states and
worsens by {other.delta_normalized:.8f} in other states. The whole-horizon
matched difference is {total.delta_normalized:.8f}. These are descriptive
retrospective findings.

The proposed ratio bootstrap cannot support its planned confidence bands on
this full matched population: empty pair-state denominators occur in
{inference[0]['draws_with_empty_cells']} of {inference[0]['total_draws']} draws
at 20-day blocks and {inference[1]['draws_with_empty_cells']} at 60-day blocks. Under the rule fixed
before new loss calculations, both entire inference families are suppressed.
No difficult pair or bootstrap draw was dropped, and no replacement procedure
was chosen after seeing the result. This limits this inference design; it
does not prove no financial effect or impossibility of all alternative tests.

Signed-threshold and realised-overshoot accounting explains where the loss
changes. The result measures risk forecasts, not bank capital, cash savings,
Expected Shortfall or trading profits. See [the financial report](FINANCIAL_RESULTS.md).

## Validation and scope

- [Mathematical audit](MATHEMATICAL_VALIDATION.md): source-bound proof and
  production-formula audit before simulation outcomes.
- [Statistical validation](STATISTICAL_VALIDATION.md): independent replay of
  all 5,000 histories, 360,000 saved coefficients including Raw, conditional
  losses and both simultaneous families.
- [Independent financial audit](FINANCIAL_INDEPENDENT_REVIEW.md): original
  calendars, calibration scales, daily signed-threshold identities and
  empty-state inference handling.
- [Reproduction instructions](../../research/r8_shape_cost/README.md): code,
  sealed design, stored histories, financial input closure and validation.

The archive receipt is produced only after extraction, member-hash checks and
replay of both validators. Passing these checks establishes the recorded
computations; it does not certify external validity or editorial acceptance.
No canonical manuscript/PDF edits, new financial forecasts, market-data
downloads or foundation-model inference were performed in this extension.
'''
    (e.ROOT/'docs/shape_cost_20260911/SUMMARY.md').write_text(text)


if __name__=='__main__':
    main()
