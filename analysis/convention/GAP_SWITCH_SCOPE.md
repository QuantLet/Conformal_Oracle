# Switching the panel to the gapped estimator: what it actually costs

Written 2026-08-31, after starting the switch and stopping it.

## The decision is right and the measurement supports it

Measured on the pipeline's own `conformal_backtest`, at $\alpha = 0.01$ over the
312 cells: Basel green **278 → 278**, Kupiec passes **230 → 229**, **0** zone
changes, **1** Christoffersen verdict, max $|\Delta\hat\pi|$ **0.000521**. Over
all four levels and 1,248 cells: **0** zone changes, **11** Kupiec flips.

Both flips at $\alpha = 0.01$ are marginal in both arms:

| flip | contiguous | gapped |
|---|---|---|
| Kupiec, EWMA / STOXX | $p = 0.0531$ | $p = 0.0486$ |
| Christoffersen, GJR-GARCH / WTI | $p = 0.0530$ | $p = 0.0455$ |

EWMA/STOXX keeps 24 violations while the window loses 14 observations, so
$\hat\pi$ moves from 0.015219 to 0.015355 and the $p$-value crosses 0.05 by
0.0014. Neither cell is a substantive change of verdict; both are a threshold
being crossed by a hair.

## Why the switch stopped

The driver was patched, the gapped panel written, and the primary consumers
regenerated. Then: `scripts/regenerate_rolling_vs_static.py` takes its own
chronological split at line 87 and imposes no gap. So do the scripts behind the
deterioration counts, the zone ledgers and the gate ledger. Switching the driver
alone leaves Table 1 on the gapped estimator and the entire cost-and-indication
argument of Section 9 on the contiguous one --- two estimators in one paper,
which is worse than either.

**76 split sites across 64 files** take a chronological calibration/test split.
Excluding `submission_IJF/` (frozen) and `legacy/`, the live count is
**about 45**. Every one that feeds a reported number must impose the same gap,
or the panel is mixed.

This is the same shape as the four quantile conventions of
`QV_CONVENTION.md`: a definition that lives in one place and is re-implemented
everywhere it is used. That defect took a config function, a migration of every
site, and an audit that fails the build on an undeclared one. The gap needs the
same three things.

## What the repository is left in

Consistent, on the contiguous split. `all_results.csv` is restored from the
pre-switch copy and `run_full_evaluation.py --verify` reproduces it exactly.

What is kept from the attempt:

- `separation_gap()` and `conformal_backtest_gapped()` in the driver, with the
  per-pair $\hat\rho$ of Corollary 4.6 and the fallback documented;
- `--gap`, **opt-in and not the default**, with a docstring saying why;
- `measure_gap_on_pipeline.py` and this file.

## What a correct switch requires

1. `cfp_config.separation_gap(scores, n_cal)` as the single definition, beside
   `conformal_quantile`.
2. Migrate the live split sites to take the gap from it. About 45.
3. Extend `audit_qv_convention.py` to fail the build on a split site that does
   not declare whether it imposes the gap --- the instrument that stops this
   recurring.
4. Regenerate, then re-verify: 36 macros move, and the chains checked by hand
   need rechecking. Known movers: `MainKupiecCorPasses` 273 → 272,
   `MainCCPass` 97 → 96, `MainCCPassPct` 53.9 → 53.3. `MainGreen` does not move.
5. Section 4.4 rewritten as a section about a condition imposed and its cost,
   under a new title; the contiguous panel becomes the robustness exhibit,
   replacing the four-pair ablation of S.4.2 with the 1,248-cell comparison.

Steps 1--3 are the work. Step 4 is an afternoon of verification. The compute is
seconds.

## The methodological note this earned

The first evidence for the switch came from `run_gap_panel.py`, which re-split
the stored scores itself and reported 297 green where the pipeline reports 278.
Its paired differences were right and its levels were its own. Deciding on that
would have been the right decision reached from a quantity computed outside the
emitter that owns it --- the same mechanism as the 4.80 variance ratio, twice in
one day. The rule is in `PROTOCOL.md`: a derived quantity is computed by the code
that owns the object, not reconstructed beside it.
