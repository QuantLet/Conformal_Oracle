# The 35 supplement literals — disposition

Guard 2 of `scripts/build_guards.py` fails on 35 bare decimals in the prose of
`supplement.tex`. Each is resolved below: recomputed from an artefact and lifted
into a macro, declared as a design constant, or corrected because it does not
reproduce.

Written 2026-08-28. Every value in the "recomputed" column was produced in this
session from the artefact named beside it, not carried from earlier context.

## Summary

| disposition | count |
|---|---|
| macro, reproduces exactly from an artefact | 21 |
| derived from the solver's own defaults | 1 |
| **does not reproduce — corrected** | 13 |

None was declared as a constant. The one design choice among the 35 — the
`\delta^\star` grid spacing of 0.004 — is read out of `delta_by_class.py`'s
signature rather than admitted to `DECLARED_CONSTANTS.md`, because the allow-list
is global and 0.005 is already in it for an unrelated reason. That collision is
the third item under "Adjacent" below, and widening the list again would have
added a second one.

## The thirteen that do not reproduce

These are the finding of this item. Two were already known and had never been
carried into the supplement; eleven are new.

### 1. The tuned-GBM ablation is quoted on nine series and runs on thirteen (5 literals)

S.5, `\label{app:gbm_tuned}`. The prose reports `$\hat\pi=.015$ (vs.\ .011),
5/9 Kupiec rejections (vs.\ 0/9), and 88.9\% Green (vs.\ 100\%)`, and adds the
defensive clause "Those counts are over the nine series carried in the tuning
ablation, not the full panel."

`tuned_gbm_qr_grid.csv` carries **104 rows = 8 configurations x 13 models**, and
`tuned_gbm_qr_summary.csv` records `n_pairs = 13` on every row. The ablation
carries thirteen series, so the clause that defends the count is itself false.

Recomputed from the grid, at the two configurations the sentence compares:

| configuration | pi-hat | Kupiec rejections | Green |
|---|---|---|---|
| QS-optimal, `n=100, d=3, eta=0.05` | 0.0155 | **8 of 13** | **11 of 13 = 84.6%** |
| conservative, `n=100, d=3, eta=0.01` | 0.0110 | **0 of 13** | **13 of 13 = 100%** |

The QS improvement is **4.92%**, printed as 5%, and that one is right.

`REPRO_NOTES_E1.md` is the source of the stale figures: it is dated 2026-05-08,
says "8 configs x 9 base models = 72 individual fits", and quotes 5/9 and 88.9%.
The artefact was re-run onto thirteen models and the note was not. 88.9% is 8/9
and corresponds to no count in the current archive.

Note that `.015`, `.011`, `5/9`, `0/9` and `100\%` are invisible to guard 2 --
see "What guard 2 cannot see" below.

### 2. The gate's Monte Carlo resolution: 2.1e-4 is neither a minimum nor a maximum (1 literal)

S.4.1: "The smallest systematic discrepancy that tolerance admits is
$2.1\times10^{-4}$ on mean $\qVstat$."

The gate is `abs(mine[q] - want[q]) > 3*se[q]` applied cell by cell, with
`se[Mean_qV] = sqrt(1/500 + 1/2000) * Std_qV` of the committed study. Recomputed
from `Quantlets/CO_simulation_study/simulation_study_results.csv` (10 cells x 500
replications):

| cell | 3 SE on mean qV |
|---|---|
| normal, T=5000 | **1.271e-04**  <- smallest |
| normal, T=1000 | 2.907e-04 |
| t5, T=5000 | 2.511e-04 |
| t3, T=5000 | 2.995e-04 |
| skewt3, T=5000 | 4.656e-04 |
| t5, T=1000 | 5.806e-04 |
| mixnormal, T=5000 | 6.387e-04 |
| t3, T=1000 | 7.703e-04 |
| skewt3, T=1000 | 1.182e-03 |
| mixnormal, T=1000 | 1.266e-03  <- largest |

A systematic bias survives the gate only if it is below the threshold in *every*
cell, so the smallest discrepancy the tolerance admits is **1.3e-04**, at the
Normal T=5000 cell. 2.1e-04 is the *mean* of the two Normal cells
(2.907 + 1.271)/2 = 2.089e-04. `analysis/k2_sim/GATE_REVISION.md` describes the
same value as "up to 2.1e-04 ... for the Normal cell", which is a maximum; the
maximum over the Normal cells is 2.9e-04. The figure is a mean wearing two
different quantifiers, and neither of them is what it is.

### 3. The score-persistence range is a six-pair range on a four-pair ablation, and one end is pre-correction (2 literals)

S.4.2: "the four pairs tested, which span the score-persistence range
$\hat\rho \in [0.18, 0.67]$."

0.18 and 0.67 are the min and max of `rho_hat` in
`Quantlets/CO_bound_validation/tab_bound_validation.csv`, which has **six** rows
and is a different exercise. Even for that table the range is now
**[0.183, 0.618]**: TimesFM's persistence moved from 0.64 to 0.62 with the
17 August sign correction, and `\nBoundRhoLo`/`\nBoundRhoHi` already carry the
corrected pair.

The four pairs actually tested are in `Quantlets/CO_robustness/gap_ablation.csv`
-- Chronos-Small-A/SP500, Lag-Llama/BTC, GJR-GARCH/WTI, Moirai-2.0/NATGAS -- and
their full-period `rho_hat` are 0.321, 0.047, 0.455, 0.193. The range is
**[+0.05, +0.46]**, and `\nGapAblRhoLo` / `\nGapAblRhoHi` already hold exactly
those values. The macros were computed; the sentence was never repointed at them.

### 4. The constructed pair's backtest p-values have no producer (6 literals)

S.8: "Simulated at $T = 500{,}000$ it returns Kupiec $p = 0.138$ and $0.314$,
Christoffersen independence $p = 0.776$ and $0.625$, ... a dynamic quantile test
at $p = 0.943$ and $0.831$."

Nothing in the repository writes these. `construct_pair.py` solves the linear
programme and writes `pair.npz` and `pair_report.json`; neither carries a
backtest. `sim.npz` holds **20,000** draws, not 500,000, and no test statistic.
Sixth artefact of its kind in this project.

The object exists, so the numbers are recomputed rather than removed: see
`analysis/phase2/pair_backtests.py`.

### 5. The Z_2 power figure has no producer (1 literal)

S.8, same paragraph: "at $T = 1{,}500$ a size-$5\%$ one-sided test calibrated on
the honest model still rejects the alternative with probability $0.079$." The
mean-ES-matched construction this describes is a *fifth* linear constraint that
`construct_pair.py` does not impose, so the alternative law it refers to is not
in the archive either. Recomputed in `pair_backtests.py`.

### 6. A control mean was rounded twice (1 literal)

S.4.1 prints the interpolated quantile's mean at $t_3$, $T = 500$ as
$-0.00071$. `gates.json` holds `-0.0007045492904967753`, which is
**-0.00070** at the five places the sentence uses. `run.log` prints the same
value to six places as `-0.000705`, and 0.00071 is that display rounded a second
time. The smallest of the thirteen, and the only one that is arithmetic rather
than a wrong object or a missing producer.

### 7. The COVID figure's caption describes a different series than the figure draws (1 literal)

S.12: "Realised volatility (right axis, grey) peaks at $\approx 0.98$ annualised
in March~2020."

The grey band is the `rvol` column of
`cfp_ijf_data/paper_outputs/tables/rolling_qv_SP500.csv`, which is exactly a
**250-day** rolling standard deviation annualised by sqrt(252) -- verified to
0.000000 maximum absolute difference against the returns. On the plotted window
(2019-07 to 2021-07) it peaks at **0.351, on 2021-02-08**.

0.98 is real, and belongs to a different estimator: the **20-day** annualised
realised volatility peaks at **0.979 on 2020-03-27**. That date is hard-coded as
`RVOL_PEAK` in `run_covid_response_lag.py` and is the reference point from which
every response lag in the figure is measured. So the figure takes its reference
date from a 20-day realised volatility and draws a 250-day one, and the caption
quotes the first while pointing at the second. Both windows are now named.

## Adjacent, found while resolving the above

- **`tab_h14_small_sample.tex` contradicts itself.** Its table note, hard-coded
  at `run_robustness_mc.py:416`, reads "At small sample sizes ($T \leq 500$),
  corrected violation rates **exceed** the nominal level". The table's own
  `pi-hat` column is 0.007-0.008 against a nominal 0.010 at those sizes, and
  S.4.1 two pages earlier says the rate is "conservative ... rather than
  excessive". The note was true under the plain empirical quantile and survived
  the re-run under equation (8) because it is a string in the emitter, not a
  computed field. PROTOCOL's fourth mode, in generated prose.
- **`numbers.tex` carries wrong values as live macros.** `\nGapVaRAlt` is 1.32,
  `\nGapRhoLo`/`\nGapRhoHi` are 0.18/0.67, `\nGapDMt`/`\nGapDMp` are 0.399/0.69
  -- the pre-correction values, passed through `phase2_numbers.json` unchanged.
  None is currently cited by either document, so nothing is misprinted today.
  They are a loaded trap: the corrected values live under different names
  (`\nPairVaRAlt`, `\nGapAblRho*`, `\nPairDMt`), and a writer reaching for the
  obvious name gets the number this project has already retracted.
- **0.005 passes guard 2 by collision.** In S.6 it is the low end of the
  tail-closure spread (`\nLitClosureRMin`); `DECLARED_CONSTANTS.md` admits 0.005
  as the detection severity cut. The guard allows the literal for the wrong
  reason. `scripts/paper_numbers.py` already names this collision in a comment,
  and the supplement site was never migrated.

## What guard 2 cannot see

The guard's extractor is `(?<![\w.\\])(\d+\.\d+)`, applied after
`\begin{tabular}...\end{tabular}` is replaced by a placeholder. Three classes of
prose literal are therefore outside its field of view, and the tuned-GBM
paragraph above sits in all three at once:

| class | example, from the live supplement | why it is missed |
|---|---|---|
| leading-dot decimals | `$\hat\pi=.015$ (vs.\ .011)` | the pattern requires a digit before the point |
| integers and ratios | `5/9 Kupiec rejections`, `278 of 312 pairs`, `88\% Green` | the pattern requires a decimal point |
| hand-authored tabulars | Table S.8's `-1.320`, `0.046`, `42.3\%` | `tabular` is stripped wholesale, on the assumption that tables arrive by `\input` from a declared emitter |

The third is the one that matters most: the strip exists because generated tables
are guard 5's business, but a `tabular` written by hand in `supplement.tex` is
prose in a table's costume, and guard 2 is the only check that would have read it.
This is the "cannot see" mode of `PROTOCOL.md` Rule 2 -- a check that fails
informatively, over a region the defect can hide in -- in the guard written for
exactly this class of defect.

### Measured, then closed

The third class was measured before it was repaired: over both documents there
are **six** hand-authored tabulars carrying **thirteen** undeclared decimals.
Eleven are column widths (`p{3.5cm}`, `p{1.3cm}`), which are typesetting lengths
and correctly not claims. The remaining two are measured results:

- `main_R2.tex` Section 8, the failure-mode table: **0.990** and **0.988**, the
  1% violation rates of TimesFM 2.5 and Moirai 2.0 while their lower quantile was
  stored with an inverted sign. Both reproduce as cell means over the 24 assets
  from `analysis/recompute/sign_verification.csv` (0.989958 and 0.987989). The
  column beside them was **already a macro** — `\nRawPiTimesFM`,
  `\nRawPiMoiraiTwo` — so the corrected rate travelled as a macro and the
  defective rate as a literal, in the same table row, for as long as the table
  existed. Guard 2 reported "no bare decimal literals in prose" over them
  throughout.
- `supplement.tex` S.11, the gate table: **99.9\%**, the monotonicity band. A
  declared gate band, and now in `DECLARED_CONSTANTS.md` beside the other four,
  together with the extremes multiplier of 50.

`_strip_tabular_specs()` replaces the tool. It removes a tabular's **column
specification** by brace-scanning and keeps the body, so lengths stay stripped
for the same reason `\includegraphics` options are, and rows are read. Generated
tables are unaffected: `\input{...}` is replaced earlier in the same function,
so no emitted table ever reaches this point and guard 5 keeps its jurisdiction.

The guard's negative control now plants a literal **inside a tabular whose
column specification carries lengths**, alongside the original prose control.
The old control passed on a guard blind to every hand-authored table in the
project, which is the whole of Rule 2's argument turned on the guard itself: the
control has to fail at the resolution of the defect, and a prose-only control
cannot fail at the resolution of a defect that only occurs in tables.

The first two classes remain outside the guard. Leading-dot decimals now number
**zero** in both documents — the `.015` and `.011` of the tuned-GBM paragraph
were the only ones, and they are closed — but the pattern still cannot see them,
so they are a hazard for the sections not yet written rather than a live defect.
Integer claims are check 5's jurisdiction in `audit_structural_claims.py`, which
covers the "N of M" form and does not cover a bare percentage.
