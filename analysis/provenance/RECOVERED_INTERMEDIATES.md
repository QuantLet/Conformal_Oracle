# Recovered pipeline intermediates

Two CSVs that `Quantlets/CO_baseline_comparison/compile_tab_baselines.py` reads
were absent from the working tree, blocking Table 3 (the recalibration-method
comparison, including the Diebold–Mariano evidence that classical parametric
models retain a Quantile Score advantage after correction).

Both were **untracked**, not deleted, by commit `b9d94d1` ("Quantlet repo
cleanup: untrack paper sources, figures, results, scripts") — the same cleanup
that hid `scripts/`. They are recoverable from commit `ae79321`:

| File | Recovered from | Notes |
|---|---|---|
| `results/rolling_w250_pooled.csv` | `ae79321` | 216 rows (9 models × 24 assets), rolling w=250 pihat/QS/width |
| `results/aci_baseline_results.csv` | `ae79321` | 216 rows, columns `pi_hat, kupiec_p, qs, width, traffic_light` |

Copies are kept here; the script expects them at `legacy/results/`.

## A substitution trap

`Quantlets/CO_aci_baseline/aci_baseline_results.csv` has the **same filename**
but is a different artefact: 648 rows (216 × 3 gamma values) with columns
`model, asset, method, gamma, violation, kupiec_p, christoffersen_p, zone,
mean_var, var_path_vol`. It is the gamma-sensitivity output and has no `pi_hat`
or `traffic_light` column. Substituting it makes the script fail with
`KeyError: 'pi_hat'` — which is the good outcome; a schema that happened to
overlap would have produced silently wrong numbers.

## Table 3 still does not reproduce

With both files restored the script runs to completion, but its output differs
structurally from the submitted table: 77 numeric tokens against 79, misaligned
from the first token, and the script itself prints *"Regenerated table differs
from committed version"*.

The submitted caption reads: *"Conformal methods are evaluated on all 10
forecasters (240 model–asset pairs); alternative post-hoc methods on the
original 9-forecaster subset (216 pairs)."* The recovered intermediates carry 9
models and 216 pairs. So Table 3 is the same pattern as Table 1: the published
table reflects a ten-forecaster analysis, and the shipped generator was never
updated past nine.

**Status: NOT_EMITTED, not an erratum.** There is no evidence the printed values
are wrong; the generator is behind the paper. Fixing it means extending the
rolling intermediate to Moirai-1.1 — the same one-line dictionary change the
bootstrap needed — and re-running.
