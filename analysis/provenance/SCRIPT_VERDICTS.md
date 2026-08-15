# Pipeline script verdicts

The seven scripts un-ignored from `b9d94d1`, run and compared against what the
paper publishes. Discipline: track, then run, then verdict — never track and
assume, since both generators examined earlier were stale at nine forecasters.

| Script | Verdict |
|---|---|
| `fz_scores.py` | **Reproduces the paper.** With the ticker fixed it gives TimesFM corrected FZ −2.3002 against the published −2.30, and Moirai-2.0 −2.2643 against −2.26. `results/fz_scores_summary.csv` is **stale** (−3.20, −3.00), predating the TSFM data regeneration. |
| `baselines_evt_fhs.py` | **Reproduces**, to one float ulp (EVT-POT QS 5.509823 vs 5.509824). Derives from returns only, so untouched by the TSFM regeneration. |
| `baseline_aci.py` | **Deterministic and current, but its published row is stale.** See below. |
| `table_c1_es_correction.py` | Runs clean; `tab_es_correction` classified COSMETIC by the manifest. |
| `r4_rolling_and_frontier.py` | Runs clean; regenerates `figures/frontier_all9.pdf` and the rolling intermediates. |
| `qV_block_bootstrap.py` | Now includes Moirai-1.1 and seeds per (asset, model). |
| `baseline_gbm_qr.py` | **Cannot be validated here** — `lightgbm` is absent from this environment though declared in `requirements.txt`. Environment gap, not a package defect. |

## The one finding that inverts the pattern

Every earlier defect in this audit was *a generator behind a correct paper*. ACI
is the reverse.

`baseline_aci.py` is deterministic (two runs byte-identical) and reads current
data. Its recovered input is stale, and **Table 4's published ACI row matches the
stale vintage exactly**:

| | π̂ | Kupiec | QS | width | Green |
|---|---|---|---|---|---|
| Published Table 4 | .013 | 6/216 | **5.37** | **.040** | **96.3** |
| Stale vintage | 0.0127 | — | 5.37 | 0.0397 | 96.3% |
| **Current data** | 0.0128 | — | **5.65** | **.0411** | **94.0%** |

What moved is unambiguous: all 24 rows of every TSFM differ (max ΔQS 6.15),
while the four classical benchmarks differ on 3 rows by ~1e-5. The TSFM
forecasts were regenerated — the same event behind TimesFM's 99% raw violation
rate and the stale FZ summary — and the classical ones were not.

**This is the first published value in the audit that would change on a faithful
re-run.** ACI is one of the alternative recalibration methods behind AE point 5.
Recomputed on current data it looks slightly *worse* than published, so the
correction weakens the AE's objection rather than the paper — but it has to be
restated on the right vintage regardless.

## `CACT` / `FCHI`

Six scripts listed the CAC index under its pre-rebuild ticker. A missing returns
file is skipped, not raised, so each silently produced a complete-looking
23-asset table. Fixed; alias recorded in `Quantlets/cfp_config.py`. Published
figures predate the rename and are unaffected.

This is the `NOT_WRITTEN` failure mode in another guise: exit code 0, plausible
output, wrong content. It is the reason the manifest requires positive evidence
of a write rather than trusting an exit status.

## `tab_tail_closure_extended`

The one differing token is in the **caption**, not the data: the script emits
`R > 1.5` where the paper says `R > 1`. All 96 data tokens are identical. The
script predates the threshold change — which Phase 4 removes entirely, so this
resolves itself.

## Note on `returns/`

`cfp_ijf_data/returns/` holds a byte-identical `* 2.csv` duplicate for all 24
assets — an iCloud sync artefact. Harmless, but they should not ship.
