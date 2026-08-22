# Phase 0 pre-registration

Written before running any recomputation. Standing rule 3.

## What I expect to find

**Repository inventory**
1. `submission_IJF/` and `Quantlets/` hold duplicated copies of several producer
   scripts. Expect at least 2 duplicate script pairs that have since diverged.
2. `cfp_ijf_data/` is a large untracked data tree. Expect it to be gitignored and
   therefore not reproducible from the repository alone.
3. Expect at least one artefact with no producer script (an orphan) and at least
   one script whose output path does not exist on disk.
4. `config.json` in `cfp_ijf_data` carries `"model": "chronos_mini"` only — expect
   no global manifest dating the whole forecast vintage. Undated artefacts likely.

**Numeric claims**
5. Macro-backed claims (145 macros in `numbers.tex`) should come back MATCHES,
   because `paper_numbers.py --check` already passes. Expect 0 DIFFERS among these.
6. Prose literals not backed by macros are where I expect failures. The prose
   audit screen says 115 literals are "found in some artefact", but that screen is
   a substring match against a 6 MB haystack and is necessary, not sufficient.
   Expect between 10 and 30 literals that are NOT_EMITTED — i.e. present in the
   text, plausible, but produced by no script.
7. Specific literals I expect to be NOT_EMITTED or NOT_RECOMPUTABLE, from having
   read the sources: the six-pair remainder range (0.109-0.138), the worst
   corrected coverage (0.968) and floor (0.891), the analytic-vs-sampled agreement
   figures (1.041 vs 1.038, 0.3%, 40 dates, 4000 draws), the forward-pass count
   (121,923), the "roughly ten minutes per checkpoint" timing, the COVID detection
   lags (77 and 161 days), the DM counts (18 of 30, 26 of 55), the baseline table
   figures quoted in prose (5.14e-4, 89.1%, 97.4%, 0.0750, 0.0443, 50.6%, 59%),
   the Monte Carlo green frequencies (76-81%, 96-98%), the closure range
   (0.005-1.70) and factor (3 to 76), the capital break-even (1.22) and W range
   (0.98-1.14), the Fisher exact p-values (0.49, 0.00078), and the Wilcoxon
   p-values (9.3e-5, 1e-16).
8. Expect the Basel green counts quoted for the sequence panel (106/312, 154, 52,
   278/312, 34, 309) to live only in a figure caption, hand-entered.

**Cross-references**
9. Expect 0 undefined `\ref` (the last build was clean).
10. Expect a large number of hand-written cross-references, because the main text
    and supplement are separate documents: every `Supplement S.x`, `Table S.xx`,
    `Figure S.x`, `Lemma S.9.1`, `Proposition S.9.4`. Expect 60-100 of these.
11. Expect at least one hand-written reference to a *main-text* number
    (`Table 1`, `Section 5.1`) inside the supplement, which is the failure mode
    that breaks silently when the main text is renumbered. I renumbered tables in
    the previous session, so expect breakage.

## What would falsify each expectation

Each is a count or a named literal. A miss is recorded in
`analysis/phase0/retracted_hypotheses.md` with the evidence.

## Deep-chain pre-registration (written before running)

The macro check above verifies macros <- derived CSVs. It does NOT verify
derived CSVs <- raw forecast series. I will now recompute three headline
quantities directly from `cfp_ijf_data/{model}/{asset}.parquet` and
`cfp_ijf_data/returns/{asset}.csv`, bypassing every intermediate CSV, using the
protocol as stated in the manuscript: contiguous 70/30 split, conformal shift =
empirical (1-alpha) quantile of the calibration scores S_t = q_lo - r_t, applied
as VaR_cp = VaR_raw + qV.

Expected, from macros already in the manuscript:
  E1  raw pi-hat, Chronos-Small-analytic, mean over 24 assets = 0.0175
  E2  raw pi-hat, GJR-GARCH, mean over 24 assets            = 0.0200
  E3  R-bar, Chronos-Small (default), mean over 24 assets    = 17.3

I expect E1 and E2 to reproduce to within rounding. I am less confident about
E3, because R-bar depends on how the per-asset ratio is averaged (mean of
ratios vs ratio of means) and the manuscript does not state which. If E3
misses, that is a specification gap in the paper, not necessarily a defect.
