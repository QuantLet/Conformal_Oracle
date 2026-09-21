# Traffic-light zones under the 1996 binomial rule — protocol

Fixed 14 September 2026, before computation. The manuscript classified each
model--asset pair by the scaled count $250\bar V$ (Green at most 4, Yellow
above 4 through 9, Red above 9). The 1996 backtesting framework (BCBS,
January 1996, Section III and the note to Table 2) prescribes, for samples
other than 250 observations, that the yellow zone begins at the smallest
exception count whose cumulative binomial probability under 99% coverage
equals or exceeds 95%, and the red zone at the smallest count whose
cumulative probability equals or exceeds 99.99%.

Inputs (read-only): `artifacts/r8_ten_comparators/results/pairs.csv`
(columns `n_test`, `viol`, `TL`, `method`, `model`, `asset`). No forecast,
correction or evaluation is recomputed.

Procedure.
1. Reproduce the stored `TL` column with the scaled rule for every row that
   carries a zone; a mismatch stops the run.
2. For each distinct `n_test`, compute the yellow and red starting counts
   under $\mathrm{Bin}(n_{\mathrm{test}},0.01)$ by cumulative sums of the binomial mass evaluated in log space;
   verify the 250-observation anchor (yellow from 5, red from 10).
3. Classify every pair under the binomial rule; count zones per method under
   both rules; record the Green ceiling (largest Green count divided by
   $n_{\mathrm{test}}$) for each window length.

Outputs: `artifacts/r8_basel_binomial/{zones.csv, zone_counts.csv,
ceilings.csv, run.json, RESULTS.md}`; displays through `displays.py`
(`numbers_basel.tex`, `tab_basel.tex`) with `--check`.
