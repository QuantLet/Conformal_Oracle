# main_R2.tex — figures still on the ten-forecaster panel

Table 1 now covers 13 forecasters and 312 pairs. These lines still quote the
10-forecaster / 240-pair panel and must be reconciled before submission. They
are NOT wrong for the analyses they describe (Phases 1, 2 and the bound were all
computed on the 240-pair panel); the question in each case is whether the
sentence refers to the panel Table 1 now shows, or to the original analysis.

| line | text | verdict needed |
|---|---|---|
| 189 | "240 model--asset pairs" (contribution 2) | refers to the audit panel -> 312 |
| 338 | "Across all 240 model--asset pairs, empirical coverage exceeds the bound" | Theorem 3.3 check, computed on 240 -> keep, but say "the ten forecasters of the original panel" |
| 434 | "240 model--asset pairs" (Section 4.1) | -> 312 |
| 469 | "203/240" Green | -> 260/312 (83.3%) |
| 542 | "203/240" Green (rolling comparison) | rolling was computed on 9 models; state the panel explicitly |
| 711 | "240 model--asset pairs" (Data and Code Availability) | replication matrices cover 240; keep or regenerate for 312 |
| 1044 | "all ten forecasters" (appendix) | -> thirteen, or scope to the original panel |

Also outstanding:

- Appendix J forensic material is referenced from the new Section 4.2.1 but was
  not physically moved into the main text, as Phase 5 specifies.
- The rolling-recalibration results (Section 4.5) were computed for 9 models and
  have not been extended to CAViaR/GAS-t or to the corrected Moirai-1.1 input.
- Table 4's ACI row is on a superseded TSFM vintage (see SCRIPT_VERDICTS.md);
  recomputed it is QS 5.65 / Green 94.0% against the printed 5.37 / 96.3%.
- 5 overfull hboxes, largest 33.9pt.
