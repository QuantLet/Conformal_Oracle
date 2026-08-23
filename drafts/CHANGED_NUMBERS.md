# Published numbers changed against the previous PDF

Computed by diffing `numbers.tex` against the previous commit and by reading the
prose. **No generated macro changed value** in blocks E–H: the artefacts were not
re-run and the pipeline was not touched.

What changed is where numbers are *printed* and what they are *called*.

| what | before | after | why |
|---|---|---|---|
| §7 opening, size of the reduction | "rather more than half" | "a little over a third", unit named | mixed two non-comparable fractions: 37.4% of the understatement against 63.6% of the truncation depth |
| §7.1, same quantity | "somewhat more than half" | "a little over a third" of the understatement, with the delta-star figure given beside it and named as a different unit | same |
| §7.2, order of magnitude | "overstates the check by an order of magnitude" | one order at the -1.800 edge (factor 11), two at -1.940 (factor 238) | the sentence followed both margins and read as applying to the second |
| Table 4, R-bar column | 3 decimals (0.001) | 4 significant figures (0.001012) | the generated note's ratios 23264 and 84 did not reproduce from the printed precision |
| Table 4, new column | absent | count of assets with negative q-V-stat | the caption asserted twice that the last column held it |
| §1, factor range | "between four and seventy" | "between five and seventy" | the minimum in Table 1 is 0.0200/0.004 = 5 |
| §8.2, zone decomposition | 99 + 2, stated total 102 | 99 already Green + 1 not-already-Green + 2 Green-to-Yellow = 102 | the parts summed to 101 |
| §3.2.4, baseline count | nine | ten | ACI was excluded from the count and included in the table |
| §1 and §7, gate composition | "eight of the ten need nothing but the series and the returns" | seven, with one needing sampled paths and two an evaluation window | Table 3 lists Tail reach as needing sampled paths |
| abstract | 208 words, 6 keywords | 244 words, 5 keywords | DQ moved out of the sigma(V) clause, Basel reworded, DQ panel result added; "Reproducibility" dropped |
| abstract, effective tail probability | "an implied delta near 0.388" | "an effective tail probability near 0.388" | delta was carrying three distinct objects |
| §6, same figure | "its implied delta is near" | "the effective tail probability it cuts off is near" | as above |
| §8.4 quantile scores | 5.03 / 5.81 given without their sample | same figures, now stated to be computed on the dates the two configurations share | they differ from Table 4's per-asset averages, which run over each series' full test window |
| equation (10) | plain empirical quantile | the ceil((w+1)(1-alpha))-th order statistic | contradicted Section 3.2.1 and the Data and Code section |

## New numbers, from computations run this round

| figure | value | source |
|---|---|---|
| dynamic quantile test, raw rejection rate | 81.7% of 312 cells | `analysis/phase3/dq_panel.csv`, macro `\nSeqDQRejRaw` |
| dynamic quantile test, corrected | 53.8% | same, `\nSeqDQRejCor` |
| Z2 pass counts reported in §6 | 0/24 truncated, 17/24 analytic | `Quantlets/CFP_ES_Correction_Z2/tab_es_correction.tex` |

## Not changed, deliberately

Every δ-star figure, every residual understatement, the 0.147 and 0.007 margins,
the 0 of 312 lower-band count, the five unsourced claims, the 79 literals, and
every panel number. The register pass changed voice, not content, and blocks E–H
changed placement and naming, not values.
