# Phase 3 — estimation windows and dynamic benchmarks

## 3a. AE point 7: does a longer window remove the need for the correction?

The AE's hypothesis: 250 observations is noisy, and with longer windows "the
results might stabilize a lot (such that re-calibration has a lower (or even
negative) effect)".

**Confirmed for one model of four, and refuted for the rest.** Mean gain from the
correction, ΔQS ×10⁴, across 24 assets:

| Model | w = 250 | w = 500 | w = 1000 | direction |
|---|---|---|---|---|
| GJR-GARCH | +0.116 | +0.150 | +0.153 | **rises** |
| GARCH-N | +0.150 | +0.167 | +0.131 | flat |
| EWMA | +0.218 | +0.242 | +0.281 | **rises** |
| **Hist-Sim** | +0.097 | +0.012 | **−0.038** | **falls, turns negative** |

The mechanism separates cleanly, and it is visible in the raw coverage:

| Model | π̂ raw, w=250 | w=500 | w=1000 | raw Green |
|---|---|---|---|---|
| GJR-GARCH | 0.0189 | 0.0187 | 0.0187 | 9 → 7 → 9 |
| GARCH-N | 0.0193 | 0.0191 | 0.0188 | 8 → 8 → 11 |
| **Hist-Sim** | **0.0158** | **0.0137** | **0.0123** | **12 → 20 → 19** |

For Historical Simulation a longer window genuinely fixes calibration — raw π̂
moves from 0.0158 towards nominal 0.0123 and raw Green rises from 12 to 19 of
24 — and once it is fixed the correction has nothing to do and begins to hurt.
That is exactly the AE's mechanism, and it should be conceded in those terms.

For the parametric models raw π̂ barely moves across a fourfold increase in
window length. Their miscalibration is therefore **not sampling noise; it is
specification.** A Gaussian-innovation GARCH under-covers the 1% tail no matter
how much data its parameters are estimated on, and no window length reaches it.

So the honest answer to AE point 7 is: the objection identifies a real effect and
locates it precisely — in the nonparametric estimator, where the correction is
substituting for window length. It does not hold for the parametric benchmarks,
where longer estimation windows leave the tail defect intact.

**Convergence.** Zero failures in ~360,000 GARCH fits across all three window
lengths (128,211 fits at w=250, 122,211 at w=500, 110,211 at w=1000 per model).
The documented EWMA fallback was never invoked. EWMA and Historical Simulation
involve no optimisation, so their zero is trivial.

## 3b. Referee 1 point ix: dynamic VaR benchmarks

Exactly two, as agreed: CAViaR (Engle & Manganelli 2004) in both the symmetric
absolute value and asymmetric slope specifications, and a score-driven GAS-t
model. Same assets, same 70/30 split, same α, same conformal correction applied
on top. Parameters estimated once on the calibration segment, then filtered
through the test segment without re-estimation.

| Model | π̂ raw | π̂ corr | QS corr ×10⁴ | Green corr | R̄ |
|---|---|---|---|---|---|
| GARCH-N | 0.0193 | 0.0104 | **4.737** | 21/24 | 0.171 |
| **CAViaR-AS** | **0.0110** | 0.0109 | 4.770 | 20/24 | **0.0010** |
| **CAViaR-SAV** | **0.0114** | 0.0114 | 4.843 | 18/24 | **0.0011** |
| EWMA | 0.0208 | 0.0114 | 4.852 | 20/24 | 0.184 |
| GJR-GARCH | 0.0042 | 0.0112 | 4.855 | 19/24 | 0.155 |
| **GAS-t** | 0.0333 | 0.0117 | 4.895 | 19/24 | 0.321 |
| Lag-Llama | 0.0294 | 0.0095 | 5.243 | 24/24 | 0.357 |
| Moirai-1.1 | 0.0154 | 0.0105 | 5.281 | 21/24 | 0.106 |

**CAViaR is already calibrated raw**: π̂ = 0.0110 and 0.0114 against a nominal
0.01, closer to nominal than any model in the paper. Its **R̄ is 0.001**, two
orders of magnitude below the paper's previous minimum, and the correction moves
nothing — Green 20→20 and 18→18, QS unchanged in the fourth decimal.

That is the cleanest confirmation available of the Phase 1 result. A forecaster
built to track the conditional quantile directly needs no unconditional
correction, and applying one is free because there is nothing to correct. The
Phase 1 gate rule skips it.

**It does not overturn the ranking.** CAViaR-AS sits between GARCH-N and EWMA:
competitive, not dominant. The claim that classical parametric models retain a
Quantile Score advantage over the TSFMs survives with CAViaR added, and the best
model remains GARCH-N.

**GAS-t behaves like the paper's benchmarks**: raw π̂ = 0.0333, Green 1/24 → 19/24
after correction, R̄ = 0.32. A case where the correction does real work.

## What Phase 3 does to the argument

R now spans **0.001 (CAViaR — needs nothing) to 23.5 (Chronos-Mini — replaced
entirely)**, and the correction's value tracks that span monotonically. Adding
the benchmarks the referee asked for widened the axis the repositioned paper is
built on rather than threatening it.
