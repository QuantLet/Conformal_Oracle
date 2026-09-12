# Reproducibility ledger

What can be regenerated from committed code, what cannot, and why. One row per
layer, with the check that decides it. A layer is not "reproducible" because a
script exists; it is reproducible when running that script returns the shipped
artefact.

The distinction is not pedantic. Every earlier verification in this repository
tested layer 3 — that the tables follow from the forecast series. None tested
layer 2 — that the forecast series follow from any committed code. GJR-GARCH
passed every layer-3 check for months while being produced by code that is
nowhere in the repository.

| Layer | Artefact | Check | Verdict |
|---|---|---|---|
| 0 | vendor prices → `cfp_ijf_data/returns/*.csv` | re-download | **NOT REPRODUCIBLE BY DESIGN** — archive only |
| 1 | returns → parametric benchmarks | `scripts/verify_producers.py` | **PARTIAL** — see below |
| 1 | returns → TSFM forecasts | GPU re-inference | **UNVERIFIABLE ON THIS MACHINE** |
| 2 | series → tables/figures | `analysis/provenance/build_manifest.py` | **PARTIAL** — 6 DIFFERS, 3 NOT_EMITTED |
| 3 | tables → prose numbers in `main_R2.tex` | `scripts/audit_prose_numbers.py` | **PASSES** — 122/123, the one exception a grant number |

---

## Layer 0 — the return series cannot be re-downloaded

Re-downloading does not reproduce the data, and this is demonstrated rather than
assumed. TLT, ICLN and IBGL are dividend-adjusted ETF histories; vendors restate
them. Their rolling window means no longer match the values implied by forecasts
written on 2026-03-22, with first divergence on 2003-07-29, 2009-06-24 and
2008-12-29. The remaining 21 assets reproduce bit-exactly from the archived CSVs.

So `cfp_ijf_data/returns/*.csv` **is** the primary datum. It is tracked in git,
which is correct, and `download_data.py` is a convenience for new work, not a
reproduction path. Anyone rerunning the pipeline from a fresh download will get
different numbers for those three assets and should expect to.

24 iCloud collision copies (`<ASSET> 2.csv`) were tracked alongside the canonical
files and have been quarantined — byte-identical, verified before the move, and
explicitly tested against the data-vintage hypothesis rather than assumed
irrelevant. See `quarantine/returns_icloud_duplicates/README.md`.

## Layer 1 — the forecast series

`scripts/verify_producers.py` re-runs the committed producer for each parametric
benchmark and compares to the shipped parquet. These four are deterministic
given the returns, so the verdict is exact rather than statistical.

| verdict | meaning |
|---|---|
| `REPRODUCES` | bit-identical (≤ 1e-12) |
| `ROUNDOFF` | ≤ 1e-5 in VaR — four orders below the printed precision of any table |
| `DATA_REVISED` | the input returns changed; layer 0, not a code fault |
| `DIFFERS` | the committed code does not produce the shipped series |

### Historical Simulation — reproduces

21 of 24 bit-identical. The three exceptions are TLT, ICLN, IBGL, and they are
layer-0 data revisions, identified by the rolling window *mean* moving — a pure
function of the input that no quantile convention can shift.

### EWMA — the notebook documents a different estimator than it ships

The notebook computes a truncated 250-day weighted sum; the shipped series is
the RiskMetrics recursion over full history,
`sigma2_t = λ·sigma2_{t-1} + (1-λ)·r_{t-1}²`. The two differ by the discarded
tail, λ²⁵⁰ = 1.9e-7 relative — which is precisely the observed offset (ratio
1 + 1.1e-7, with random scatter of the same order, not a constant factor).
Switching to the recursion cuts the gap from 2.6e-8 to 1.2e-10 on SP500, GOLD
and BTC alike; the residual is insensitive to the seed (λ⁶⁰⁰⁰ erases it) and is
arithmetic precision in the original environment.

Both forms are legitimate EWMA and the numerical consequence is nil — seven
orders below anything printed. What is not legitimate is a notebook that
documents one and ships the other. `rerun_ewma_recursive` is committed as the
identified producer; the notebook has **not** been edited, because it is part of
the frozen submission set and its correction belongs with the GJR decision
rather than being slipped in separately.

### GJR-GARCH — the committed code does not produce the shipped series

This one is material, and it is the reason this ledger exists.

| α | shipped `(VaR−mean)/std` | raw t₅ | standardised t₅ | `norm.ppf` (notebook) |
|---|---|---|---|---|
| 0.01 | −3.36493 | **−3.36493** | −2.60654 | −2.32635 |
| 0.025 | −2.57058 | **−2.57058** | −1.99117 | −1.95996 |
| 0.05 | −2.01505 | **−2.01505** | −1.56077 | −1.64485 |
| 0.10 | −1.47588 | **−1.47588** | −1.14318 | −1.28155 |

The implied quantile takes **one distinct value across all 25 GJR files and
every date**, and it is `stats.t.ppf(α, 5)`: the degrees of freedom hard-coded at
5, and the *raw* t quantile rather than the standardised one, so the series is
too wide by a further √(5/3) = 1.29 even read as a Student-t model. Predicted σ
is sound — predicted/realised is 0.937 against GARCH-N's 0.946 — so the variance
dynamics are fine and only the multiplier is wrong. Coverage is π̂ = 0.0042
against a nominal 0.01.

Every committed version of `pipeline/CFP_Parametric_Benchmarks.ipynb` is
byte-identical (sha256 `30d4a943429c…`) and computes
`mu + sigma * stats.norm.ppf(alpha)`, which yields −2.32635. **No version of the
notebook produces the shipped data.** The parquets were written 2026-03-22; the
earliest commit of any version of that notebook is 2026-04-12. The notebook is a
post-hoc reconstruction that does not reconstruct.

The notebook additionally fits GJR with `dist='skewt'` while taking a Gaussian
quantile — a real defect in its own right, and one that contradicts §3.3 and
Appendix E, which describe GJR-GARCH as a Gaussian-innovation model. That is a
third disagreement: the paper, the notebook, and the data each describe a
different model.

`analysis/gjr_quantile/rebuild_gjr.py` builds both defensible repairs as
candidates — `dist='normal'` (matching the manuscript) and `dist='skewt'` with
the fitted skewed-t quantile (better econometrics, but redefines the benchmark).
Neither is promoted; the choice is a modelling decision, not a bug fix. GJR is
the normalising denominator for the W/GJR column, so either repair moves Table 1
and the §5.3 capital arithmetic.

### TSFM series — unverifiable here

Chronos, TimesFM, Moirai and Lag-Llama came from GPU inference on an A30.
Sampling is not bit-reproducible across backends, so re-running on this machine
answers a different question. These are recorded as `UNVERIFIABLE_HERE`, which
is a statement about the hardware available, **not a pass**. What *can* be
checked without the original hardware is structural, and is what caught the two
confirmed TSFM defects: `scripts/promotion_gate.py` (sign, monotonicity, scale,
α-response, coverage, alignment, dispersion, cardinality, tail reach).

## Layer 2 — series to tables

`analysis/provenance/MANIFEST.md`, one row per table and figure: 6 DIFFERS, 3
NOT_EMITTED, 10 COSMETIC, 6 OK, 9 figures. `NOT_EMITTED` is a reproducibility
gap rather than an erratum — the printed value may well be right, but nothing in
the repository regenerates it.

The manifest's own false-pass hole is closed: a generator that exits 0 without
touching its output file is `NOT_WRITTEN`, not `OK`. That check exists because
the earlier version tested `path.exists()`, which is trivially true for a file
that was already there, and every `OK` verdict issued before it was in principle
unearned.

## Layer 3 — tables to prose

The manifest compares generated `.tex` table files against their submitted
copies. Numbers written inline in the prose were checked by nothing until now,
and both errors found there — the 205/240 claim, which no code path produces and
which has been withdrawn, and the commodity 64%/61% erratum — were caught by
reading, which does not scale and did not generalise.

`scripts/audit_prose_numbers.py` extracts numeric literals from the prose
(tables excluded; the manifest owns those) and looks for each in the emitted
`.tex` and `.csv` artefacts. Current state: **123 candidates, 122 sourced**, the
single exception being the grant contract number `390090/11.11.2025` in the
acknowledgements.

Two limits, stated because a clean result invites over-reading. A match is
necessary, not sufficient — a literal can coincide with an unrelated value
elsewhere in 2.5 MB of artefacts, so this bounds the hand-entered surface rather
than proving provenance. And the haystack must exclude prose documentation and
the script's own output: including them made the check circular and it reported
0 unsourced regardless of the truth. The tell was that *shrinking* the haystack
lowered the unsourced count, which is arithmetically impossible; `SKIP_STEMS`
exists because of it.

## Standing rule

An unexplained asymmetry between forecasters blocks. It does not get filed as a
convention difference, a granularity artefact, or a property of the model. It
gets traced to a line of code or it stays open. Four defect classes have now been
traced this way — sign flip, `top_k=50`, the `CACT`/`FCHI` alias, and the GJR
quantile map — and every one had been visible for months as an anomaly with a
plausible-sounding explanation attached.
