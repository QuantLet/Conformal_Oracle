# V3 — expected training optimism for the fixed conformal shift

The authorisation is for a derivation and checks using stored synthetic data.
No manuscript edit, new random history, model fit or financial-panel execution
belongs to this stage. Read PROTOCOL.md, DECISIONS.md and the source-bound
results/theory_loop_v3/RESULTS.md.

The complete proof is docs/theory_loop_v3_20260911/MATHEMATICAL_REVIEW.md.
The independent verifier does not import engine.py: it reconstructs all losses
from saved score arrays, validates the contiguous suffixes and mixture risk,
and recomputes the simultaneous bootstrap through frequency weights.

## Verify existing completion

From the project root in the recorded environment:

```sh
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_theory_loop_v3/finalize.py --verify
```

The original and v2 artifacts must still be available. Their own verifiers run
as part of conservation checking. The root project has no Git metadata; the
bounded protocol has its own local protocol_repository with commit
973132284a16235c780162406cf393031e96e036. That repository contains the exact
protocol committed before the new numerical summaries.

## Replay without overwriting the completion

```sh
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_theory_loop_v3/engine.py --output /private/tmp/irfa-optimism-replay-new
```

Use an empty output directory. All numerical CSVs should match those in
results/theory_loop_v3/diagnostic exactly. The stored replay uses a separate
process and output directory. The original bootstrap indices are reused;
neither a generator nor a financial-data reader exists in this entry point.
Do not run --bind again: it is the one-time pre-execution binding action and
refuses to overwrite the existing lock.

Python, NumPy, SciPy and pandas versions are in completion.json. The check is
for fresh processes in this environment, not a clean installation. Relocation
requires an explicit rebinding of provenance paths to verified identical bytes;
never suppress lock checks. report.py writes canonical diagnostic reports and
should only be rerun as a documented new build, because it updates mtimes.

## Reading the result

The established expansion is about unconditional expected loss. Population
leading A0 differs from exact finite-sample A_n and from the noisy Ahat.
Bands containing one are compatible with a leading approximation, not proof
of equality. The GARCH score process is dependent but its true target hits are
iid, so nonzero hit-covariance effects are covered by the proof alone here.
All failed original feasibility criteria remain failed; none is replaced by
a favourable mean penalty ratio.
