# Static contiguous-block loss extension

The current manuscript contains Corollary 4.2 and its proof in S.3.6.
The corollary transfers the expected loss expansion to a fixed correction
evaluated on a contiguous block with H/sqrt(n) tending to infinity.
It does not cover a rolling correction or conditional daily coverage.

`NOTE.md` is the immutable development note written before integration.
Its other arguments (expected-average coverage, a rolling counterexample,
and a Gaussian testing bound) were not promoted to manuscript contributions.
Originality of the contiguous-loss consequence has not been established by
an exhaustive literature review. See `docs/IRFA_CONTIGUOUS_LOSS_BRIDGE.md`.

## Reproduction

Use the project's recorded Python environment. `check.py` originally ran
before manuscript integration and protects that historical source snapshot.
The current-safe deterministic replay below repeats its calculations in a
temporary output directory, checks exact table hashes and compares the
current sources against the separately archived pre-edit snapshot:

```sh
python research/r8_horizon_bridge/validate.py
```

`horizon.py` implements the fixed design in `HORIZON_PROTOCOL.md`. It reads
the existing mechanism paths and replication tables, analytically integrates
the conditional Gaussian AR future and generates no random observations.
Its outputs and exact fresh-process replay are stored in the `horizon` and
`horizon_replay` subdirectories of `artifacts/r8_horizon_bridge`.
To recompute, copy the project to a disposable directory and remove only
those two output directories from that copy, then run:

```sh
python research/r8_horizon_bridge/horizon.py
python research/r8_horizon_bridge/horizon.py --replay
python research/r8_horizon_bridge/validate.py
```

Neither production command overwrites existing outputs. The validator binds
all input and producer hashes. Comparison with archived independent losses
and independent numerical quadrature checks the added future integration.

## Documents and release

The usual R8 document guards and isolated source-package build remain in
force. The release helper creates a new complete archive from the validated
referee archive and current changed files, preserving that preceding ZIP.
It verifies each predecessor member and every newly archived byte:

```sh
python research/r8_horizon_bridge/package_release.py
```

The completion receipt is `artifacts/r8_horizon_bridge/full_release.json`.
No model inference, new data retrieval, publication or submission is part
of this extension.
