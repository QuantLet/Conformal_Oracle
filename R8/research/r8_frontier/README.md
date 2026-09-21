# R8 risk-bridge development

This directory develops a proposed connection between dependence, estimation
cost and the decision to recalibrate. It does not modify the validated R8
manuscript, supplement or empirical results. Read `RISK_BRIDGE.md` for the
mathematical statements, assumptions, proofs and novelty limits, and
`../../docs/IRFA_TOP_PAPER_RESEARCH_DIRECTION.md` for the editorial assessment.

## Reproduce

Use the R8 statistical environment described in
`release/R8_20260909/README.md`, built from
`artifacts/extension_20260831/quality/conda-analysis-minimal-explicit.txt`
plus `requirements-conda-overlay.txt` in the same directory. From the
repository root, with that environment active:

```sh
python research/r8_frontier/check_risk_bridge.py
MPLCONFIGDIR=/tmp/irfa-frontier-mpl XDG_CACHE_HOME=/tmp/irfa-frontier-cache python research/r8_frontier/refresh_chain.py
```

The first command reads the original, hash-checked constant-distortion
simulation outputs in `artifacts/review_20260909/complexity_mc`. It does not
invoke their producer or generate new paths. The second command uses an
exact finite-state recursion and deterministic integration; it does not
sample the Markov chain. The declared calculation grid is recorded in
`REFRESH_CHAIN_PROTOCOL.md`.

Outputs are written under `artifacts/r8_frontier`. Two validation JSON
files bind the numerical outputs to producers and inputs. The figure is
available as transparent PNG and SVG, with its legend below the panels.
The SVG omits the creation timestamp and uses a fixed ID salt.

The mathematical statements and all printed empirical interpretations
still require scientific review. Numerical checks do not establish their
novelty, primitive assumptions for a financial process, or the performance
of an implementable deployment rule. The supplied manuscript's existing
theorem is not extended to contiguous or rolling recalibration.
