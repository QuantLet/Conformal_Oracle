# R8 theory-loop research stage

The locked primary estimators fail synthetic admission. Financial deliverables
1–3 were not run. No forecaster fitting, market data retrieval or panel inference
is part of these scripts. Current findings: `results/theory_loop/RESULTS.md`.

## Revalidate existing outputs

From the project root, with the recorded Python/R environment:

```bash
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_theory_loop/verify_synthetic.py
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_theory_loop/finalize.py --verify
```

The independent replay includes every R Sheather–Jones fit, all calibration
arrays and aggregate metrics, the common-reference expansion, integration
checks and the failed-admission barrier. It reuses stored synthetic paths;
it cannot read or evaluate a financial panel. The final provenance verifier
checks all file hashes and mtimes; report/validation logs rewritten during
replay may differ in mtime and must not be misrepresented as byte-and-time
unchanged originals. For a strict original-manifest check run finalize first.

## Complete isolated reconstruction

Do not overwrite the original research directory. Copy the required tree into
a fresh working directory, preserving relative paths and the locked plan:

- `analysis_plan_theory_loop.md`, `DECISIONS.md`;
- `research/r8_theory_loop/` and its protocol commit;
- `research/r8_review/` and `research/r8_shape_cost/` for generator and loss checks;
- dependencies imported by `research/r8_review/controlled_comparisons.py`;
- `results/theory_loop/lock.json` and `preflight.json`;
- the files referenced by the preserved R8 receipt if preservation checks run.

The original runtime versions are recorded in `execution_binding.json` and
`synthetic/R_runtime_and_selector.txt`. The implementation uses Python, NumPy,
SciPy, pandas, matplotlib and R stats; no newly fitted financial model is needed.
For the full clean archive, preserve the existing project hierarchy rather than
manually choosing dependencies. In the isolated copy remove only its generated
`results/theory_loop/synthetic` directory and create it empty, then:

```bash
python research/r8_theory_loop/checks.py
python research/r8_theory_loop/run_synthetic.py
python research/r8_theory_loop/verify_synthetic.py
python research/r8_theory_loop/report.py
```

The runner refuses to overwrite an existing admission receipt. Seeds, all five
sample sizes, 500 histories per law, truth simulations, density rules and gates
are fixed. The two logged deviations are part of the final implementation; the
original committed protocol is never rewritten. Run in the recorded environment
for exact binary and R-output replay. Render the two generated PDFs and inspect
them before authoring a fresh final provenance receipt.

## Failure-aware outputs

`preflight.json` and `independent_validation.json` distinguish computational
checks from statistical admission. A check accepts a valid case only after its
constructed faulty counterpart was rejected. `admission.json` says FAIL;
`deliverable_status.csv` says NOT_RUN for the financial work.

The original 1,024-state expansion comparison is retained under
`prior_expansion_definition/`, together with its historical source binding.
It is superseded by the common long-reference evaluation and the corrected
two-density finite-state diagnostic. Do not present that historical comparison
as the current theoretical prediction.

The local Git repository records the plan commit only. It does not replace or
repair historical Git provenance of the much larger manuscript repository.
