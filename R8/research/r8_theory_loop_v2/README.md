# Bounded diagnostic, separate from the failed original theory-loop study

Read `PROTOCOL.md`, `DECISIONS.md` and `results/theory_loop_v2/RESULTS.md` first.
This directory runs neither forecaster inference nor new random histories.
It does not admit the financial-panel deliverables in the original protocol.

The protocol was committed in `protocol_repository` before numerical execution.
The project supplied to this task has no root Git metadata. Its original
`analysis_plan_theory_loop.md`, result manifest and R8 artifacts stay unchanged.

## Existing-artifact verification

From the project root, use the recorded Python runtime:

```sh
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_theory_loop_v2/finalize.py --verify
```

The original files must still be present. The lock binds their bytes and mtimes.
Do not run `prepare.py` over an existing lock. It was the one-time initial
extractor of twelve windows, after the protocol commit.

## Fresh-process replay

Use a new empty output directory for each invocation; retain the canonical
lock. Example paths below intentionally differ from the stored results:

```sh
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_theory_loop_v2/risk_map.py --output /private/tmp/irfa-risk-new
THEORY_LOOP_PYTHON=/private/tmp/irfa-r8-conda-clean/bin/python /usr/local/bin/Rscript --vanilla research/r8_theory_loop_v2/sj_diagnostic.R results/theory_loop_v2/inputs/metadata.csv /private/tmp/irfa-sj-new results/theory_loop_v2/protocol_lock.json
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_theory_loop_v2/analyse_sj.py --sj /private/tmp/irfa-sj-new --output /private/tmp/irfa-density-new
```

Compare all CSV bytes with the canonical risk, SJ and diagnostic outputs. RDS
holds runtime paths, so compare its numerical cases/settings/input records
separately. Stored fresh-process comparisons are under `replay`, `sj_replay`
and `sj_execution`. `runtime.json` and `sj/runtime.txt` record the environment.
This verifies computation in the recorded environment, not installation on a
different machine. Absolute provenance paths must be deliberately rebound and
documented if the archive is relocated; never disable input integrity checks.

`verify_risk.py` independently uses the full-line Stein identity and exact
rational Uniform formulas. It takes `--results`, `--producer`, `--lock` and
`--report`. The R script independently enumerates binned pair counts and
reconstructs both Gaussian derivative sums. The Python density diagnostic
replays densities from the raw binary windows and separates quantile-location
effects from KDE error.

`report.py` produces the standalone two-page PDF. It does not touch any R8
source, figure or manuscript PDF. All generated PDF backgrounds are transparent
and legends sit below the axes. Re-running it changes output mtimes, so use a
separate PDF output path (`--pdf`) when preserving the completion manifest;
its default Markdown report path is canonical and should be regenerated only
as a documented new run.
