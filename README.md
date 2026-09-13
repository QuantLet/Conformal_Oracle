# Conformal Recalibration of Extreme Tail Quantiles under Temporal Dependence

The current R8 replication package is in [`R8/`](R8/). It includes the
intermediate numerical results needed to regenerate the paper's tables and
figures, the original producers, a locked Python environment specification and
a command that verifies the regenerated files against the deposited outputs.

```sh
conda create -y --prefix /tmp/r8-env --file R8/environment-osx-arm64.explicit.txt
/tmp/r8-env/bin/python -m pip install --no-deps -r R8/requirements-conda-overlay.txt
/tmp/r8-env/bin/python R8/reproduce.py replay --workdir /tmp/r8-replay --report /tmp/r8-replay.json
```

These exact binary builds target macOS on Apple Silicon, including FreeType
2.13.3. Package version pins alone do not fix the graphics renderer; details are
in the R8 instructions. Choose a new work directory for each replay. The command starts without any
pre-existing output tables or figures, runs all eight display producers, and
compares all 61 generated files byte for byte. It also runs the statistical
unit tests, the exact rational counterexample, and corruption controls.

- [R8 instructions and scope](R8/README.md)
- [Reproducibility status and validation](R8/REPRODUCIBILITY.md)
- [Input/output manifest](R8/REPLAY_MANIFEST.json)
- [Python estimator package](python/README.md)

The reproducible display release is tag **`R8-2026-09-12-repro1`**. The original
`R8-2026-09-12` tag is preserved as a historical deposit. The replay uses archived
intermediate results; it does not retrain foundation models or rerun the full
simulation and forecasting pipeline. Full native draws and fitted-path archives
remain a separate research archive.

Older Quantlets and the `data-v1` download belong to earlier revisions. They are
retained for historical work and are not inputs to the R8 command above. The
previous project description is archived in `R8/history/`.

## Extensions of 13 September 2026

Tag **`R8-2026-09-13`** adds [`R8/extensions_20260913/`](R8/extensions_20260913/):
the prespecified power analysis of the main-panel loss contrast, the
nuisance-free optimism-estimator study, and the second external test on the
Developed ex-US 25 size/value portfolios, each with protocol, producer,
`--check` replay, outputs and results. The display replay of
`R8-2026-09-12-repro1` is unchanged.

## Authors

Daniel Traian Pele, Vlad Bolovăneanu, Andrei Theodor Ginavar,
Stefan Lessmann and Wolfgang Karl Härdle.

## License

Source code is distributed under the repository's MIT license. Financial data
originate from the providers identified in the manuscript and archived metadata.
