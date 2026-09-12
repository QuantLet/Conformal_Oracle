# Current R8: ten native-tail models and the July external test

The canonical article is `source/main_R8.tex`; its companion is
`source/supplement_R8.tex`. The main native-tail comparison contains ten
models and 240 pairs on common 512-warm-up dates. The reference analyses
retain seven models and 168 pairs for four tail levels, broader baselines,
gap sensitivities, windows and historical origins. The industry test is
separate: four models, twelve portfolios, 48 pairs through July 2026.
The main market endpoint remains August. The referee revision leads the
theory with the expected-loss proposition and its oracle-shrinkage
consequence. Corollary 4.2 now extends that loss expansion to the expected
average of a growing contiguous static test block, without an unused gap.
Its proof is in S.3.6. This does not extend the separated-point coverage
theorem to rolling forecasts. The conditional GARCH application is a remark.
Proposition 4.3 adds an exact finite-sample counterexample: identical score
margins and all pairwise laws can accompany opposite correction effects.
Its complete proof is in S.3.7. The witness is close to break-even and its
loss differences are small. Coverage is now Theorem 4.8 and the GARCH
application Remark 4.9.

The reporting family now contains eight contrasts, including Raw and Vol-ERM.
The static-minus-raw simultaneous bands include zero at both block lengths.
The projected-DtACI implementation is a diagnostic; its contrast remains in
the multiplicity adjustment. The original external protocol is unchanged.
An additional direct-quantile supervised GBM is evaluated separately on the
same 24 assets, with its own 22-contrast family. It does not enlarge the
primary 240-pair correction family. See `research/r8_referee_revision`.

## Rebuild documents from validated results

Use the recorded analysis environment (Python 3.13.9, NumPy 2.3.5,
pandas 2.3.3, SciPy 1.16.3, arch 8.0.0, pyarrow 21.0.0) and TeX Live:

```sh
python source/scripts/extension_20260831/build_paper_outputs.py
python research/r8_integration/build.py
python research/r8_regime/build_paper.py
python research/r8_ten_integration/build.py
python research/r8_ten_integration/build.py --check
```

From `source`, run `latexmk -g -pdf -interaction=nonstopmode -halt-on-error`
on `main_R8.tex`, then `supplement_R8.tex`, then both again. From the
project root run `python source/scripts/extension_20260831/validate_r8.py`.
This checks the reference displays and the thirteen new displays, including
exact regeneration, corruption controls, numerical literals, producer
assignments, references and clean compilation. The new display builder
also verifies that the result matrices are the independently validated ones.

`package_sources.py` produces the current portable source ZIP and compares
both PDF texts with an isolated clean build. `package_release.py` preserves
the previous numerical release and writes a new full archive containing
the updated dependency closure, with every archived file hashed and checked.
Neither command uploads or submits anything.

## Reproduce the research calculations

The model-inference and common-support instructions remain in
`research/r8_model_extension`; the stronger comparisons are in
`research/r8_ten_comparators/README.md`. The external data, all fresh base
fits, correction replays and independent validations are described in
`research/r8_external/README.md`. These use saved input responses and
pinned checkpoints; a new live data retrieval is a different input vintage.

Historical research validators protect their recorded pre-integration
manuscript hashes. Preserve that safeguard. Make a disposable project
copy outside the current project, then, **from the original project**, run:

```sh
python research/r8_ten_integration/restore_research_snapshot.py /tmp/r8-ten-replay
```

Run the ten-model stronger-comparator fitting/replay validators inside
that copy. The helper restores the 46 archived document files only in
the copy and never changes the current paper. Earlier mechanism/regime
validators have their separate historical staging instructions. The
external validators do not depend on manuscript text.

The external result is deliberately retained despite failing to establish
any of the seven protocol-specified loss-transfer advantages. All four
correction and three indication simultaneous bands include zero at both
block lengths. This is not a reason to change its models or thresholds.
The stored per-pair matrices and period breakdowns accompany the main
results. Public deposit and final submission assessment remain separate.

## Referee extension and GBM replication

The added inference uses existing paths and the original paired calendar
draws. The GBM uses LightGBM 4.6.0 and scikit-learn, with all configuration
and fitted trees archived. Both have independent numerical validators.
Their protocol, run order, immutable-output handling and separate archive
instructions are in `research/r8_referee_revision/README.md`.

The subsequent contiguous-loss extension, deterministic proof checks and
analytic future-loss calculation on the saved AR histories are documented
in `research/r8_horizon_bridge/README.md`. Its release helper preserves the
referee ZIP and writes the preceding complete archive. The current extension,
its exact certificate, deterministic 48-cell diagnostic and full-release
helper are documented in `research/r8_count_law/README.md`. Its archive
preserves both earlier releases. Main and external empirical results remain
unchanged.
