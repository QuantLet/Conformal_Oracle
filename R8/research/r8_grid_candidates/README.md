# Reproducing the native-grid pilots and full PatchTST evaluation

Results and interpretation: `docs/IRFA_GRID_MODEL_PILOTS.md`.
The main protocol predates inference; `NUMERICS.md` records the subsequent
precision experiment before it ran. Neither experiment changes paper results.

## Existing environments on this machine

```sh
/private/tmp/irfa-grid-tsicl/bin/python research/r8_grid_candidates/pilot.py --model tsicl --device cpu --replay
/private/tmp/irfa-grid-patchtst/bin/python research/r8_grid_candidates/pilot.py --model patchtst --device mps --replay
/private/tmp/irfa-grid-patchtst/bin/python research/r8_grid_candidates/precision_check.py --replay
/private/tmp/irfa-grid-tsicl/bin/python research/r8_grid_candidates/validate.py
```

Inference is offline. MPS calls require access to the Mac GPU, which the
Codex sandbox does not provide. The producer reads only the pinned local
source, weights and existing return files. Replay verifies all 6,336 native
values in each configuration. The first pilot uses the shipped mixed
precision; `precision_check.py` is the recommended float32 configuration.

## Recreating the environments

Use Python 3.12 (original: 3.12.14), then run:

```sh
python3.12 research/r8_grid_candidates/bootstrap.py --model patchtst --env-dir /private/tmp/irfa-grid-patchtst-recreated
python3.12 research/r8_grid_candidates/bootstrap.py --model tsicl --env-dir /private/tmp/irfa-grid-tsicl-recreated
```

The archived package locks retain the original host-specific local project
reference for provenance. The bootstrap omits that reference and installs
the exact dependency versions; inference imports the immutable source tree
directly. Thus the original Granite build's incomplete package discovery
does not control model loading. The local build label `0.0.0+fe7a356` is a
source-build identifier, not an official Granite release number. Original
installation logs and both successful `pip check` results accompany the
locks. The bootstrap was executed into two new environments; both dependency
checks pass, and `environment_replay.py` reproduces all 6,336 native values
exactly for each selected configuration (PatchTST float32 and TS-ICL CPU).

Reuse the archived weights and source tarballs. `fetch.py` can retrieve their
immutable revisions again, but downloading is unnecessary for replay and
overwrites the retrieval receipts. Do not run it over the preserved evidence.
Verify hashes first. Same-backend replay is exact on this machine; another
device, library build or precision can differ and must be checked explicitly.

`validate.py` uses NumPy/pandas only, checks every saved forecast against its
source date/context, verifies the code against the tarballs and independently
checks float32 GPU/CPU agreement. It does not load or run a model.

## Output layout

- `artifacts/r8_grid_candidates/models`: immutable checkpoint files/cards.
- `sources`: immutable tarballs and extracted source trees.
- `source_review`: retrieval metadata, original install logs, exact package
  locks and execution logs.
- `patchtst`: official MPS mixed-precision pilot/replay and four CPU cases.
- `patchtst_float32`: separate MPS float32 pilot and exact replay.
- `tsicl`: CPU pilot and exact replay, including captured native grids.
- `validation.json`: independent artefact-check receipt.

The original directories above preserve the technical pilots. The subsequent
full run uses `patchtst_full` and `common_evaluation`, described below. None
of these candidate outputs is yet integrated into the canonical article.

## Full PatchTST and common-date evaluation

Design: `FULL_PROTOCOL.md`. Interpretation and validation status:
`docs/IRFA_PATCHTST_FULL_EVALUATION.md`.

Full native replay against the existing archive uses a fresh process in the
recreated environment:

```sh
/private/tmp/irfa-grid-patchtst-recreated/bin/python research/r8_grid_candidates/full_patchtst.py --replay
```

This compares all 12,364,506 saved native values, dates and positions exactly.
It also checks context, input, source and checkpoint hashes. Omit `--replay`
only to create an absent production archive or resume an interrupted one;
do not recreate completed production receipts over preserved evaluation
bindings. For a from-scratch reproduction, use a separate copy with the
generated `patchtst_full` and `common_evaluation` output directories absent.
Retain all input archives, manifests, protocols and source files.

The common-date comparison is generated once in that fresh copy by:

```sh
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_grid_candidates/evaluate_full.py
```

It uses frozen reference forecasts, including the corrected Lag-Llama
actual-calendar archive, and recomputes all corrections on common dates.
It refuses to overwrite a completed comparison. The analysis environment
uses Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3 and pyarrow 21.0.0;
the existing full environment specification is
`artifacts/extension_20260831/quality/conda-analysis-explicit.txt`, with its
adjacent `requirements-conda-overlay.txt`. Actual run versions are also
recorded in `common_evaluation/complete.json`.

Independent reconstruction, without running models or importing the
production metric helpers, and figure regeneration:

```sh
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_grid_candidates/validate_full.py
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_grid_candidates/plot_full.py
```

The validator requires completed full inference replay. It verifies all
daily thresholds, backtests and bootstrap draws, as well as the original
Chronos result agreement and the monitored article/supplement hashes.
The four-file `common_evaluation_protected.json` is an editing-session
preservation check, not an input needed to define the statistical method.
Figures are PNG/PDF/SVG with transparent backgrounds and legends outside
below each plot. Complete comparison outputs, native inference and plotting
have separate hash receipts.

## Existing decision rule on common dates

`POLICY_PROTOCOL.md` fixes the subsequent transfer of the existing past-loss
gate, including its original one-sided bootstrap convention. It uses the
same nine base models and 24 asset calendars. The reverse-centred convention
is a sensitivity only. See `docs/IRFA_NATIVE_POLICY_NEXT_STEP.md` for results,
the literature check and the next scientific steps.

```sh
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_grid_candidates/policy_full.py
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_grid_candidates/validate_policy.py
```

Production refuses to overwrite `policy_evaluation/complete.json`; run it
in a separate copy with that output folder absent for a fresh production
replay. The independent validator reconstructs every fitted correction,
decision, evaluated threshold, inner resample and panel interval from the
preserved inputs. Both commands use the same analysis environment as above.
The previously recorded `research/r8_decision/methods.py` is imported
unchanged. No base-model inference or new return simulation is performed.
