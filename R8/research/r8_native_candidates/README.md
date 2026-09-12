# Native-tail candidate experiments

Read `PROTOCOL_PILOT.md`, `PROTOCOL.md` and
`docs/IRFA_NATIVE_MODEL_TESTS.md` first. The pilot protocol and its subsequent
full-history extension have separate hashes; neither is retrospectively
rewritten to match outcomes. The 168-pair paper does not consume these files.

The recorded environments use Python 3.11.16, PyTorch 2.4.1 and Apple MPS.
Package freezes are `artifacts/r8_native_candidates/requirements_chronos.txt`
and `requirements_sundial.txt`. Chronos uses chronos-forecasting 2.3.2 and
Transformers 4.57.6; Sundial uses Transformers 4.40.1. Install these in
separate environments; do not replace the original paper's environment.
The stored checkpoint manifest binds the model revisions and downloaded
code. Inputs are the existing unfiltered return CSVs, independently hashed.
All inference after download is local and offline.

From the repository root, using each model's recorded environment:

```sh
python research/r8_native_candidates/pilot.py --model chronos-2 --replay
python research/r8_native_candidates/full_chronos.py --replay
```

In the Sundial environment:

```sh
python research/r8_native_candidates/pilot.py --model sundial-base-128m --replay
```

The Chronos pilot replay checks all 64 dates; the Sundial replay checks four
dates and their full 1,000-draw arrays. Full Chronos replay compares all
124,894 native quantile vectors exactly and recalculates the 72 method–asset
metric rows. Exactness is established only on the recorded backend and
package versions. Original results are retained in replay mode.

Validate saved files and recompute scoring independently, without model
loading or new inference:

```sh
python research/r8_native_candidates/validate.py --require-replay
```

The validation records central-quantile crossings rather than suppressing
them. A computationally reproduced result is not automatically an admitted
model or a calibrated predictive distribution. Full Sundial evaluation,
Chronos comparator/policy integration and evaluation with a fully aligned
cross-model support remain separate work.
