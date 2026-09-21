# Native-grid technical pilots — 10 September 2026

The author approved local technical pilots of IBM Granite PatchTST-FM-r1
and EDF TS-ICL after the additional-model source screen. This protocol is
written after source inspection and before model inference. It does not
select models using a pilot loss or change the canonical 168-pair panel.

## Fixed inputs and outputs

- Existing SP500 and BTC return files in
  `artifacts/extension_20260831/data/returns`, unchanged and individually hashed.
- Last 32 observed dates per asset through 31 August 2026; 512 immediately
  preceding observed returns; horizon one observation on each native calendar.
- Float32 contexts, no cleaning, outcome-based selection, cross-asset inputs,
  covariates, fine-tuning, tail fit, or added half-normal construction.
- Save all 99 trained quantiles and directly select the native 0.01 entry.
  Record crossings; do not sort, clip or repair forecast values.
- Seed 20260910, inference/evaluation mode, two CPU threads. Local checkpoint
  loading only; automatic checkpoint downloads disabled during inference.

## Immutable model and implementation versions

| Item | Revision |
|---|---|
| ibm-granite/granite-timeseries-patchtst-fm-r1 | 151f9c6d576281b95c2ff784d0863bd3f12c80f1 |
| ibm-granite/granite-tsfm | fe7a35697723e2a2f5246ae979474bfc554e26c0 |
| taharnbl/TS-ICL | 19c94031439fb31f36ce395088ee50a6762d3774 |
| EDF-Lab/ts-icl | 349f3eae4f01f78536b16a6ea53c0837760166ec |

Source archives, model cards, configurations and weights are downloaded and
hashed separately. Separate Python 3.12 environments respect the incompatible
upstream torch/scikit-learn requirements. Record complete package locks.

PatchTST uses the public tensor-input prediction wrapper, requesting horizon
one and all configured quantiles. Its shipped implementation internally
reconstructs at least 128 future positions for this checkpoint, then selects
the first requested position. Only 512 past returns are supplied. Padding
and future placeholders are based on the past mean and excluded from fitted
normalisation statistics. Use the shipped pruning and MPS bfloat16 autocast;
record effective settings and compare selected cases against CPU float32.
If MPS is unavailable or unsupported, archive the failure and use CPU with
its precision explicitly recorded. Do not silently modify model computation.

TS-ICL uses the forecasting component, context_length=512, horizon one,
denormalize=True, no automatic imputation/covariate forecasting and the
official CPU device path. Require both checkpoint state dictionaries and
strict loading. Its public level selector multiplies requested levels by
100 and compares exact floating-point equality. If this rejects trained
levels in the full grid, capture the full output of the unchanged forecasting
routine immediately before the public selector slices it. Request the
supported levels 0.01, 0.50 and 0.99 through the public API and require exact
agreement with their entries in the captured grid. This observer may not
interpolate, extrapolate or alter the head. Record any native sorting performed by
the shipped model separately from any audit of its unsorted head.

## Verification and interpretation

1. Assert context length, dates, finiteness, hashes, exact native level and
   checkpoint loading; perturb the target and all future returns and verify
   identical extracted contexts.
2. Check complete output shape and values; retain every forecast and crossing.
3. Capture internal inputs/masks or normalisation and confirm the observed
   values used by the forecasting component are precisely the past context.
4. Repeat all 64 forecasts in a fresh process and compare every native entry;
   report exact equality or the actual maximum difference, without assuming
   cross-device/precision identity.
5. Compare first/last cases per asset in batches and individually; report
   absolute and relative numerical differences. Check native 1% extraction
   against the public selected-level call and inverse scaling against the
   shipped implementation.
6. Save source/weight/input/output hashes, environment, seeds, runtime,
   diagnostics, failures and limitations. Validate saved outputs separately
   from inference. Estimate full-history runtime without extrapolating loss.

These pilots assess execution, provenance and computational feasibility.
Sixty-four dates cannot establish 1% coverage or rank forecasting accuracy.
Past-only contexts do not establish absence of overlap with pretraining.
Full-history evaluation, common-support comparison, corrections, policies
and uncertainty must follow under a separately recorded evaluation protocol.
