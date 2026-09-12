# Native-tail model extension on the current fund panel

Authorised on 10 September 2026, before full TS-ICL inference and before
recomputing candidate losses on the commodity-fund panel. Earlier Chronos-2
and PatchTST results on the preceding universe have already been observed;
this is a documented extension, not a prospectively untouched test sample.

Use the current 24 histories selected by `r8_commodity_etp/panel_scope.py`,
including USO, GLD and UNG, through each asset's last August 2026 date.
Bind their exact bytes and common forecast/split dates in `support.csv`.
Context: 512 past returns; horizon: one observation; calibration fraction:
floor(0.70 * eligible forecasts). Keep native calendars and all observations.

Candidates are Chronos-2, PatchTST-FM-r1 and TS-ICL. Reuse the first two
models' fully replayed native forecasts (21 unchanged histories plus the
three newly replayed fund histories). TS-ICL uses the pinned source and
checkpoint, unmodified CPU float32 forecasting path, and observation-only
adapter in `r8_grid_candidates/pilot.py`. Run two PyTorch threads per
process, batches of 16 and chunks of 512, seed 20260910. Assets may run in
separate processes; each asset's batching stays fixed. Repeat every output
in a fresh process in the independently recreated environment. Preserve all
99 native quantiles, crossings, contexts and input/output hashes. No
sorting, tail fit, half-normal construction, fine-tuning or new data fetch.

The seven established references remain fixed. Primary endpoint: native
1% quantile. Do not interpolate the candidates' absent 2.5% output. The
existing four-alpha reference study remains separately identified.
Admission requires complete finite outputs, correct past-only interface and
full replay; it does not require favourable calibration or loss. Retain
unfavourable results and document any interface failure before deciding scope.

First evaluate Raw, Static and Rolling250 using unchanged score and
conformal conventions. A nine-model interim evaluation of the two already
complete candidates is permitted and retained separately. Its six contrasts
are unchanged from `r8_grid_candidates/FULL_PROTOCOL.md`. The final ten-model
family adds TS-ICL Static minus Raw and Rolling250 minus Raw, and TS-ICL
minus each other candidate for Raw and Static (twelve contrasts total).
Use 999 paired circular calendar-block draws, lengths 20 and 60, seeds
20260910 + length, equal weighting across assets, and the existing
studentised maximum-deviation simultaneous intervals. Report individual
losses, violations, unconditional/conditional tests and threshold width.

Then transfer the existing `r8_grid_candidates/POLICY_PROTOCOL.md` without
changing its fitting, candidate set, inner bootstrap seeds, thresholds or
reverse-convention diagnostic. Recompute every correction on the common
support. Preserve its five simultaneous policy contrasts. Independently
reconstruct native selections, losses, tests and both bootstrap layers.
Do not update canonical empirical claims before the corresponding complete
evaluation and independent validation pass. Preserve preceding archives.
