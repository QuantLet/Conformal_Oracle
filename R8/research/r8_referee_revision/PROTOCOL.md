# Referee-requested reporting-family expansion

Fixed before calculating the expanded bands on 10 September 2026.
Use the existing 240 model–asset pairs, fixed paths, calendar support,
scales, 999 paired calendar draws, seeds, block lengths 20 and 60,
and centred max-absolute-standardised-deviation convention.

The reporting family contains eight differences relative to Shift-CP:
Raw, Vol-ERM, State-L1, POT-Shift, POT-Vol, projected-DtACI expected loss,
Loss-gate and Past-minimum. This adds Raw and Vol-ERM to the original six.
Report both block lengths, return-unit losses, calibration-scale
normalisation, and exclusion of Bitcoin/Ethereum without selecting a
preferred result. Static-minus-raw bands reverse the sign and endpoints
of the Raw-minus-Shift-CP contrast in this same family.

Retain the projected-DtACI contrast in multiplicity adjustment even if its
row is moved to an implementation diagnostic. Do not narrow the family
after seeing bands. Preserve the original six-contrast outputs. This is
a requested extension of retrospective reporting, not an originally
prespecified family or new model/forecast run. The external protocol and
its original families remain unchanged.
