# WTI contract P&L reconstruction — 10 September 2026

The author authorised replacing the exclusion-based treatment of WTI with
an economically defined exposure that retains the April 2020 shock. This
is a retrospective specification change prompted by that event. It is not
a strategy selected without knowledge of the episode. All existing results
remain archived until complete replacement inputs and forecasts validate.

## Exposure and target

Use one long NYMEX CL contract (1,000 barrels), marked to daily settlement.
Roll at settlement on the penultimate exchange session before each
contract's last trading date. Use published contract expirations and an
explicit exchange-session calendar, including historical exceptions. The
rule applies to every month; no event-specific roll overrides are allowed.
The position held from the preceding settlement determines today's P&L:

    pnl_usd[t] = 1000 * (F[t, held_contract] - F[previous, held_contract]).

On a roll day, mark the outgoing contract first, then replace it with one
incoming contract at that day's settlement. Never count the price
difference between two maturities as a gain or loss. Both settlements at
the roll date must be available. This is a frictionless marked exposure,
not evidence of executable fills in stressed liquidity, a collateral
model, or an economically optimal roll policy. Transaction costs and
collateral interest are omitted and must be stated.

Report USD P&L/VaR per contract. An optional fixed USD 100,000 reference
capital provides a dimensionless scale, not a realised account-equity
return or a claim about required margin. Do not average that arbitrary
capital-scaled loss into the other assets' log-return QS. WTI must have a
separate exposure-specific evaluation unless a common economically
defensible panel normalisation is explicitly established. Retain results
on both sides of the shock; no shift, absolute-price denominator,
interpolation, winsorisation or replacement of negative settlements.

## Data admission

Required daily unadjusted settlement data: `date`, `contract`,
`settlement_usd_per_barrel`, with source provenance. Required contracts:
`contract`, `delivery_month`, `last_trade_date`. Required exchange calendar:
`date`. Derive the penultimate-session roll date from that calendar, not
from observed prices. The contract schedule must span the full sample.

The intended complete interval is the existing WTI price-history start
through 31 August 2026. Every held contract needs consecutive settlements;
every roll needs overlapping maturities. Missing values, duplicate keys,
uncovered sessions and incomplete endpoints stop admission. Expiry dates
must be supported by the provider/exchange; do not infer them from a
generic continuous price series or price jumps.

Yahoo `CL=F` lacks daily contract identifiers in the stored response.
Its arithmetic differences alone cannot supply this dataset. EIA's
official nearby-contract histories stop on 5 April 2024. They are useful
for source reconciliation and the April 2020 example, but cannot be
silently promoted to an August 2026 replacement.

## April 2020 check

The May contract's last trading date was 21 April 2020, so this roll rule
holds it through 20 April settlement and then rolls to June. Verify the
May loss on 20 April and the June change on 21 April separately. The May
contract's recovery on its expiration date is an observed market movement
but is not the gain of a position already rolled into June. Preserve that
distinction in the audit. This rule differs from earlier-rolling strategies,
which did not hold the negative-settlement contract on 20 April.

## Evaluation after admission

Recompute the WTI histories, all affected base forecasts and corrections
from the new target. Never reuse log-return forecasts as P&L forecasts.
Keep the existing model configurations, alpha levels, past-only contexts,
chronological split and disclosed static/rolling definitions. Save native
outputs, parameters, context hashes and replay evidence. Reassess the
separated theorem's assumptions for the new score process; the data change
does not confer a theorem on the contiguous or rolling estimators.

Verify the return builder against hand-calculated negative-price and
contract-roll cases, validate past-only contract selection, and test that
missing overlap or an incomplete endpoint fails loudly. Reconcile the
event against official sources. Produce a readiness receipt before model
inference. Canonical manuscript and aggregate replacements wait for full
data admission and forecast/result replay; do not claim this protocol alone
repairs the current paper.

## Primary sources

- CME CL specification: https://www.cmegroup.com/rulebook/NYMEX/2/200.pdf
- CFTC event report: https://www.cftc.gov/media/5296/InterimStaffReportNYMEX_WTICrudeOil/download
- EIA futures definitions: https://www.eia.gov/dnav/pet/TblDefs/pet_pri_fut_tbldef2.asp
- EIA coverage: https://www.eia.gov/dnav/pet/pet_pri_fut_s1_d.htm
- EIA first/second nearby histories: https://www.eia.gov/dnav/pet/hist/RCLC1D.htm and https://www.eia.gov/dnav/pet/hist/RCLC2D.htm
