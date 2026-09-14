# Stepdown and model confidence set: results

Exploratory; protocol fixed before computation. Level 5%, two-sided, 999 stored draws.

## 20-day blocks

Single-step critical values: eight-member 2.590626, six-member 2.553778. Stepdown steps: [{"step": 1, "critical_value": 2.5906261924435245, "remaining": 8}, {"step": 2, "critical_value": 2.54133241682074, "remaining": 7}]

| method | difference | sd | t | rejected single-step | stepdown step |
|---|---|---|---|---|---|
| Raw | 0.1593 | 0.0725 | 2.198 | False | none |
| Vol-ERM | -0.0321 | 0.0247 | -1.299 | False | none |
| State-L1 | 0.0124 | 0.0220 | 0.561 | False | none |
| POT-Shift | -0.0131 | 0.0064 | -2.036 | False | none |
| POT-Vol | -0.0173 | 0.0254 | -0.684 | False | none |
| DtACI-projected-expected | 0.1376 | 0.0473 | 2.907 | True | 1 |
| Loss-gate | 0.0694 | 0.0563 | 1.233 | False | none |
| Past-minimum | 0.0170 | 0.0343 | 0.496 | False | none |

Model confidence set at 95%: ['Shift-CP', 'Vol-ERM', 'State-L1', 'POT-Shift', 'POT-Vol', 'Loss-gate', 'Past-minimum']

## 60-day blocks

Single-step critical values: eight-member 2.552901, six-member 2.524171. Stepdown steps: [{"step": 1, "critical_value": 2.552900798900319, "remaining": 8}, {"step": 2, "critical_value": 2.5183121169800287, "remaining": 7}]

| method | difference | sd | t | rejected single-step | stepdown step |
|---|---|---|---|---|---|
| Raw | 0.1593 | 0.0772 | 2.063 | False | none |
| Vol-ERM | -0.0321 | 0.0250 | -1.287 | False | none |
| State-L1 | 0.0124 | 0.0226 | 0.547 | False | none |
| POT-Shift | -0.0131 | 0.0060 | -2.162 | False | none |
| POT-Vol | -0.0173 | 0.0256 | -0.677 | False | none |
| DtACI-projected-expected | 0.1376 | 0.0505 | 2.727 | True | 1 |
| Loss-gate | 0.0694 | 0.0569 | 1.219 | False | none |
| Past-minimum | 0.0170 | 0.0349 | 0.487 | False | none |

Model confidence set at 95%: ['Shift-CP', 'Raw', 'Vol-ERM', 'State-L1', 'POT-Shift', 'POT-Vol', 'Loss-gate', 'Past-minimum']

