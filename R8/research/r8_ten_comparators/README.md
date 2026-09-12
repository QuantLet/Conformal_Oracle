# Twenty corrections on ten native-tail models

The fixed extension contains 240 pairs at 1%, with common 512-observation
warm-up and calibration/test boundaries. `PROTOCOL.md` identifies the
unchanged earlier numerical functions. No new native inference is needed.

```sh
python research/r8_ten_comparators/run.py --workers 4
python research/r8_ten_comparators/run.py --replay --workers 4
python research/r8_ten_comparators/validate.py
python research/r8_ten_comparators/aggregate.py
python research/r8_ten_comparators/validate_aggregate.py
python -m pytest -q research/r8_decision/test_methods.py
```

Use the existing locked analysis environment. Production and replay
folders remain separate; completed fits are hash-verified, not rerun.
To force a new replay, use a disposable copy and remove only its replay
outputs. The source manuscript hashes in `before.json` protect the
pre-integration state; after manuscript integration, use a disposable
copy with that historical state to rerun the research validator.

Every score, backtest, daily path, expert mixture, fit, selection and seed
is saved. The independent aggregate check reconstructs all calendar
draws and weighting from pair losses. Details and limitations are in
`docs/IRFA_TEN_STRONG_COMPARATORS.md`. The broad GAMLSS/boosting table and
four-tail-level studies have not been transferred to ten models here.
