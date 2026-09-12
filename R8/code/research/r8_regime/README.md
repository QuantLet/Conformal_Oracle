# Regime-change study

Run from the repository root in the existing R8 Conda analysis environment
(the recorded build uses `/private/tmp/irfa-r8-conda-clean/bin/python`):

```sh
python -m pytest -q research/r8_regime/test_engine.py
python research/r8_regime/run.py --init
python research/r8_regime/run.py --workers 3
python research/r8_regime/run.py --replay --workers 3
python research/r8_regime/aggregate.py
python research/r8_regime/validate.py
python research/r8_regime/plot.py
```

The protocol precedes production; binding.json contains producer, protocol,
input and protected-manuscript hashes, package versions and creation time.
Each of ten blocks contains 100 independent histories of one innovation law,
reused across eight scenarios and two levels. Daily paths, expert forecasts,
pre-outcome weights, selected windows, gates and per-history metrics are
retained. Aggregation never treats different methods, dates or scenarios as
independent histories. Every paired comparison uses identical simulations.

Completed blocks are verified and skipped. Use `--single normal 0` (or t5,
and starts 0/100/200/300/400) for one block. `--replay` regenerates innovations
and recomputes all policies in fresh processes, writing to a separate replay
directory and comparing each numerical array exactly. For a new independent
replay after one is already complete, move only the replay directory aside;
do not remove production blocks, alter bindings or relabel a cache check as
new computation. Source changes require a separate documented run.

Canonical manuscript files and existing empirical results remain protected
during this research phase. If the validated study is subsequently adopted
in the manuscript, preserve those historical guards and replay in a copy
with the recorded pre-integration files, as for the earlier R8 phases.

```sh
python research/r8_regime/stage_replay.py /tmp/r8-regime-replay
cd /tmp/r8-regime-replay
python research/r8_regime/run.py --replay --workers 3
python research/r8_regime/validate.py
python research/r8_regime/aggregate.py
```

The staging command requires a new directory. It copies original production
outputs for comparison, restores the historical files only in that copy,
and does not copy a completed replay cache. The current manuscript stays intact.

The initial independent scalar EWMA test used strict bitwise equality despite
a different floating-point multiplication order. Before production, its
tolerance was set to 2e-15 relative error (zero absolute tolerance). The
scientific producer and design were unchanged. This is a test correction,
not a changed numerical result or a relaxed forecast-timing check.

The first aggregation invocation resolved the neighbouring decision-study
`run.py` because of its import search path. Before inspecting aggregate
results, aggregation/validation/plotting were changed to load their local
runner by its explicit path. No simulated path, forecast, fitting rule or
scientific producer changed; the entire study still receives fresh replay.
