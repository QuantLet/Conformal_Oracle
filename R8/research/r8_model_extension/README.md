# Current-panel native-model extension

Protocol: `PROTOCOL.md`. Results/status: `docs/IRFA_NEW_MODELS_CURRENT_PANEL.md`.
This directory adds isolated evaluation outputs; it does not overwrite the
completed commodity-fund release or canonical manuscript.

Use the analysis environment `/private/tmp/irfa-r8-conda-clean/bin/python`.
Run commands from the project root. Producers refuse to replace complete
receipts; use validators on an existing completed archive, or reconstruct
into a fresh copy of the project without this extension's output directory.

Completed nine-model reconstruction sequence:

```sh
python research/r8_model_extension/prepare.py
python research/r8_model_extension/verify_sources.py
python research/r8_model_extension/evaluate.py
python research/r8_model_extension/validate.py
python research/r8_model_extension/policy.py
python research/r8_model_extension/validate_policy.py
python research/r8_model_extension/plot.py
```

The prepare step depends on the full Chronos/PatchTST original forecast and
replay receipts, the fund replacement's three-asset forecast/replay receipts,
and the staged current core panel. It verifies matching input hashes before
reusing any forecast. No historical WTI/GOLD/NATGAS rows enter this evaluation.

Full TS-ICL production and fresh replay:

```sh
python research/r8_model_extension/run_tsicl.py
python research/r8_model_extension/finish_ten.py
```

The first command uses the pinned `irfa-grid-tsicl` and separately recreated
`irfa-grid-tsicl-recreated` Python environments. The absolute paths are local
launch conveniences; `tsicl.py --asset ASSET [--replay]` can be run directly
from reconstructed environments. Package versions, checkpoint/source hashes
and settings are recorded in every asset receipt. The six workers process
independent assets; native batches and chunks remain fixed. The second
command waits for all exact replays, assembles native archives, and executes
the ten-model evaluation and its independent policy validator. It writes
status to `artifacts/r8_model_extension/ten_pipeline.json`.

These stages do not change paper prose or automatically declare submission
readiness. Additional stronger-comparator reconstruction and manuscript
integration remain separate work. Original archives remain required for
reconstruction; this directory is not a standalone replication package.
