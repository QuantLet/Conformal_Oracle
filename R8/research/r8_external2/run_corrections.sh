#!/bin/zsh
# Correction, aggregation and validation chain; stops at the first failing stage.
set -e
PROJECT="/Users/danpele/Library/Mobile Documents/com~apple~CloudDocs/Documents/2026 CFP LLM VaR"
PY=/private/tmp/irfa-r8-conda-clean/bin/python
OUT="$PROJECT/artifacts/r8_external2/devexus"
W=${WORKERS:-12}
T="$OUT/stage_times.log"
stamp(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1" >> "$T"; }
cd "$PROJECT/research/r8_external2"
stamp "corrections start"
$PY corrections.py --workers $W > "$OUT/corrections.log" 2>&1
stamp "corrections end"
stamp "corrections_replay start"
$PY corrections.py --replay --workers $W > "$OUT/corrections_replay.log" 2>&1
stamp "corrections_replay end"
stamp "validate_corrections start"
$PY validate_corrections.py > "$OUT/correction_validation.log" 2>&1
stamp "validate_corrections end"
stamp "aggregate start"
$PY aggregate.py > "$OUT/aggregation.log" 2>&1
stamp "aggregate end"
stamp "validate_aggregate start"
$PY validate_aggregate.py > "$OUT/aggregation_validation.log" 2>&1
stamp "validate_aggregate end"
echo CORRECTION_CHAIN_OK >> "$T"
