#!/bin/zsh
# Base-forecast chain for the Developed ex-US 25 universe; stops at the first failing stage.
set -e
PROJECT="/Users/danpele/Library/Mobile Documents/com~apple~CloudDocs/Documents/2026 CFP LLM VaR"
PY=/private/tmp/irfa-r8-conda-clean/bin/python
OUT="$PROJECT/artifacts/r8_external2/devexus"
W=${WORKERS:-12}
T="$OUT/stage_times.log"
stamp(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1" >> "$T"; }
cd "$PROJECT"
stamp "classical start"
$PY source/scripts/extension_20260831/classical.py --root artifacts/r8_external2/devexus --models hs gjr_t --workers $W > "$OUT/classical.log" 2>&1
stamp "classical end"
stamp "dynamic start"
(cd research/r8_external2 && $PY dynamic.py --workers $W) > "$OUT/dynamic.log" 2>&1
stamp "dynamic end"
stamp "classical_replay start"
mkdir -p "$OUT/classical_replay/data/returns"
cp "$OUT"/data/returns/*.csv "$OUT/classical_replay/data/returns/"
$PY source/scripts/extension_20260831/classical.py --root artifacts/r8_external2/devexus/classical_replay --models hs gjr_t --workers $W > "$OUT/classical_replay.log" 2>&1
stamp "classical_replay end"
stamp "dynamic_replay start"
(cd research/r8_external2 && $PY dynamic.py --replay --workers $W) > "$OUT/dynamic_replay.log" 2>&1
stamp "dynamic_replay end"
stamp "validate_base start"
(cd research/r8_external2 && $PY validate_base.py) > "$OUT/base_validation.log" 2>&1
stamp "validate_base end"
echo BASE_CHAIN_OK >> "$T"
