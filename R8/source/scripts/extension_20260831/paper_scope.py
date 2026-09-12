"""Current manuscript panel; the original nine-model archive stays reproducible."""
from pathlib import Path
from panel_statistics import ROOT as ARCHIVE, MODELS as ARCHIVE_MODELS

PROJECT = Path(__file__).resolve().parents[3]
EXCLUDED = ('TimesFM-2.5', 'Moirai-2.0')
MODELS = {name: spec for name, spec in ARCHIVE_MODELS.items() if name not in EXCLUDED}
N_ASSETS = 24
N_PAIRS = N_ASSETS * len(MODELS)
ROOT = PROJECT / 'artifacts/r8_native_panel/base'
DECISION = PROJECT / 'artifacts/r8_native_panel/decision'

# Selection is by predictive interface, never by coverage or test loss.
# Parametric predictive laws (GARCH-t, Lag-Llama) and tail-model comparators
# remain eligible. Excluded: fitting an extra law to a central quantile grid
# solely to supply a missing 1% base forecast.
