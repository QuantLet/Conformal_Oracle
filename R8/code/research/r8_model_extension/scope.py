"""Current input support and isolated native-model extension locations."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'artifacts/r8_model_extension'
GRID = PROJECT / 'artifacts/r8_grid_candidates'
CHRONOS = PROJECT / 'artifacts/r8_native_candidates/chronos-2-full'
FUNDS = PROJECT / 'artifacts/r8_commodity_etp'
EXT = FUNDS / 'panel/base'
sys.path.insert(0, str(PROJECT / 'research/r8_commodity_etp'))
from panel_scope import ASSETS, MODELS as REFERENCES


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def bind(paths):
    return {str(Path(p).relative_to(PROJECT)): sha(p) for p in paths}
