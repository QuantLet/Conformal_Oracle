"""Restore protected historical files only in a new disposable replay copy."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('destination', type=Path)
    a = ap.parse_args()
    dest = a.destination.resolve()
    if dest.exists():
        raise FileExistsError('Use a new replay destination')
    dest.mkdir(parents=True)
    for rel in ('research/r8_regime',):
        shutil.copytree(ROOT/rel, dest/rel, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    rel = 'research/r8_decision/methods.py'
    (dest/rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/rel, dest/rel)
    binding = json.loads((ROOT/'artifacts/r8_regime/binding.json').read_text())
    for rel in binding['protected']:
        p = dest/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'artifacts/r8_regime/before'/rel, p)
    art = dest/'artifacts/r8_regime'
    art.mkdir(parents=True)
    for rel in ('binding.json', 'innovations_normal.npy', 'innovations_t5.npy'):
        shutil.copy2(ROOT/'artifacts/r8_regime'/rel, art/rel)
    shutil.copytree(ROOT/'artifacts/r8_regime/blocks', art/'blocks')
    print(f'Ready: {dest}. Run research/r8_regime/run.py --replay, then validate.py.')


if __name__ == '__main__':
    main()
