#!/usr/bin/env python3
"""Sequential MPS production/replay in the existing model-specific environments."""
import argparse
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
ENV = {'moirai': '/private/tmp/irfa-august-tsfm-env/bin/python',
       'lagllama': '/private/tmp/irfa-august-tsfm-env/bin/python',
       'chronos2': '/private/tmp/irfa-native-chronos/bin/python',
       'patchtst': '/private/tmp/irfa-grid-patchtst-recreated/bin/python'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=list(ENV), default=list(ENV))
    args = parser.parse_args()
    logs = PROJECT/'artifacts/r8_commodity_etp/logs'; logs.mkdir(exist_ok=True)
    for model in args.models:
        for replay in [False, True]:
            phase = 'replay' if replay else 'production'
            cmd = [ENV[model], str(Path(__file__).with_name('native.py')), '--model', model]
            if replay: cmd += ['--replay']
            print(model, phase, 'started', flush=True)
            with (logs/f'{model}_{phase}.log').open('w') as log:
                result = subprocess.run(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            print(model, phase, 'exit', result.returncode, flush=True)
            if result.returncode: sys.exit(result.returncode)


if __name__ == '__main__': main()
