#!/usr/bin/env python3
"""Re-estimate the new classical fits from scratch, then compare every output."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'artifacts/r8_commodity_etp'


def main():
    with tempfile.TemporaryDirectory(prefix='irfa-commodity-classical-') as name:
        fresh = Path(name); (fresh/'data/returns').mkdir(parents=True)
        for path in (ROOT/'data/returns').glob('*.csv'):
            shutil.copy2(path, fresh/'data/returns'/path.name)
        with (ROOT/'logs/classical_fresh_replay.log').open('w') as log:
            subprocess.run([sys.executable, str(PROJECT/'source/scripts/extension_20260831/classical.py'),
                '--root', str(fresh), '--workers', '3'], stdout=log, stderr=subprocess.STDOUT, check=True)
        count = 0
        for original in sorted((ROOT/'data/benchmarks').glob('*.parquet')):
            relative = original.relative_to(ROOT)
            pd.testing.assert_frame_equal(pd.read_parquet(original), pd.read_parquet(fresh/relative), check_exact=True)
            count += 1
        fits = 0
        for model in ['hs','ewma','garch_n','gjr_garch','gjr_t']:
            for original in sorted((ROOT/'parameters'/model).glob('*.parquet')):
                pd.testing.assert_frame_equal(pd.read_parquet(original),
                    pd.read_parquet(fresh/original.relative_to(ROOT)), check_exact=True)
                fits += 1
        assert count == fits == 15
        report = dict(status='passed', forecast_files_exact=count, parameter_files_exact=fits,
            fresh_estimation=True, producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (ROOT/'quality/classical_fresh_replay.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
