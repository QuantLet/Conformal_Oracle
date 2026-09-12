#!/usr/bin/env python3
"""Rebuild all analytical returns from frozen source responses in a fresh folder."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from panel_statistics import ROOT


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='irfa-returns-replay-') as name:
        out=Path(name);(out/'raw_responses').symlink_to(ROOT/'raw_responses',target_is_directory=True)
        (out/'quality').mkdir();shutil.copy2(ROOT/'quality/primary_returns_sources.json',out/'quality/primary_returns_sources.json')
        subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_returns.py')),'--root',str(out)],check=True)
        rows=[]
        for f in sorted((ROOT/'data/returns').glob('*.csv')):
            got=out/'data/returns'/f.name;assert f.read_bytes()==got.read_bytes(),f.name
            rows.append(dict(asset=f.stem,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        assert len(rows)==24
    (ROOT/'quality/returns_replay.json').write_text(json.dumps(dict(exact=True,assets=rows),indent=2)+'\n')
    print('PASS all 24 return files reconstructed byte-for-byte from frozen source responses',flush=True)
