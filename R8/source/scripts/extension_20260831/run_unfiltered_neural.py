#!/usr/bin/env python3
"""Rebuild both corrected inputs with the already frozen inference producer."""
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'artifacts/extension_20260831/unfiltered_repair'
for model,device in [('moirai','cpu'),('timesfm25','mps'),('moirai2','mps'),('lagllama','mps')]:
    with (ROOT/f'inference_{model}.log').open('w') as log:
        subprocess.run([sys.executable,str(HERE/'infer_tsfm.py'),'--root',str(ROOT),'--model',model,'--device',device,
                        '--assets','ETH','NATGAS'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(model,'complete',flush=True)
