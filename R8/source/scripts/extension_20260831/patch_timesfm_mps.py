#!/usr/bin/env python3
"""Reorder an existing float32 cast before MPS transfer; no numeric change.

TimesFM 3.0.1 pads incomplete batches with float64 zeros, then transfers the
batch before casting it to float32. MPS rejects that transfer. Full batches
are already float32. This patch retains the same final tensors everywhere.
Run only in the isolated inference environment, before launching TimesFM.
"""
import hashlib
import json
from pathlib import Path
import timesfm

root=Path(__file__).resolve().parents[3]/'artifacts/extension_20260831'
path=Path(timesfm.__file__).parent/'timesfm_2p5/timesfm_2p5_torch.py'
before=path.read_text()
old='torch.from_numpy(np.array(inputs)).to(self.model.device).to(torch.float32)'
new='torch.from_numpy(np.array(inputs)).to(torch.float32).to(self.model.device)'
record=root/'quality/timesfm_mps_patch.json'
if old in before:
    assert before.count(old)==1
    after=before.replace(old,new)
    record.write_text(json.dumps(dict(package='timesfm==3.0.1',file='timesfm_2p5/timesfm_2p5_torch.py',
                                      before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                                      after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                                      old=old,new=new),indent=2)+'\n')
    path.write_text(after)
else:
    assert new in before and record.exists()
    assert hashlib.sha256(before.encode()).hexdigest()==json.loads(record.read_text())['after_sha256']
print('TimesFM cast-order compatibility patch verified')
