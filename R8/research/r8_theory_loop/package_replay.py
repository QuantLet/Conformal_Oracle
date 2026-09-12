"""Make a standalone research archive, including its original local plan commit."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/theory_loop'


def main():
    paths=set()
    for folder in (ROOT/'research/r8_theory_loop',OUT):
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name not in (
                'replay_package.zip','package_manifest.json','archive_replay.json','archive_replay.log',
                'provenance_manifest.json','final_receipt.json'):
                paths.add(p)
    paths.update(ROOT/n for n in ('analysis_plan_theory_loop.md','DECISIONS.md',
        'artifacts/r8_financial_argument/final_validation.json'))
    preserved=json.loads((OUT/'preserved_R8.json').read_text())
    paths.update(ROOT/n for n in preserved)
    # Import-only discovery of the exact existing simulation reference dependencies.
    # None of their data-loading functions is called.
    sys.path.insert(0,str(ROOT/'research/r8_review'))
    sys.path.insert(0,str(ROOT/'research/r8_shape_cost'))
    import complexity_simulation, validate_simulation
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p:
            p=Path(p).resolve()
            if p.is_relative_to(ROOT) and p.suffix=='.py':
                paths.add(p)
    members={}
    for p in sorted(paths):
        stat=p.stat();name=str(p.relative_to(ROOT))
        members[name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                           size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    record={'members':members,'scope':'Synthetic theory-loop research; financial panels not included or executed',
            'protocol_commit':json.loads((OUT/'lock.json').read_text())['protocol_commit']}
    archive=OUT/'replay_package.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in members:
            z.write(ROOT/name,name)
        z.writestr('THEORY_LOOP_PACKAGE_MANIFEST.json',json.dumps(record,indent=2)+'\n')
    (OUT/'package_manifest.json').write_text(json.dumps({**record,
          'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'members':len(members),'bytes':archive.stat().st_size,
                      'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))


if __name__=='__main__':
    main()
