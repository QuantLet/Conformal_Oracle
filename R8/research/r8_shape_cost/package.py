"""Package the study's full numerical input closure and replay after extraction."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import engine as e


def main():
    root=e.ROOT
    final=json.loads((root/'artifacts/r8_shape_cost/final_validation.json').read_text())
    assert final['status']=='passed'
    finance=json.loads((root/'artifacts/r8_shape_cost/financial/receipt.json').read_text())
    before=json.loads((e.OUT/'before.json').read_text())
    names=set(finance['inputs'])|set(before['inputs'])
    for prefix in ['research/r8_shape_cost','artifacts/r8_shape_cost','docs/shape_cost_20260911']:
        names.update(str(p.relative_to(root)) for p in (root/prefix).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts
                     and p.name not in ['release.json','package.log','.DS_Store'])
    expected={name:e.sha(root/name) for name in sorted(names)}
    for name,digest in final['sources'].items():
        assert expected[name]==digest,name
    for name,digest in finance['inputs'].items():
        assert expected[name]==digest,name
    prefix='R8_20260911_shape_cost_study'
    target=root/'release'/f'{prefix}.zip'
    partial=target.with_suffix('.partial.zip')
    assert not target.exists() and not partial.exists(), 'Preserve prior releases'
    manifest=dict(study='Known-scale shape cost and retrospective financial state allocation',
                  files=expected,canonical_manuscript_updated=False,
                  independent_external_review=False)
    size=sum((root/name).stat().st_size for name in names)
    print('Packaging',len(names),'files;',size,'bytes',flush=True)
    with zipfile.ZipFile(partial,'w',allowZip64=True) as archive:
        for name in sorted(names):
            compression=zipfile.ZIP_STORED if Path(name).suffix in ['.npz','.parquet','.png','.pdf','.zip'] else zipfile.ZIP_DEFLATED
            archive.write(root/name,prefix+'/'+name,compress_type=compression)
        archive.writestr(prefix+'/STUDY_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='irfa-shape-cost-replay-') as folder:
        dest=Path(folder)
        with zipfile.ZipFile(partial) as archive:
            assert len(archive.namelist())==len(names)+1
            for name,digest in expected.items():
                p=dest/name
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(archive.read(prefix+'/'+name))
                assert e.sha(p)==digest,name
        commands=[['research/r8_shape_cost/validate_simulation.py'],
                  ['research/r8_shape_cost/financial.py','--validate'],
                  ['research/r8_shape_cost/validate.py']]
        env=os.environ.copy()
        for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
            env[key]='1'
        for args in commands:
            print('Extracted replay:', ' '.join(args),flush=True)
            run=subprocess.run([sys.executable,*args],cwd=dest,env=env,capture_output=True,text=True)
            assert run.returncode==0,(args,run.stdout,run.stderr)
        replay=json.loads((dest/'artifacts/r8_shape_cost/final_validation.json').read_text())
        assert replay==final,'Extracted study validation differs'
    partial.rename(target)
    result=dict(status='complete',archive=str(target.relative_to(root)),archive_sha256=e.sha(target),
                input_and_output_files=len(names),uncompressed_bytes=size,every_member_verified=True,
                archive_only_simulation_and_financial_replay=True,final_validation_replay_exact=True,
                simulation_histories=5000,financial_input_files=len(finance['inputs']),
                manuscript_integrated=False,public_deposit_performed=False)
    (root/'artifacts/r8_shape_cost/release.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
