"""Bind independent checks, retained inputs and the delivered research artifacts."""
from importlib.metadata import version
import json
import platform
from pathlib import Path
import subprocess
from PIL import Image
import engine as e


def main():
    root=e.ROOT
    environment=json.loads((Path(__file__).with_name('environment.json')).read_text())
    assert platform.python_version()==environment['python']
    for name,expected in environment['packages'].items():
        assert version(name)==expected,name
    lock=json.loads((e.OUT/'lock.json').read_text())
    for name,digest in lock['files'].items():
        assert e.sha(root/name)==digest,name
    before=json.loads((e.OUT/'before.json').read_text())
    for name,digest in before['inputs'].items():
        assert e.sha(root/name)==digest,('Canonical input changed',name)
    execution=json.loads((e.OUT/'execution.json').read_text())
    findings=json.loads((e.OUT/'findings.json').read_text())
    assert execution['status']==findings['status']=='complete'
    for receipt in [execution,findings]:
        for name,digest in receipt['outputs'].items():
            assert e.sha(e.OUT/name)==digest,name
    simulation=json.loads((root/'artifacts/r8_shape_cost/validation/simulation_validation.json').read_text())
    assert simulation['status']=='passed'
    assert simulation['validator_sha256']==e.sha(Path(__file__).with_name('validate_simulation.py'))
    replay=simulation['output_replay']
    assert replay['execution_receipt_sha256']==e.sha(e.OUT/'execution.json')
    assert replay['findings_receipt_sha256']==e.sha(e.OUT/'findings.json')
    assert replay['protocol_lock_sha256']==e.sha(e.OUT/'lock.json')
    assert replay['saved_histories_reproduced']==5000
    assert replay['stored_rows']==360000
    assert replay['primary_sign_crossing']==findings['primary_sign_crossing']=='supported'
    finance=root/'artifacts/r8_shape_cost/financial'
    receipt=json.loads((finance/'receipt.json').read_text())
    checked=json.loads((finance/'validation.json').read_text())
    assert checked['status']=='passed'
    assert checked['producer_sha256']==receipt['producer_sha256']==e.sha(Path(__file__).with_name('financial.py'))
    assert checked['receipt_sha256']==e.sha(finance/'receipt.json')
    for name,digest in receipt['inputs'].items():
        assert e.sha(root/name)==digest,name
    for name,digest in checked['outputs'].items():
        assert e.sha(finance/name)==digest,name
    assert checked['report_sha256']==e.sha(root/'docs/shape_cost_20260911/FINANCIAL_RESULTS.md')
    states=json.loads((finance/'inference_status.json').read_text())
    assert len(states)==2 and all(s['inference']=='aborted_empty_state' for s in states)
    assert all(s['no_pairs_dropped'] and s['no_draws_redrawn'] for s in states)
    figure=root/'artifacts/r8_shape_cost/figures'
    plot_receipt=json.loads((figure/'receipt.json').read_text())
    assert plot_receipt['producer_sha256']==e.sha(Path(__file__).with_name('plot.py'))
    assert plot_receipt['input_sha256']==e.sha(e.OUT/'contrasts.csv')
    for name,digest in plot_receipt['files'].items():
        assert e.sha(root/name)==digest,name
    visual=json.loads((figure/'visual_validation.json').read_text())
    assert visual['status']=='passed' and visual['legend_outside_bottom']
    assert visual['png_sha256']==e.sha(figure/'shape_cost.png')
    with Image.open(figure/'shape_cost.png') as im:
        assert im.mode=='RGBA' and im.getpixel((0,0))[3]==0
    docs=root/'docs/shape_cost_20260911'
    sources={}
    for folder in [Path(__file__).parent,docs]:
        for p in sorted(folder.iterdir()):
            if p.is_file():
                sources[str(p.relative_to(root))]=e.sha(p)
                if p.suffix in {'.py','.md','.tex','.txt'}:
                    check=subprocess.run(['git','diff','--no-index','--check','/dev/null',str(p)],
                        capture_output=True,text=True)
                    assert check.returncode in [0,1] and not check.stdout and not check.stderr,(p,check.stdout)
    result=dict(status='passed',sources=sources,environment=environment,
        canonical_input_files_unchanged=len(before['inputs']),financial_input_files_unchanged=len(receipt['inputs']),
        simulation_validation_sha256=e.sha(root/'artifacts/r8_shape_cost/validation/simulation_validation.json'),
        financial_validation_sha256=e.sha(finance/'validation.json'),
        primary_sign_crossing=findings['primary_sign_crossing'],
        financial_inference=[dict(block=s['block_calendar_days'],status=s['inference'],
            empty_draws=s['draws_with_empty_cells']) for s in states],
        quantitative_equivalence_claimed=False,financial_superiority_established=False,
        visual_inspection=visual,canonical_manuscript_or_pdf_edits=False,
        document_build_required=False,reason='Canonical sources and PDFs are unchanged; standalone research artifacts only.')
    (root/'artifacts/r8_shape_cost/final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','environment','visual_inspection']},indent=2))


if __name__=='__main__':
    main()
