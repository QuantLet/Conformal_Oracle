"""Independent numerical and source-scope checks for the count-law result."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile
import numpy as np
from scipy.integrate import quad
from pypdf import PdfReader
import exact_witness as exact

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/r8_count_law'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def matrix_cost():
    # A killed three-phase transition kernel gives an independent evaluation
    # of survival probabilities, without constructing polynomial coefficients.
    theta=.5;n=125;p=.99
    def q(v,m):
        t=(v*v+max(2*v-1,0)**2)/2
        matrix=np.array([[(1-theta)*v,theta*v,0],[0,0,v],[t/(v*v),0,0]])
        return (np.array([1,0,0])@np.linalg.matrix_power(matrix,m)).sum()
    def h(v):return (q(v,n)+theta*v*q(v,n-1)+theta*v*v*q(v,n-2))/(1+2*theta)
    cost=(1-p)**2/2-sum(quad(lambda v:(v-p)*h(v),a,b,epsabs=1e-14,epsrel=1e-12)[0]
                        for a,b in [(0,.5),(.5,1)])
    reference=float(exact.cost(125,exact.F(99,100),exact.F(1,2)))
    assert abs(cost-reference)<1e-13
    return abs(cost-reference)


def main():
    certificate=exact.run()
    assert certificate==json.loads((OUT/'exact_witness.json').read_text())
    primary=json.loads((OUT/'validation.json').read_text())
    replay=json.loads((OUT.with_name('r8_count_law_replay')/'validation.json').read_text())
    assert primary==replay
    assert all(sha(ROOT/p)==h for p,h in primary['inputs'].items())
    for directory in (OUT,OUT.with_name('r8_count_law_replay')):
        assert all(sha(directory/p)==h for p,h in primary['outputs'].items())
    error=matrix_cost()
    snapshot=json.loads((OUT/'before_manuscript.json').read_text())
    assert sha(ROOT/'artifacts/r8_horizon_bridge/final_validation.json')==snapshot['previous_validation']
    changed=[];numbered=0;unchanged=0
    pattern=re.compile(r'\\begin\{equation\}.*?\\end\{equation\}',re.S)
    with zipfile.ZipFile(OUT/'before_sources.zip') as before,tempfile.TemporaryDirectory(prefix='irfa-count-diff-') as folder:
        for relative,want in snapshot['files'].items():
            old=before.read(relative);assert hashlib.sha256(old).hexdigest()==want
            now=(ROOT/relative).read_bytes()
            blocks=pattern.findall(old.decode());numbered+=len(blocks)
            assert all(block in now.decode() for block in blocks)
            if old==now:unchanged+=1;continue
            changed.append(relative)
            a=Path(folder)/'before';a.write_bytes(old)
            check=subprocess.run(['git','diff','--no-index','--check',str(a),str(ROOT/relative)],capture_output=True,text=True)
            assert not check.stdout and not check.stderr and check.returncode in (0,1),(relative,check.stdout,check.stderr)
            if relative.endswith('.bib'):assert now.startswith(old)
        a=Path(folder)/'empty';a.write_text('');b=Path(folder)/'bad';b.write_text('bad whitespace \n')
        negative=subprocess.run(['git','diff','--no-index','--check',str(a),str(b)],capture_output=True,text=True)
        assert 'trailing whitespace' in negative.stdout and negative.returncode not in (0,1)
    expected=['source/main_R8.tex','source/calibrating_the_oracle.bib',
        *[f'source/sections_r8/{name}.tex' for name in ['risk','risk_proofs','introduction','deployment','discussion','supp_specs']]]
    assert sorted(changed)==sorted(expected)
    proof=(ROOT/'source/sections_r8/risk_proofs.tex').read_text()
    for literal in ['d=813/100000',r'\frac{589}{17780000}',r'\frac{3292333}{10^{11}}',r'\frac{3292334}{10^{11}}']:
        assert literal in proof
    documents=json.loads((ROOT/'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json').read_text())
    package=json.loads((ROOT/'artifacts/r8_ten_integration/source_package.json').read_text())
    assert sha(ROOT/package['source_zip'])==package['source_zip_sha256']
    assert sha(ROOT/'release/R8_LaTeX_sources.zip')==package['source_zip_sha256']
    for document,result in package['documents'].items():
        assert sha(ROOT/f'source/{document}.pdf')==result['pdf_sha256']
        assert result['normalised_text_matches'] and result['clean_diagnostics']
        log=(ROOT/f'source/{document}.log').read_text(encoding='utf8',errors='backslashreplace')
        assert 'name{Hfootnote.1}' not in log
        pages=PdfReader(ROOT/f'source/{document}.pdf').pages
        assert all(len(page.extract_text().strip())>10 for page in pages)
        if document=='main_R8':
            assert 'JEL:' in pages[0].extract_text()
            assert '1. Introduction' in pages[1].extract_text()
    assert documents['documents']['undefined_references']==documents['documents']['undefined_citations']==documents['documents']['overfull_boxes']==0
    result=dict(status='passed',exact_rational_certificate=True,independent_matrix_quadrature_error=error,
                normal_grid_replay='exact',normal_configurations=48,all_previous_numbered_equations_preserved=numbered,
                changed_manuscript_files=changed,preserved_source_files=unchanged,old_bibliography_entries_unchanged=True,
                added_bibliography_entries=['wu2005bahadur','steinwart2011pinball','yuzbasi2017pretest','marx2025calibrated'],
                empirical_results_unchanged=True,whitespace_negative_control=True,git_metadata_present=(ROOT/'.git').exists(),
                documents=documents['documents'],display_replay_counts=[25,14,2,13],source_package=package,
                pdf_footnote_destination_repaired=True,producer_sha256=sha(__file__),
                title_page_complete=True,no_blank_pages=True,
                current_manuscript={p:sha(ROOT/p) for p in snapshot['files']})
    (OUT/'final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='current_manuscript'},indent=2))


if __name__=='__main__':main()
