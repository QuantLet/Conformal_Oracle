"""Current portable LaTeX closure with independently compiled PDF comparison."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zipfile import ZipFile,ZIP_DEFLATED
from pypdf import PdfReader
import hashlib
PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/"source"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT=PROJECT/"artifacts/r8_optimism_integration"


def main():
    release=PROJECT/'release';target=release/'R8_20260911_optimism_sources.zip'
    with ZipFile(release/'R8_20260910_commodity_sources.zip') as old:names=set(old.namelist())-{'BUILD.txt'}
    names.update(['sections_r8/external.tex','sections_r8/numbers_ten_external.tex','sections_r8/tab_native_forecasters.tex',
                  'sections_r8/tab_ten_strong.tex','sections_r8/tab_external.tex'])
    names.update('figures/'+n+'.pdf' for n in ['fig_ten_traffic','fig_ten_strong','fig_external'])
    names.update(['sections_r8/numbers_partial.tex','sections_r8/tab_partial.tex','sections_r8/supp_partial.tex'])
    names.update(['sections_r8/numbers_information.tex','sections_r8/information.tex','sections_r8/information_proof.tex'])
    names.update(['sections_r8/contiguous_coverage.tex','sections_r8/contiguous_proof.tex'])
    names.update(['sections_r8/shape_cost.tex','sections_r8/shape_proof.tex','sections_r8/shape_design.tex','sections_r8/numbers_shape.tex','figures/fig_shape_cost.pdf'])
    names.update(['sections_r8/numbers_optimism.tex','sections_r8/tab_optimism.tex','sections_r8/optimism_proof.tex','sections_r8/supp_optimism.tex'])
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for name in sorted(names):z.write(SOURCE/name,name)
        z.writestr('BUILD.txt', '''R8 document sources and generated displays: TeX-only compilation.
Validated toolchain: TeX Live 2026, pdfTeX 1.40.29, latexmk 4.88.
The elsarticle class is bundled. From the extracted directory run:

latexmk -g -pdf -interaction=nonstopmode -halt-on-error main_R8.tex
latexmk -g -pdf -interaction=nonstopmode -halt-on-error supplement_R8.tex
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main_R8.tex
latexmk -g -pdf -interaction=nonstopmode -halt-on-error supplement_R8.tex

Repeated builds resolve references between the two documents. The portable
build is checked for equal normalized PDF text, not cross-platform PDF bytes.
Main market sample ends in August 2026; external industries in July 2026.
This ZIP excludes numerical data and Python validators. For stored-result
validation use the full current project and research/r8_optimism_integration/README.md,
which identifies the captured Python environment and required receipts.
''')
    env=dict(os.environ);env['PATH']='/Library/TeX/texbin:'+env.get('PATH','')
    record=dict(producer_sha256=sha(Path(__file__)), source_zip=str(target.relative_to(PROJECT)),source_zip_sha256=sha(target),source_members=len(names)+1,documents={})
    with tempfile.TemporaryDirectory(prefix='irfa-ten-portable-') as folder:
        work=Path(folder)
        with ZipFile(target) as z:z.extractall(work)
        for i,doc in enumerate(['main_R8','supplement_R8']*2):
            with (OUT/f'portable_{i}_{doc}.log').open('w') as log:
                subprocess.run(['latexmk','-g','-pdf','-interaction=nonstopmode','-halt-on-error',doc+'.tex'],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for doc in ['main_R8','supplement_R8']:
            a=PdfReader(SOURCE/(doc+'.pdf'));b=PdfReader(work/(doc+'.pdf'))
            norm=lambda r:re.sub(r'\s+',' ','\n'.join(p.extract_text() or '' for p in r.pages)).strip()
            assert norm(a)==norm(b)
            log=(work/(doc+'.log')).read_text(encoding='utf-8',errors='backslashreplace');assert not re.search(r'(?:Reference|Citation).*undefined|There were undefined|Overfull \\[hv]box|multiply defined',log)
            record['documents'][doc]=dict(pages=len(a.pages),normalised_text_matches=True,clean_diagnostics=True,pdf_sha256=sha(SOURCE/(doc+'.pdf')))
    shutil.copy2(SOURCE/'main_R8.pdf',PROJECT/'Manuscript_R8.pdf')
    shutil.copy2(target,release/'R8_LaTeX_sources.zip')
    (OUT/'source_package.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


if __name__=='__main__':main()
