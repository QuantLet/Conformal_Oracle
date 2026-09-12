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
from build import PROJECT,SOURCE,OUT,sha,check


def main():
    check();release=PROJECT/'release';target=release/'R8_20260910_ten_external_sources.zip'
    with ZipFile(release/'R8_20260910_commodity_sources.zip') as old:names=set(old.namelist())-{'BUILD.txt'}
    names.update(['sections_r8/external.tex','sections_r8/numbers_ten_external.tex','sections_r8/tab_native_forecasters.tex',
                  'sections_r8/tab_ten_strong.tex','sections_r8/tab_external.tex'])
    names.update('figures/'+n+'.pdf' for n in ['fig_ten_traffic','fig_ten_strong','fig_external'])
    names.update(['sections_r8/numbers_partial.tex','sections_r8/tab_partial.tex','sections_r8/supp_partial.tex'])
    names.update(['sections_r8/numbers_information.tex','sections_r8/information.tex','sections_r8/information_proof.tex'])
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for name in sorted(names):z.write(SOURCE/name,name)
        z.writestr('BUILD.txt','TeX Live: run latexmk -g -pdf main_R8.tex, then supplement_R8.tex, then both again. Main market sample through August; external industries through July 2026. This package contains document sources and displays, not the full numerical archive.\n')
    env=dict(os.environ);env['PATH']='/Library/TeX/texbin:'+env.get('PATH','')
    record=dict(source_zip=str(target.relative_to(PROJECT)),source_zip_sha256=sha(target),source_members=len(names)+1,documents={})
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
