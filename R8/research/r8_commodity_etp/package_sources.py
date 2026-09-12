"""Package current portable LaTeX sources and verify an isolated clean build."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile
from pypdf import PdfReader

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / 'artifacts/r8_commodity_etp/logs'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    release = PROJECT / 'release'
    previous = release / 'R8_20260910_native_sources.zip'
    target = release / 'R8_20260910_commodity_sources.zip'
    with zipfile.ZipFile(previous) as old, zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as new:
        for name in old.namelist():
            if name == 'sections_r8/tab_closures.tex':
                continue
            if name == 'BUILD.txt':
                new.writestr(name, old.read(name))
            else:
                new.write(PROJECT / 'source' / name, name)
    receipt = {'source_zip_sha256': sha(target), 'documents': {}}
    env = dict(os.environ)
    env['PATH'] = '/Library/TeX/texbin:' + env.get('PATH', '')
    with tempfile.TemporaryDirectory(prefix='irfa-native-sources-') as folder:
        work = Path(folder)
        with zipfile.ZipFile(target) as archive:
            archive.extractall(work)
            receipt['source_members_verified'] = len(archive.namelist())
            for name in archive.namelist():
                if name != 'BUILD.txt':
                    assert archive.read(name) == (PROJECT / 'source' / name).read_bytes()
        for i, doc in enumerate(['main_R8', 'supplement_R8'] * 2):
            with (OUT / f'clean_source_{i}_{doc}.log').open('w') as log:
                subprocess.run(['latexmk', '-g', '-pdf', '-interaction=nonstopmode',
                                '-halt-on-error', doc + '.tex'], cwd=work, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        for doc in ['main_R8', 'supplement_R8']:
            current = PROJECT / 'source' / (doc + '.pdf')
            built = work / (doc + '.pdf')
            texts = ['\n'.join(page.extract_text() or '' for page in PdfReader(p).pages) for p in [current, built]]
            assert re.sub(r'\s+', ' ', texts[0]).strip() == re.sub(r'\s+', ' ', texts[1]).strip(), doc
            log = (work / (doc + '.log')).read_text(errors='replace')
            assert not re.search(r'(?:Reference|Citation).*undefined|There were undefined|Overfull \\[hv]box|multiply defined', log)
            info = subprocess.check_output(['pdfinfo', str(built)], text=True)
            receipt['documents'][doc] = {'pages': int(re.search(r'Pages:\s+(\d+)', info).group(1)),
                'normalised_text_matches': True, 'clean_diagnostics': True, 'pdf_sha256': sha(current)}
    shutil.copy2(PROJECT / 'source/main_R8.pdf', PROJECT / 'Manuscript_R8.pdf')
    receipt['root_pdf_matches'] = sha(PROJECT / 'Manuscript_R8.pdf') == sha(PROJECT / 'source/main_R8.pdf')
    shutil.copy2(target, release / 'R8_LaTeX_sources.zip')
    (release / 'R8_commodity_source_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
