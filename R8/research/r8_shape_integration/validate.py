"""Validate the integrated documents without rerunning or changing the study."""
import collections, hashlib, json, re, subprocess, tempfile, zipfile, sys
from pathlib import Path
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/r8_shape_integration'
DOC=ROOT/'docs/shape_integration_20260911'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def scope():
 before=json.loads((OUT/'before.json').read_text())
 assert sha(OUT/'before_sources.zip')==before['before_zip_sha256']
 allowed={'source/main_R8.tex','source/supplement_R8.tex','source/main_R8.pdf','source/supplement_R8.pdf','docs/IRFA_REVIEW_STATE.md','source/analysis/provenance_r8/PRODUCERS.tsv'}
 allowed.update('source/sections_r8/'+x+'.tex' for x in ['introduction','montecarlo','discussion','results','theory','supp_sensitivities'])
 formal=r'\\begin\{(theorem|proposition|corollary|lemma|assumption)\}.*?\\end\{\1\}'
 maths=r'\\begin\{(equation\*?|align\*?)\}.*?\\end\{\1\}|\\\[.*?\\\]'
 floats=r'\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}'
 extract=lambda pat,s:[m.group() for m in re.finditer(pat,s,re.S)]
 changes=[];counts=collections.Counter();oldcites=set();newcites=set()
 with zipfile.ZipFile(OUT/'before_sources.zip') as z, tempfile.TemporaryDirectory() as tmp:
  for name,h in before['files'].items():
   a=z.read(name);b=(ROOT/name).read_bytes();assert hashlib.sha256(a).hexdigest()==h
   if a!=b:
    assert name in allowed,('unexpected changed file',name);changes.append(name)
   if name.endswith('.tex'):
    a,b=a.decode(),b.decode()
    for pat in [formal,maths]:assert extract(pat,a)==extract(pat,b),(name,'old mathematics')
    for oldfloat in extract(floats,a):assert oldfloat in extract(floats,b),(name,'old float')
    for label in extract(r'\\label\{[^}]+\}',a):assert label in b,(name,label)
    counts.update(m.group(1) for m in re.finditer(formal,a,re.S))
    for s,d in [(a,oldcites),(b,newcites)]:
     for keys in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',s):d.update(keys.split(','))
   if name in changes and name.endswith(('.tex','.md','.tsv')):
    p=Path(tmp)/'before';p.write_bytes(z.read(name))
    run=subprocess.run(['git','diff','--no-index','--check',str(p),str(ROOT/name)],capture_output=True,text=True)
    assert run.returncode in [0,1] and not run.stdout and not run.stderr,(name,run.stdout,run.stderr)
  bad=Path(tmp)/'bad';bad.write_text('trailing \n');empty=Path(tmp)/'empty';empty.write_text('')
  run=subprocess.run(['git','diff','--no-index','--check',str(empty),str(bad)],capture_output=True,text=True)
  assert 'trailing whitespace' in run.stdout
 assert oldcites==newcites
 # Check old portable assets too; their bytes were outside the textual snapshot.
 previous=json.loads((ROOT/'artifacts/r8_contiguous_integration/source_package.json').read_text())
 with zipfile.ZipFile(ROOT/previous['source_zip']) as z:
  for name in z.namelist():
   if name.startswith('figures/') or name.endswith('.bib'):assert z.read(name)==(ROOT/'source'/name).read_bytes(),name
 for name in ['shape_cost','shape_proof']:
  assert sha(ROOT/'source/sections_r8'/f'{name}.tex') in (DOC/'MATH_RECHECK.md').read_text(),name
 for name in ['results','supp_sensitivities']:
  assert sha(ROOT/'source/sections_r8'/f'{name}.tex') in (DOC/'FINANCIAL_RECHECK.md').read_text(),name
 assert '\\begin{proof}' in (ROOT/'source/sections_r8/shape_proof.tex').read_text()
 return dict(changed_snapshot_files=changes,unchanged_snapshot_files=len(before['files'])-len(changes),original_formal_statements_preserved=dict(counts),old_math_figures_tables_bibliography_preserved=True,git_metadata_present=(ROOT/'.git').exists(),no_index_whitespace_check=True,whitespace_negative_control=True)

def study():
 # Historical study files stay immutable; its current prose bindings are superseded.
 receipt=json.loads((ROOT/'artifacts/r8_shape_cost/release.json').read_text())
 assert receipt['status']=='complete' and receipt['archive_only_simulation_and_financial_replay']
 assert sha(ROOT/receipt['archive'])==receipt['archive_sha256']
 with zipfile.ZipFile(ROOT/receipt['archive']) as z:
  m,=[n for n in z.namelist() if n.endswith('/STUDY_MANIFEST.json')]
  manifest=json.loads(z.read(m));prefix=m.rsplit('/',1)[0]
  oldinputs=json.loads(z.read(prefix+'/artifacts/r8_shape_cost/simulation/before.json'))['inputs']
  checked={n:h for n,h in manifest['files'].items() if n not in oldinputs}
  for n,h in checked.items():assert sha(ROOT/n)==h,('changed study',n)
 return dict(archive_sha256=receipt['archive_sha256'],immutable_study_files=len(checked),original_replay_receipt=receipt,unchanged_files=checked)

def documents():
 guard=json.loads((ROOT/'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json').read_text())
 assert guard['documents']['negative_controls']==4
 assert all(guard['documents'][k]==0 for k in ['undefined_references','undefined_citations','overfull_boxes'])
 n=guard['displays']['displays']+sum(guard[k]['exact_display_replay'] for k in ['risk_displays','regime_displays','ten_external_displays','partial_displays','information_displays']);assert n==57
 package=json.loads((OUT/'source_package.json').read_text());assert package['source_members']==73
 assert sha(ROOT/package['source_zip'])==package['source_zip_sha256']==sha(ROOT/'release/R8_LaTeX_sources.zip')
 with zipfile.ZipFile(ROOT/package['source_zip']) as z:
  for name in z.namelist():
   if name!='BUILD.txt':assert z.read(name)==(ROOT/'source'/name).read_bytes(),name
 visual=json.loads((OUT/'visual_validation.json').read_text())
 for name,h in visual['rendered_files'].items():assert sha(ROOT/name)==h,name
 for doc,info in package['documents'].items():
  path=ROOT/'source'/(doc+'.pdf');pages=PdfReader(path).pages
  assert sha(path)==info['pdf_sha256']==visual['documents'][doc]['pdf_sha256']
  assert len(pages)==info['pages'] and info['normalised_text_matches'] and info['clean_diagnostics']
  assert all(len(p.extract_text().strip())>10 for p in pages)
  log=(ROOT/'source'/(doc+'.log')).read_text(errors='backslashreplace')
  assert not re.search(r'(?:Reference|Citation).*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!',log,re.M)
 assert sha(ROOT/'Manuscript_R8.pdf')==package['documents']['main_R8']['pdf_sha256']
 return dict(existing_exact_displays=n,guards=guard['documents'],source_package=package,visual_inspection=visual)

def display_check():
 subprocess.run([sys.executable,str(ROOT/'research/r8_shape_integration/displays.py'),'--check'],check=True,capture_output=True,text=True)
 return json.loads((OUT/'display_check.json').read_text())

def main():
 result=dict(status='passed',producer_sha256=sha(__file__),scope=scope(),documents=documents(),study=study(),new_display_check=display_check(),new_simulations_or_forecasts_during_integration=False)
 names=set(json.loads((OUT/'before.json').read_text())['files'])|set(result['study']['unchanged_files'])
 for base in ['research/r8_shape_integration','docs/shape_integration_20260911']:
  names.update(str(p.relative_to(ROOT)) for p in (ROOT/base).glob('*') if p.is_file())
 names.update(['docs/IRFA_CLAIM_EVIDENCE_MAP.md','artifacts/r8_shape_cost/release.json','release/R8_20260911_shape_cost_study.zip','release/R8_20260911_contiguous_sources.zip','artifacts/r8_contiguous_integration/source_package.json','artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json','release/R8_20260911_shape_sources.zip','release/R8_LaTeX_sources.zip','Manuscript_R8.pdf'])
 with zipfile.ZipFile(ROOT/'release/R8_20260911_shape_sources.zip') as z:names.update('source/'+n for n in z.namelist() if n!='BUILD.txt')
 result['scope']['current_files']={n:sha(ROOT/n) for n in sorted(names)}
 (OUT/'final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':'passed','pages':{d:i['pages'] for d,i in result['documents']['source_package']['documents'].items()},'existing_displays':57,'new_display_check':result['new_display_check'],'unchanged_study_files':result['study']['immutable_study_files']},indent=2))

if __name__=='__main__':main()
