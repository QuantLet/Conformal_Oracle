"""Check reused numerical functions, with the explicit dynamic path adaptation."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pairs = [
        ('controlled_comparisons.py', 'research/r8_review/controlled_comparisons.py',
         ['loss','weighted_quantile','qr_fit','candidates','past_volatility','gate_and_window']),
        ('analyse_panel.py', 'source/scripts/extension_20260831/analyse_panel.py', ['loss','rollshift','bootstrap']),
        ('diagnostics.py', 'source/scripts/extension_20260831/additional_diagnostics.py', ['dq']),
        ('decision.py', 'research/r8_decision/run.py', ['compute','work']),
        ('dynamic.py', 'source/scripts/extension_20260831/dynamic.py', ['work'])]
    records=[]
    for name, old_name, functions in pairs:
        current, old = HERE/name, PROJECT/old_name
        a, b = current.read_text(), old.read_text()
        adaptation = None
        if name == 'dynamic.py':
            # This exact directory literal is the sole change inside work().
            a = a.replace("out=ROOT/'artifacts/r8_commodity_etp'", "out=ROOT/'artifacts/extension_20260831'")
            adaptation = 'Only the explicit output/archive directory literal is normalised.'
        af={n.name:n for n in ast.parse(a).body if isinstance(n,ast.FunctionDef)}
        bf={n.name:n for n in ast.parse(b).body if isinstance(n,ast.FunctionDef)}
        for n in functions:
            assert ast.dump(af[n],include_attributes=False)==ast.dump(bf[n],include_attributes=False),(name,n)
        records.append(dict(adapted=str(current.relative_to(PROJECT)),original=old_name,
            adapted_sha256=sha(current),original_sha256=sha(old),functions=functions,path_adaptation=adaptation))
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))
    files += [PROJECT/'source/scripts/extension_20260831'/n for n in ['commodity_scope.py','build_paper_outputs.py','validate_r8.py']]
    files += [PROJECT/'research/r8_integration/build.py']+list((PROJECT/'source/sections_r8').glob('*.tex'))
    for p in files:
        result=subprocess.run(['git','diff','--no-index','--check','/dev/null',str(p)],capture_output=True,text=True)
        assert result.returncode in [0,1] and not result.stdout and not result.stderr,(str(p),result.stdout,result.stderr)
    report=dict(status='passed',producer_sha256=sha(Path(__file__)),records=records,
                whitespace_files=len(files),whitespace_check='git diff --no-index --check; export has no .git metadata')
    (PROJECT/'artifacts/r8_commodity_etp/quality/producer_scope_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Numerical functions and',len(files),'file whitespace checks passed')


if __name__ == '__main__': main()
