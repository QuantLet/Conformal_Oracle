#!/usr/bin/env python3
"""Rebuild R8 tables and figures from the public intermediate-result package.

Run `python R8/reproduce.py replay --workdir /tmp/r8-replay` in the explicit
environment documented in R8/README.md. The working directory must be new. No forecast
model is fitted, no simulation is run, and published results are never edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes the package: {relative}')
    return path


def verify_files(manifest, root=ROOT):
    for name, wanted in manifest['files'].items():
        path = contained(root, name)
        if not path.is_file() or digest(path) != wanted:
            raise ValueError(f'Missing or changed input: {name}')
    for name, entry in manifest['expected_outputs'].items():
        path = contained(root, entry['reference'])
        if not path.is_file() or digest(path) != entry['sha256']:
            raise ValueError(f'Missing or changed reference output: {name}')


def verify_outputs(manifest, workspace):
    for name, entry in manifest['expected_outputs'].items():
        path = contained(workspace, name)
        if not path.is_file() or digest(path) != entry['sha256']:
            actual = digest(path) if path.is_file() else 'NOT_WRITTEN'
            raise ValueError(f'Regenerated output differs: {name} ({actual})')


def run(command, workspace, env, label):
    log = workspace / 'replay_logs' / (label + '.log')
    started = time.monotonic()
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=workspace, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-14:])
        raise RuntimeError(f'{label} failed (exit {result.returncode}); see {log}\n{tail}')
    print(f'PASS {label}', flush=True)
    return dict(command=command, exit_code=0, seconds=round(time.monotonic()-started, 3),
                log=str(log.relative_to(workspace)))


def replay(manifest, workspace):
    workspace = workspace.resolve()
    if workspace == ROOT or workspace.is_relative_to(ROOT) or ROOT.is_relative_to(workspace):
        raise ValueError('Replay must use a new directory outside the published R8 tree')
    workspace.mkdir(parents=True, exist_ok=False)
    for name in manifest['files']:
        target = contained(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contained(ROOT, name), target)
    # Expected tables/figures are deliberately not copied into the workspace.
    for name in manifest['expected_outputs']:
        if contained(workspace, name).exists():
            raise ValueError(f'Output already present before execution: {name}')
    (workspace / 'source/sections_r8').mkdir(parents=True, exist_ok=True)
    (workspace / 'source/figures').mkdir(parents=True, exist_ok=True)
    (workspace / 'replay_logs').mkdir()
    env = os.environ.copy()
    for key in ['PYTHONPATH', 'PYTHONHOME']:
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR=str(workspace/'cache/matplotlib'),
               XDG_CACHE_HOME=str(workspace/'cache'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment_code = (
        "import json,sys,platform,importlib.metadata as m;import matplotlib.ft2font as ft;"
        "assert ft.__freetype_version__=='2.13.3',"
        "'Exact figures require FreeType 2.13.3; use environment-osx-arm64.explicit.txt and requirements-conda-overlay.txt';"
        "print(json.dumps(dict(python=sys.version,platform=platform.platform(),"
        "freetype=ft.__freetype_version__,packages={p:m.version(p) for p in "
        "['numpy','pandas','scipy','matplotlib','pyarrow','pypdf','Pillow','pytest']})))"
    )
    environment_run = run([sys.executable, '-c', environment_code], workspace, env, 'environment')
    environment = json.loads((workspace/'replay_logs/environment.log').read_text().splitlines()[-1])
    steps = []
    for i, script in enumerate(manifest['steps']):
        steps.append(run([sys.executable, str(workspace/script)], workspace, env, f'producer_{i+1}'))
    verify_outputs(manifest, workspace)
    print(f'PASS regenerated {len(manifest["expected_outputs"])} outputs exactly', flush=True)

    # Check that missing execution and changed outputs cannot be reported as passes.
    name = next(iter(manifest['expected_outputs']))
    target = workspace / name
    content = target.read_bytes()
    target.unlink()
    try:
        verify_outputs(manifest, workspace)
    except ValueError:
        missing_rejected = True
    else:
        raise AssertionError('Missing-output negative control did not fail')
    target.write_bytes(content + b'\ncorruption\n')
    try:
        verify_outputs(manifest, workspace)
    except ValueError:
        corruption_rejected = True
    else:
        raise AssertionError('Output-corruption negative control did not fail')
    finally:
        target.write_bytes(content)
    changed = json.loads(json.dumps(manifest))
    changed['files'][next(iter(changed['files']))] = '0' * 64
    try:
        verify_files(changed)
    except ValueError:
        input_change_rejected = True
    else:
        raise AssertionError('Input-corruption negative control did not fail')

    tests = run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                 'source/scripts/extension_20260831/test_statistics.py'], workspace, env, 'statistics_tests')
    exact = run([sys.executable, 'research/r8_count_law/exact_witness.py'], workspace, env, 'exact_witness')
    check = (
        "import sys,json;sys.path.insert(0,'research/r8_count_law');"
        "import validate;error=validate.matrix_cost();"
        "sys.path.insert(0,'research/r8_information_limit');import engine;"
        "values=[engine.error(250,.01,.2,r) for r in [0.,.5]];"
        "assert abs(values[0]-.3550353060461644)<1e-13;"
        "assert abs(values[1]-.39493849863088804)<1e-13;"
        "print(json.dumps(dict(matrix_error=error,selection_error=values)))"
    )
    bounded = run([sys.executable, '-c', check], workspace, env, 'bounded_numerical_checks')
    verify_outputs(manifest, workspace)
    verify_files(manifest)
    return dict(status='passed', scope=manifest['scope'],
                input_manifest_sha256=digest(ROOT/'REPLAY_MANIFEST.json'),
                python=sys.version, interpreter=sys.executable, workspace=str(workspace),
                environment=environment, environment_check=environment_run,
                copied_input_files=len(manifest['files']),
                expected_outputs_not_prepopulated=True,
                exact_regenerated_outputs=len(manifest['expected_outputs']),
                output_sha256={name:digest(workspace/name) for name in manifest['expected_outputs']},
                negative_controls=dict(missing_output=missing_rejected, changed_output=corruption_rejected,
                                       changed_input=input_change_rejected),
                producer_runs=steps, statistics_tests=tests, exact_witness=exact,
                bounded_numerical_checks=bounded,
                no_model_fitting=True, no_research_simulation=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['verify', 'replay'], nargs='?', default='verify')
    parser.add_argument('--workdir', type=Path, help='New directory outside this package; retained after execution')
    parser.add_argument('--report', type=Path, help='Write the measured JSON receipt to this path')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'REPLAY_MANIFEST.json').read_text())
    verify_files(manifest)
    print(f'PASS integrity of {len(manifest["files"])} files and {len(manifest["expected_outputs"])} references', flush=True)
    if args.mode == 'verify':
        result = dict(status='passed', scope='Package integrity only; producers were not executed',
                      files=len(manifest['files']), references=len(manifest['expected_outputs']))
    elif args.workdir:
        result = replay(manifest, args.workdir)
    else:
        with tempfile.TemporaryDirectory(prefix='r8-public-replay-') as folder:
            result = replay(manifest, Path(folder)/'work')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ['status', 'scope', 'exact_regenerated_outputs'] if k in result}, indent=2))


if __name__ == '__main__':
    main()
