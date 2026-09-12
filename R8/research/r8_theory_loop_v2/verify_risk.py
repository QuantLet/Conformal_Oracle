"""Independent deterministic checks for the locked Gaussian-limit experiment.

This module performs no computation on import. Its execution requires the
protocol lock and the completed producer outputs.
No simulation or financial-data loader belongs here.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import tempfile
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from common import validate_lock


REFERENCE_ATOL = 1e-12
REFERENCE_RTOL = 1e-12
AGREEMENT_ATOL = 1e-10


def phi(z: float) -> float:
    return math.exp(-z*z/2) / math.sqrt(2*math.pi)


def action_reference(x: float) -> float:
    return 0.0 if abs(x) <= 1 else x - 1/x


def integrate_full_line(fn: Callable[[float], float], a: float) -> tuple[float, float]:
    """Gaussian-weighted integration with branch and mass-region breakpoints.

    The primary producer integrates squared error on finite [-12,12]. Here the
    entire real line is retained, and explicit central breakpoints prevent a
    distant branch at a=128 from hiding the normal mass from adaptive quad.
    """
    cuts = [-math.inf, *sorted(set([-a-1, -a+1, -12.0, 0.0, 12.0])), math.inf]
    value = error = 0.0
    for left, right in zip(cuts[:-1], cuts[1:]):
        v, e = quad(lambda z: fn(z)*phi(z), left, right,
                    epsabs=REFERENCE_ATOL, epsrel=REFERENCE_RTOL, limit=250)
        value += v
        error += e
    return value, error


def stein_risk(a: float) -> tuple[float, float]:
    """Independent risk identity: 1 + E[u(X)^2+2u'(X)], X=a+Z.

    u(x)=-x inside [-1,1], -1/x outside; continuity at both endpoints removes
    jump terms. This integrand differs algebraically from direct squared loss.
    """
    def integrand(z: float) -> float:
        x = a+z
        return x*x-2 if abs(x) <= 1 else 3/(x*x)
    value, error = integrate_full_line(integrand, a)
    return 1+value, error


def direct_risk(a: float, action: Callable[[float], float]) -> float:
    return integrate_full_line(lambda z: (action(a+z)-a)**2, a)[0]


def active_probability(a: float) -> float:
    return float(ndtr(a-1) + ndtr(-a-1))


def benchmarks(a: float) -> dict[str, float]:
    return dict(raw=a*a, full=1.0, half=(a*a+1)/4,
                fixed_oracle=a*a/(1+a*a))


def fixed_formula_mutant(a: float) -> float:
    """Incorrect: average fixed-lambda risk after estimating lambda from X."""
    def integrand(z: float) -> float:
        x = a+z
        lam = 0.0 if abs(x) <= 1 else 1-1/(x*x)
        return lam*lam + (1-lam)**2*a*a
    return integrate_full_line(integrand, a)[0]


def reversed_coefficient_mutant(x: float) -> float:
    lam = 0.0 if abs(x) <= 1 else 1-1/(x*x)
    return (1-lam)*x


def missing_positive_part_mutant(x: float) -> float:
    return 0.0 if x == 0 else x-1/x


def uniform_exact(n: int) -> dict[str, Fraction | int]:
    """Two independent exact order-statistic calculations, no quadrature."""
    p = Fraction(99,100)
    alpha = 1-p
    v = p*alpha
    k = (99*(n+1)+99)//100
    delta = Fraction(k)-n*p
    training_delta = (delta*delta-delta-n*v)/(2*n*(n+1))
    gap_training_loss = (alpha*k*(k-1) + p*(n-k)*(n-k+1))/(2*n*(n+1))
    if training_delta != gap_training_loss-v/2:
        raise AssertionError('Independent exact training identities disagree')
    mean = Fraction(k,n+1)-p
    variance = Fraction(k*(n+1-k),(n+1)**2*(n+2))
    test_delta = (Fraction(k*(k+1),(n+1)*(n+2))
                  - 2*p*Fraction(k,n+1) + p*p)/2
    if test_delta != (mean*mean+variance)/2:
        raise AssertionError('Independent exact test identities disagree')
    leading = v/(2*n)
    return dict(n=n, k=k, delta=delta, training_delta=training_delta,
                test_delta=test_delta, leading_A=leading,
                one_A_prediction=training_delta+leading,
                two_A_prediction=training_delta+2*leading,
                mean_C=mean, variance_C=variance)


def scan_crossings(reference: dict[float, float]) -> list[dict[str, float | str]]:
    """Bracketed findings on [0,8]; does not establish global uniqueness."""
    grid = sorted(a for a in reference if 0 <= a <= 8)
    found = []
    for name in ('raw','full','half','fixed_oracle'):
        def contrast(a: float) -> float:
            return stein_risk(a)[0]-benchmarks(a)[name]
        for left, right in zip(grid[:-1],grid[1:]):
            fleft = reference[left]-benchmarks(left)[name]
            fright = reference[right]-benchmarks(right)[name]
            if fleft*fright < 0:
                root = brentq(contrast,left,right,xtol=2e-13,rtol=1e-13)
                found.append(dict(comparator=name, left=left, right=right,
                                  root=root, residual=contrast(root)))
    return found


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline='') as stream:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def rejection_first(records, name, bad_call, good_call, predicate):
    def accepts(call):
        try:
            return bool(predicate(call()))
        except (ArithmeticError, ValueError, TypeError, AssertionError):
            return False
    rejected = not accepts(bad_call)
    accepted = accepts(good_call) if rejected else False
    records.append(dict(name=name, mutated_case_rejected=rejected,
                        valid_case_accepted=accepted,
                        status='PASS' if rejected and accepted else 'FAIL'))
    if not (rejected and accepted):
        raise AssertionError(records[-1])


def run(results: Path, producer_path: Path, lock_path: Path, report_path: Path):
    if not lock_path.is_file():
        raise RuntimeError('Protocol lock is required before verification')
    lock = json.loads(lock_path.read_text())
    if not lock:
        raise RuntimeError('Empty protocol lock')
    records = []
    entry_lock_sha = digest(lock_path)
    expected_protocol_sha = lock['protocol_sha256']
    def validate_payload(payload):
        # The complete reconstructible fixture is defined by the locked input
        # JSON plus the one-field mutation recorded in the final receipt.
        with tempfile.TemporaryDirectory(prefix='risk-verifier-lock-') as temp:
            fixture = Path(temp)/'stale_lock.json'
            fixture.write_text(json.dumps(payload))
            return validate_lock(fixture)
    lock_predicate = lambda value: value['protocol_sha256']==expected_protocol_sha
    stale_protocol = json.loads(json.dumps(lock))
    stale_protocol['protocol_sha256'] = '0'*64
    stale_input = json.loads(json.dumps(lock))
    stale_input['input_files'][0]['sha256'] = '0'*64
    rejection_first(records,'entry_protocol_SHA_rebinding',
                    lambda: validate_payload(stale_protocol),
                    lambda: validate_lock(lock_path),lock_predicate)
    rejection_first(records,'entry_input_rebinding',
                    lambda: validate_payload(stale_input),
                    lambda: validate_lock(lock_path),lock_predicate)
    spec = importlib.util.spec_from_file_location('locked_v2_risk_producer', producer_path)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    table_path = results/'risk_map.csv'
    uniform_path = results/'uniform.csv'
    crossing_path = results/'crossings.csv'
    rows = load_rows(table_path)
    uniform_rows = load_rows(uniform_path)
    with crossing_path.open(newline='') as stream:
        crossing_rows = list(csv.DictReader(stream))
    close = lambda x,y,tol=AGREEMENT_ATOL: bool(np.allclose(x,y,rtol=0,atol=tol))

    points = [-2.,-1.,-.5,0.,.5,1.,2.]
    expected_action = [action_reference(x) for x in points]
    actual_action = lambda: [float(producer.action(x)) for x in points]
    rejection_first(records,'implementation_reversed_coefficient',
                    lambda: [reversed_coefficient_mutant(x) for x in points],
                    actual_action, lambda x: close(x,expected_action,1e-14))
    rejection_first(records,'implementation_removed_positive_part',
                    lambda: [missing_positive_part_mutant(x) for x in points],
                    actual_action, lambda x: close(x,expected_action,1e-14))
    exact_zero = 4*(phi(1)-float(ndtr(-1)))
    rejection_first(records,'implementation_random_lambda_in_fixed_formula',
                    lambda: fixed_formula_mutant(0.),
                    lambda: producer.risk(0.)[0],
                    lambda value: abs(value-exact_zero) <= AGREEMENT_ATOL)

    expected_grid = [i/20 for i in range(-160,161)] + [-128.,-64.,-32.,-16.,16.,32.,64.,128.]
    expected_grid = np.array(sorted(expected_grid))
    actual_grid = np.array(sorted(row['a'] for row in rows))
    rejection_first(records,'complete_declared_grid',lambda: actual_grid+.001,
                    lambda: actual_grid,
                    lambda value: len(value)==len(expected_grid) and close(value,expected_grid,1e-12))
    # Bind values to their actual decimal grid key, with symmetric .05 snapping.
    def canonical_a(value):
        return round(value*20)/20 if abs(value)<=8.000001 else float(value)
    by_a = {canonical_a(row['a']):row for row in rows}
    reference = {}
    reference_errors = {}
    for a in expected_grid:
        value,error = stein_risk(float(a))
        reference[float(a)] = value
        reference_errors[float(a)] = error
    saved = np.array([by_a[float(a)]['risk'] for a in expected_grid])
    truth = np.array([reference[float(a)] for a in expected_grid])
    rejection_first(records,'independent_Stein_full_line',lambda: saved+.01,
                    lambda: saved,lambda value: close(value,truth))
    # Fresh primary calls at every grid point verify that CSV values reflect code.
    primary_calls = [producer.risk(float(a)) for a in expected_grid]
    primary = np.array([value for value,error in primary_calls])
    rejection_first(records,'producer_function_matches_saved_results',lambda: primary+.01,
                    lambda: primary,lambda value: close(value,saved,1e-12))
    error_estimates = np.r_[[row['quad_error']+row['tail_bound'] for row in rows],
                           [error for value,error in primary_calls],
                           list(reference_errors.values())]
    rejection_first(records,'quadrature_error_estimates_within_locked_agreement',
                    lambda: np.full_like(error_estimates,2*AGREEMENT_ATOL),
                    lambda: error_estimates,
                    lambda value: bool(np.all(np.isfinite(value))
                                       and np.all(value>=0)
                                       and np.all(value<=AGREEMENT_ATOL)))
    reflected = np.array([by_a[float(-a)]['risk'] for a in expected_grid])
    rejection_first(records,'symmetry',lambda: reflected+.01,
                    lambda: reflected,lambda value: close(value,saved))
    rejection_first(records,'Stein_zero_identity',lambda: reference[0.]+.01,
                    lambda: reference[0.],lambda value: abs(value-exact_zero)<=AGREEMENT_ATOL)
    actual_active = np.array([by_a[float(a)]['active_probability'] for a in expected_grid])
    true_active = np.array([active_probability(float(a)) for a in expected_grid])
    rejection_first(records,'activity_probability',lambda: actual_active+.01,
                    lambda: actual_active,lambda value: close(value,true_active,1e-13))
    function_active = np.array([producer.active_probability(float(a)) for a in expected_grid])
    rejection_first(records,'activity_function',lambda: function_active+.01,
                    lambda: function_active,lambda value: close(value,true_active,1e-13))
    for name in ('raw','full','half','fixed_oracle'):
        actual = np.array([by_a[float(a)][name] for a in expected_grid])
        wanted = np.array([benchmarks(float(a))[name] for a in expected_grid])
        rejection_first(records,'benchmark_'+name,lambda: actual+.01,
                        lambda: actual,lambda value: close(value,wanted,1e-12))
    scaled_large = {str(a): a*a*(reference[a]-1) for a in (32.,64.,128.)}
    actual_large = np.array(list(scaled_large.values()))
    rejection_first(records,'large_signal_asymptote',lambda: actual_large+1,
                    lambda: actual_large,lambda value: bool(np.all(np.abs(value-3)<.05)))
    bound = 2*((12+2)*phi(12)+2*float(ndtr(-12)))
    actual_bounds = np.array([row['tail_bound'] for row in rows])
    rejection_first(records,'finite_domain_tail_bound',lambda: actual_bounds*2,
                    lambda: actual_bounds,
                    lambda value: bool(np.allclose(value,bound,rtol=1e-12,atol=0)))

    uniform_ref = {n:uniform_exact(n) for n in (250,500,700,1000,2000,10000)}
    rejection_first(records,'complete_uniform_grid',lambda: sorted(uniform_ref)[:-1],
                    lambda: sorted(int(row['n']) for row in uniform_rows),
                    lambda value: value==sorted(uniform_ref))
    for row in uniform_rows:
        n = int(row['n'])
        wanted = uniform_ref[n]
        keys = list(wanted)
        actual = np.array([row[key] for key in keys])
        expected = np.array([float(wanted[key]) for key in keys])
        mutant = actual.copy()
        mutant[keys.index('training_delta')] = row['test_delta']
        rejection_first(records,'uniform_exact_'+str(n),lambda: mutant,
                        lambda: actual,lambda value: close(value,expected,1e-13))
    independent_crossings = scan_crossings(reference)
    wanted_crossings = sorted(independent_crossings,key=lambda x:(x['comparator'],x['root']))
    actual_crossings = sorted(crossing_rows,key=lambda x:(x['comparator'],float(x['root'])))
    def crossings_match(value):
        if len(value)!=len(wanted_crossings):
            return False
        return all(row['comparator']==wanted['comparator']
                   and abs(float(row['root'])-wanted['root'])<=1e-9
                   and abs(stein_risk(float(row['root']))[0]
                           -benchmarks(float(row['root']))[row['comparator']])<=1e-10
                   for row,wanted in zip(value,wanted_crossings))
    wrong_crossings = [dict(row) for row in actual_crossings]
    if wrong_crossings:
        wrong_crossings[0]['root'] = float(wrong_crossings[0]['root'])+.1
    else:
        wrong_crossings=[dict(comparator='raw',root=.123)]
    rejection_first(records,'bounded_crossing_scan',lambda: wrong_crossings,
                    lambda: actual_crossings,crossings_match)

    rejection_first(records,'completion_protocol_and_input_rebinding',
                    lambda: validate_payload(stale_input),
                    lambda: validate_lock(lock_path),lock_predicate)
    rejection_first(records,'lock_bytes_unchanged',lambda: '0'*64,
                    lambda: digest(lock_path),lambda value:value==entry_lock_sha)
    inputs = [Path(__file__),producer_path,Path(__file__).with_name('common.py'),
              lock_path,Path(lock['protocol_path']),table_path,uniform_path,crossing_path]
    summary = dict(status='PASS',checks=records,negative_controls=len(records),
                   max_risk_discrepancy=float(np.max(np.abs(saved-truth))),
                   max_reference_quad_error=max(reference_errors.values()),
                   max_all_quad_error_estimates=float(np.max(error_estimates)),
                   gaussian_null_risk=exact_zero,gaussian_null_half_risk=.25,
                   gaussian_null_probability_active=active_probability(0.),
                   tail_bound=bound,large_signal_scaled_excess=scaled_large,
                   crossings=independent_crossings,
                   scope='Deterministic local Gaussian experiment; no financial evidence or global crossing uniqueness.',
                   python=platform.python_version(),numpy=np.__version__,
                   protocol_commit=lock['protocol_commit'],
                   lock_fixture_rules=[{'field':'protocol_sha256','replacement':'0'*64},
                                       {'field':'input_files[0].sha256','replacement':'0'*64}],
                   source_sha256={str(p.resolve()):digest(p) for p in inputs})
    rendered = '# Independent verification: deterministic Gaussian-limit risk\n\n'
    rendered += '**Status: PASS.** Verification ran only after the root supplied the protocol lock and producer outputs.\n\n'
    rendered += ('The producer integrates direct squared error on `[-12,12]`. This verifier '
                 'uses the full-real-line Stein identity, with separate mass-region and branch '
                 'breakpoints. No random histories or financial arrays enter the calculation.\n\n')
    rendered += 'The Gaussian identity is `r(a)=1+E[(X^2-2)1{|X|<=1}+3/X^2 1{|X|>1}]`, `X~N(a,1)`.\n\n'
    rendered += 'At zero signal, `r(0)=4[phi(1)-Q(1)]`; fixed half has risk `1/4`. The large-signal limit checked is `a^2(r(a)-1)->3`.\n\n'
    rendered += ('Uniform training optimism is checked with exact rational arithmetic in two ways: '
                 'order-statistic gap sums versus the closed-form training expectation, and '
                 'second moments versus variance plus squared mean for the test expectation. '
                 'This verifies the stated example, not a universal `2*Ahat` correction.\n\n')
    rendered += ('The three implementation mutants reverse the fraction, remove its positive part, '
                 'or insert the random fraction into the fixed-fraction risk formula. Each is '
                 'rejected before its valid producer counterpart is accepted.\n\n')
    rendered += ('Crossings are bracketed numerical findings on `[0,8]`, reflected by symmetry. '
                 'They do not establish global uniqueness or behaviour between unsampled intervals '
                 'without a bracket. The original failed nuisance gate and panel NOT_RUN verdict remain unchanged.\n\n')
    rendered += '```json\n'+json.dumps(summary,indent=2)+'\n```\n'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(rendered)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--producer',required=True,type=Path)
    parser.add_argument('--lock',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    args = parser.parse_args()
    run(args.results,args.producer,args.lock,args.report)
