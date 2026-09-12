"""Independent rejection-first checks; authored after failing_cases.json."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/theory_loop'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def record_check(records, name, bad, good, predicate):
    """The broken case is evaluated FIRST; no PASS without that rejection."""
    def accepts(x):
        try:
            return bool(predicate(x))
        except (AssertionError, ValueError, FileNotFoundError, KeyError):
            return False
    rejected = not accepts(bad)
    accepted = accepts(good) if rejected else False
    records.append(dict(check=name, built_to_fail_rejected=rejected,
                        valid_case_accepted=accepted,
                        status='PASS' if rejected and accepted else 'FAIL'))
    if not (rejected and accepted):
        raise AssertionError(records[-1])


def admission(metrics):
    required = [r for r in metrics if r['n'] in (700, 1000)]
    return (len(required) == 4 and all(
        r['finite_histories'] == 500 and r['median_omega_relative_error'] < .15
        and r['median_density_relative_error'] < .15
        and .85 <= r['variance_ratio'] <= .98 for r in required))


def require_panel_admission():
    obj = json.loads((OUT / 'synthetic/admission.json').read_text())
    if obj['status'] != 'PASS':
        raise RuntimeError('Synthetic admission failed; panel access is prohibited.')


def preflight():
    import estimators as e
    fixtures = json.loads((ROOT / 'research/r8_theory_loop/failing_cases.json').read_text())
    lock = json.loads((OUT / 'lock.json').read_text())
    checks = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        good = ROOT / 'analysis_plan_theory_loop.md'
        bad = td / 'altered'; bad.write_bytes(good.read_bytes()+b'x')
        record_check(checks, 'plan_lock', bad, good, lambda p: sha(p)==lock['plan_sha256'])
        good_fixture = ROOT / 'research/r8_theory_loop/failing_cases.json'
        record_check(checks, 'fixture_lock', bad, good_fixture, lambda p: sha(p)==lock['fixtures_sha256'])
        committed = subprocess.check_output(['git', '-C', str(ROOT/'research/r8_theory_loop/protocol_repository'),
                                             'show', lock['protocol_commit']+':analysis_plan_theory_loop.md'])
        record_check(checks, 'committed_plan_matches', committed+b'x', committed,
                     lambda b: hashlib.sha256(b).hexdigest()==lock['plan_sha256'])
        for n in (125,250,500,700,1000,2000):
            expected = (99*(n+1)+99)//100
            record_check(checks, f'rank_{n}', expected-1, e.rank(n), lambda x: x==expected)
        s=np.linspace(-3.,1.,250)
        for c in (-1.2,.3):
            ref=np.mean(.01*np.maximum(c-s,0)+.99*np.maximum(s-c,0)
                        -.01*np.maximum(-s,0)-.99*np.maximum(s,0))
            val=e.ecdf_integral(s,c)
            record_check(checks, f'ecdf_integral_{c}', -val, val,
                         lambda x: np.isclose(x,ref,rtol=1e-11,atol=1e-13))
        hits=np.array([1,1,1,0,0,0,1,1,0,1,1,1,1,0.],float)
        centered=hits-hits.mean(); L=3; n=len(hits)
        matrix=np.fromfunction(lambda i,j: np.maximum(0,1-np.abs(i-j)/(L+1)),(n,n))
        ref=float(centered@matrix@centered/n)
        bad=float((hits-.99)@matrix@(hits-.99)/n)
        val=e.hac(hits,L,False)[0]
        record_check(checks,'hac_sample_center',bad,val,lambda x:abs(x-ref)<1e-13)
        flat=float(np.mean(centered**2)+2*sum(centered[j:]@centered[:-j]/n for j in range(1,L+1)))
        record_check(checks,'hac_bartlett_weights',flat,val,lambda x:abs(x-ref)<1e-13)
        # Independent AR(1) coefficient and continuously weighted Bartlett sum.
        rho=sum(centered[i]*centered[i-1] for i in range(1,n))/sum(x*x for x in centered[:-1])
        bw=1.1447*(n*4*rho*rho/(1-rho*rho)**2)**(1/3)
        refa=sum(x*x for x in centered)/n+2*sum(max(0,1-j/bw)*
                    sum(centered[i]*centered[i-j] for i in range(j,n))/n for j in range(1,n))
        actual=e.andrews(hits,False)[0]
        record_check(checks,'andrews_automatic',actual*2,actual,lambda x:abs(x-refa)<1e-13)
        floor=e.hac(np.ones(200),4,True)[0]
        record_check(checks,'omega_floor',0.,floor,lambda x:x==.00099)
        c,f,omega,n=.02,5.,.0099,700
        refh=max(0,n*c*c-omega/(f*f))
        record_check(checks,'h2_bias_correction',n*c*c-omega/(n*f*f),e.h2(c,n,omega,f),
                     lambda x:abs(x-refh)<1e-13)
        lam=f*f*refh/(omega+f*f*refh)
        record_check(checks,'lambda',1-lam,e.shrinkage(c,n,omega,f),lambda x:abs(x-lam)<1e-13)
        x=np.array([-2.,-1.,0.,1.,2.]); bw=.5
        refd=sum(np.exp(-.5*(u/bw)**2)/(bw*np.sqrt(2*np.pi)) for u in x)/len(x)
        actual=e.gaussian_density(x,0.,bw)
        record_check(checks,'kernel_density',actual*2,actual,lambda x:abs(x-refd)<1e-13)
        def density_domain(bandwidth):
            return np.isfinite(e.gaussian_density(x,0.,bandwidth))
        record_check(checks,'bandwidth_domain',-.5,.5,density_domain)
        spacing=e.spacing(np.arange(700,dtype=float))
        record_check(checks,'spacing_rank_failure',{'status':'OK','f':1.},spacing,
                     lambda r:r['status']=='OUT_OF_RANGE' and r['upper']>700)
        # Admissible spacing arithmetic (median probability only for this unit fixture).
        actual=e.spacing(np.arange(100,dtype=float),p=.5, m=2)
        record_check(checks,'spacing_arithmetic',actual['f']*2,actual['f'],
                     lambda x:abs(x-(4/100)/3)<1e-13)
        data=np.linspace(-1,1,1000); future=data.copy();future[700:]+=100
        safe=lambda a:e.rank_shift(a[:700])
        bad=float(safe(future)+future[700]-data[700])
        record_check(checks,'future_exclusion',bad,safe(future),lambda x:x==safe(data))
        A,B,C,g,h=1.5,5/3,4/3,1.,.1
        delta=g*g*h-.0099*(B-C)/(A-C)
        record_check(checks,'boundary_direction',-delta,e.boundary(np.array([1.,2.]),g,h),
                     lambda x:abs(x-delta)<1e-13)
        record_check(checks,'variance_ratio_threshold',1.2,.9,lambda x:.85<=x<=.98)
        for field in ('omega','density'):
            record_check(checks,field+'_error_threshold',.16,.1,lambda x:x<.15)
        good_rows=[dict(n=n,law=law,finite_histories=500,median_omega_relative_error=.1,
                     median_density_relative_error=.1,variance_ratio=.9,coverage=.95)
                   for n in (700,1000) for law in ('normal','t5')]
        bad_rows=[{**r,'variance_ratio':1.2} for r in good_rows]
        record_check(checks,'coverage_cannot_replace_variance_ratio',bad_rows,good_rows,admission)
        bad_rows=[{**r,'median_density_relative_error':.16} for r in good_rows]
        record_check(checks,'density_blocks_panel',bad_rows,good_rows,admission)
        record_check(checks,'truth_precision',.03,.01,lambda x:x<=.02)
        for law in ('normal','t5'):
            d=stats.norm() if law=='normal' else stats.t(5,scale=np.sqrt(3/5))
            sigma=.017; q=-.035
            ref=(integrate.quad(lambda z:.99*(q-sigma*z)*d.pdf(z),-np.inf,q/sigma,
                               epsabs=1e-13)[0]+integrate.quad(lambda z:.01*(sigma*z-q)*d.pdf(z),
                               q/sigma,np.inf,epsabs=1e-13)[0])
            val=float(e.expected_loss(law,q,sigma))
            record_check(checks,f'expected_loss_{law}',val*2,val,
                         lambda x:np.isclose(x,ref,rtol=4e-11,atol=4e-11))
        digest=sha(good)
        record_check(checks,'manifest_corruption',bad if isinstance(bad,Path) else td/'altered',
                     good,lambda p:sha(p)==digest)
        record_check(checks,'manifest_missing',td/'missing',good,lambda p:sha(p)==digest)
        old=json.loads((ROOT/'artifacts/r8_financial_argument/final_validation.json').read_text())['current_file_sha256']
        broken=dict(old);first=next(iter(broken));broken[first]='0'*64
        verify=lambda binding:all(sha(ROOT/p)==h for p,h in binding.items())
        record_check(checks,'R8_conservation',broken,old,verify)
        (OUT/'preserved_R8.json').write_text(json.dumps(old,indent=2)+'\n')
        empty=td/'empty';empty.write_text(''); space=td/'space';space.write_text('bad \n')
        def whitespace(path):
            r=subprocess.run(['git','diff','--no-index','--check',str(empty),str(path)],capture_output=True,text=True)
            return r.returncode in (0,1) and not r.stdout and not r.stderr
        record_check(checks,'whitespace_guard',space,good,whitespace)
    report={'status':'PASS','checks':checks,'negative_controls':len(checks),
            'fixtures_sha256':sha(ROOT/'research/r8_theory_loop/failing_cases.json')}
    (OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'preflight':report['status'],'rejection_first_checks':len(checks)}))


if __name__=='__main__':
    preflight()
