"""New deterministic checks from pinball loss; no study-producer imports or random draws."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/r8_math_audit_fresh'

def rho(x,alpha):return alpha*x if x>=0 else (alpha-1)*x

def integral_rho(lo,hi,alpha):
 def primitive(x):return (alpha if x>=0 else alpha-1)*x*x/2
 return primitive(hi)-primitive(lo)

def uniform_risk(c,sigma,d,alpha):
 # S=sigma*(Z+d), Z uniform[-p,alpha]; integrate rho(c-S)
 p=1-alpha;lo=c-sigma*(alpha+d);hi=c-sigma*(-p+d)
 return integral_rho(lo,hi,alpha)/sigma

def weighted_quantile(values,weights,p):
 total=sum(weights);running=F(0)
 for x,w in sorted(zip(values,weights)):
  running+=w
  if running>=p*total:return x
 raise AssertionError('No rank')

def main():
 counts={'estimator_paths':0,'population_loss_integrals':0,'scale_moment_identities':0,'oscillation_cases':0}
 bad_weight=None
 observations=list(product(map(F,[1,2,3]),map(F,[-3,-1,0,2])))
 for obs in product(observations,repeat=3):
  scale,z=zip(*obs);scores=[s*x for s,x in obs]
  for alpha in map(F,['1/100','1/10','1/4','1/2']):
   c=weighted_quantile(scores,[F(1)]*3,1-alpha)
   b=weighted_quantile(z,scale,1-alpha)
   candidates_c=sorted(set(scores));candidates_b=sorted(set(z))
   f=lambda t:sum(rho(t-x,alpha) for x in scores)
   g=lambda t:sum(rho(t*s-x,alpha) for s,x in zip(scale,scores))
   assert c==min(candidates_c,key=f) and b==min(candidates_b,key=g)
   u=weighted_quantile(z,[F(1)]*3,1-alpha)
   if g(u)>g(b) and bad_weight is None:bad_weight={'scale':list(map(str,scale)),'z':list(map(str,z)),'alpha':str(alpha),'weighted':str(b),'unweighted':str(u),'weighted_objective':str(g(b)),'unweighted_objective':str(g(u))}
   counts['estimator_paths']+=1
 assert bad_weight is not None,'Weight replacement never detected'
 distributions=[([F(1),F(2)],[F(1,2)]*2),([F(1,2),F(1),F(3)],[F(1,4),F(1,2),F(1,4)]),([F(2)],[F(1)])]
 for scales,weights in distributions:
  A=sum(s*w for s,w in zip(scales,weights));B=sum(s*s*w for s,w in zip(scales,weights))/A;C=1/sum(w/s for s,w in zip(scales,weights))
  assert B>=A>=C
  for alpha,d in product(map(F,['1/100','1/10','1/4','1/2']),map(F,['-1/10000','0','1/10000'])):
   cstar=d*C;bstar=d
   riskc=lambda c:sum(w*uniform_risk(c,s,d,alpha) for s,w in zip(scales,weights))
   riskb=lambda b:sum(w*uniform_risk(b*s,s,d,alpha) for s,w in zip(scales,weights))
   assert riskc(0)-riskc(cstar)==d*d*C/2
   assert riskb(0)-riskb(bstar)==d*d*A/2
   for err in map(F,['-1/10000','1/10000']):
    assert riskc(cstar+err)-riskc(cstar)==err*err/(2*C)
    assert riskb(bstar+err)-riskb(bstar)==err*err*A/2
   counts['population_loss_integrals']+=6
  counts['scale_moment_identities']+=1
 for s,t in product(map(F,['1/3','1','2','4']),repeat=2):
  A=(s+t)/2;B=(s*s+t*t)/(s+t);C=2*s*t/(s+t)
  assert B-C==2*(A-C)
  if s!=t:assert (B-C)/(A-C)==2
  counts['scale_moment_identities']+=1
 # Oscillation over both test score and changing bounded scale.
 for alpha,x,x0 in product(map(F,['1/100','1/4','1/2','9/10']),map(F,[-2,0,3]),map(F,[-1,0,2])):
  values=[rho(a*x-s,alpha)-rho(a*x0-s,alpha) for a,s in product(map(F,['1/2','1','3']),map(F,[-100,-2,0,3,100]))]
  assert max(values)-min(values)<=3*abs(x-x0)
  counts['oscillation_cases']+=1
 before=json.loads((OUT/'before.json').read_text())
 for name,h in before['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 result={'status':'passed','independent_implementation':True,'counts':counts,'weighting_negative_control':bad_weight,'canonical_files_unchanged':len(before['files']),'new_random_draws':0,'proof_scope':'Finite exact cases check identities and detect implementation confusions; they do not prove asymptotic theorems.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'shape_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
