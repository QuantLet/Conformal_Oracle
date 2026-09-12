"""Apply the proved boundary penalty to every primary configuration."""
import json
import pandas as pd
from run import ROOT, OUT, sha


def main():
    data=pd.read_csv(OUT/'run/finite.csv')
    rows=[]
    for row in data.itertuples():
        horizon=int(3*row.n//7)
        r=row.retention
        factor=r*(1-r**horizon)/(horizon*(1-r))
        average_theta=(row.theta0+row.theta1)/2
        binary_penalty=average_theta*.375*factor
        scalar_penalty=average_theta*.5*factor
        direct=sum(r**j for j in range(1,horizon+1))/horizon
        assert abs(factor-direct)<1e-14
        rows.append(dict(n=row.n,H=horizon,alpha=row.alpha,epsilon=row.epsilon,retention=r,
                         binary_independent_lower=row.binary_regret,binary_boundary_penalty=binary_penalty,
                         binary_contiguous_lower=row.binary_regret-binary_penalty,
                         scalar_independent_lower=row.scalar_bayes_regret,scalar_boundary_penalty=scalar_penalty,
                         scalar_contiguous_lower=row.scalar_bayes_regret-scalar_penalty))
    path=OUT/'contiguous.csv'
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g')
    inputs=['research/r8_information_limit/CONTIGUOUS_ADDENDUM.md','research/r8_information_limit/PROOF.md',
            'artifacts/r8_information_limit/run/finite.csv']
    result=dict(producer_sha256=sha(__file__),inputs={p:sha(ROOT/p) for p in inputs},
                output_sha256=sha(path),configurations=len(rows),negative_bounds_retained=True)
    (OUT/'contiguous.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
