#!/usr/bin/env python3
"""Compare prespecified closures on actual native deciles for all 48 grid pairs."""
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from panel_statistics import ROOT,load_pair,qshift,scores


def main():
    rows=[];out=ROOT/'results/closures';out.mkdir(exist_ok=True)
    probs=np.arange(1,10)/10;z=norm.ppf(probs);den=z@z
    for name,model in [('TimesFM-2.5','timesfm25'),('Moirai-2.0','moirai2')]:
        for ret in sorted((ROOT/'data/returns').glob('*.csv')):
            asset=ret.stem;y,f=load_pair(name,asset);nc=int(.7*len(y))
            p=pd.read_parquet(ROOT/'parameters'/model/f'{asset}.parquet')
            grid=p[[f'q_{u:g}' for u in probs]].to_numpy();mu=grid.mean(axis=1)
            sigma=np.maximum((grid-mu[:,None])@z/den,1e-8)
            paths={'Student-t':f['VaR_0.01'].to_numpy(),'Gaussian':mu+sigma*norm.ppf(.01),
                   'Linear':grid[:,0]+(.01-.1)*(grid[:,1]-grid[:,0])/.1}
            daily=pd.DataFrame(paths,index=f.index);daily['r']=y;daily.to_parquet(out/f'{model}_{asset}.parquet')
            for closure,q in paths.items():
                shift=qshift(q[:nc]-y[:nc]);raw=scores(y[nc:],q[nc:]);cp=scores(y[nc:],q[nc:]-shift)
                rows.append(dict(model=name,asset=asset,closure=closure,n_cal=nc,qV=shift,
                                 **{k+'_raw':v for k,v in raw.items()},**{k+'_static':v for k,v in cp.items()}))
    pd.DataFrame(rows).to_csv(ROOT/'results/closure_sensitivity.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(producer_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
         grid_source='actual saved native deciles, not reconstructed Student-t grids',
         student='original archived model-specific Student-t fit',gaussian='exact least squares to nine deciles with positive scale',
         linear='linear probability extrapolation from native 0.10 and 0.20 quantiles'),indent=2)+'\n')
    print('48 pairs x 3 native-grid closures complete',flush=True)


if __name__=='__main__':main()
