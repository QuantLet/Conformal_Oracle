"""Admission and information-boundary checks for the external adapter."""
import io
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from prepare import ASSETS,parse,OUT
from dynamic import fit_context,forecast


def zipped(rows,columns=None):
    b=io.BytesIO()
    s='\n'.join(['202607 CRSP','Average Value Weighted Returns -- Daily',
                 ','+','.join(columns or ASSETS),*rows,'',
                 'Average Equal Weighted Returns -- Daily',','+','.join(ASSETS),
                 '20260731,'+','.join(['99']*12)])
    with ZipFile(b,'w') as z:z.writestr('example.csv',s)
    return b.getvalue()


def test_parser_selects_value_weighted_and_preserves_crash():
    f,_=parse(zipped(['20260730,'+','.join(['-20']*12),'20260731,'+','.join(['-99.99','-999']+['1']*10)]))
    assert (f.iloc[0]==-20).all() and f.iloc[1,:2].isna().all()
    assert (f.iloc[1,2:]==1).all()


def test_duplicate_dates_fail():
    row='20260731,'+','.join(['1']*12)
    with pytest.raises(AssertionError):parse(zipped([row,row]))


def test_wrong_portfolio_fails():
    with pytest.raises(AssertionError):parse(zipped(['20260731,'+','.join(['1']*12)],ASSETS[:-1]+['New']))


@pytest.mark.parametrize('name',['CAViaR-AS','GAS-t'])
def test_past_only_refit_and_independent_recursion(name):
    ret=pd.read_csv(OUT/'data/returns/NoDur.csv',index_col='date',parse_dates=True).log_return
    y=ret.to_numpy();first=int(np.flatnonzero(ret.index.year==2000)[0]);last=first+251
    changed=y.copy();changed[first+50:]*=-11
    a=fit_context(y[first-1250:first],name);b=fit_context(changed[first-1250:first],name)
    np.testing.assert_array_equal(a[0],b[0]);assert a[1:]==b[1:]
    th,q0,_=a
    one,scale=forecast(th,y[first-1250:last],name,q0)
    two,_=forecast(th,changed[first-1250:last],name,q0)
    np.testing.assert_array_equal(one[:1301],two[:1301])
    z=y[first-1250:last];ref=np.empty(len(z))
    if name=='CAViaR-AS':
        ref[0]=q0
        for i in range(1,len(z)):ref[i]=th[0]+th[1]*ref[i-1]+th[2]*max(z[i-1],0)+th[3]*max(-z[i-1],0)
        np.testing.assert_array_equal(one,ref)
    else:
        ref[0]=np.log(max(np.std(z[:250]),1e-8))
        for i in range(1,len(z)):
            e=z[i-1]/np.exp(ref[i-1]);score=(th[3]+1)/(th[3]+e**2)*e**2-1
            ref[i]=th[0]+th[2]*ref[i-1]+th[1]*score
        np.testing.assert_allclose(scale,np.exp(ref),rtol=5e-13,atol=1e-15)
