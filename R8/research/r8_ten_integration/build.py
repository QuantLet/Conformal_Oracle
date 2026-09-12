"""Paper displays from validated native-ten and external-industry results only."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/irfa-ten-paper-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/irfa-ten-paper-cache')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

PROJECT=Path(__file__).resolve().parents[2];SOURCE=PROJECT/'source';OUT=PROJECT/'artifacts/r8_ten_integration'
TEN=PROJECT/'artifacts/r8_ten_comparators';NATIVE=PROJECT/'artifacts/r8_model_extension/ten_common_evaluation'
EXT=PROJECT/'artifacts/r8_external/july2026'
REV=PROJECT/'artifacts/r8_referee_revision'
GBM=REV/'gbm'
MODELS=['Chronos-2','PatchTST-FM','TS-ICL','Moirai-1.1','Lag-Llama','GJR-GARCH','GJR-GARCH-t','GARCH-N','Hist-Sim','EWMA']
LABELS={'GJR-GARCH':'GJR-N','GJR-GARCH-t':'GJR-t','Hist-Sim':'Historical simulation','DtACI-projected-expected':'Projected DtACI',
        'State-L1':'Regularised state','Rolling500':'Rolling 500','Selected-rolling':'Selected rolling',
        'Gate-selected-rolling':'Coverage-gated rolling','Loss-gate':'Loss gate','Past-minimum':'Past-loss minimum'}
GROUPS={'Equity':['SP500','GDAXI','FCHI','FTSE100','STOXX','NIKKEI','HSI','ASX200','BOVESPA','NIFTY','ICLN'],
        'Bonds':['TLT','IBGL','CBU0'],'Commodity':['DJCI','GLD','USO','UNG'],'Crypto':['BTC','ETH'],'FX':['EURUSD','GBPUSD','USDJPY','AUDUSD']}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_plots():
    p=PROJECT/'research/r8_integration/paper_figures.py';spec=importlib.util.spec_from_file_location('ten_paper_plots',p)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def traffic(pairs):
    assets=sum(GROUPS.values(),[]);colors=['#00AD70','#FFCC00','#F04438'];plots=load_plots();plots.style()
    fig,axes=plt.subplots(2,1,figsize=(8.2,6.4),sharex=True)
    for ax,method,title in zip(axes,['Raw','Shift-CP'],['A. Raw forecasts','B. Static conformal correction']):
        frame=pairs[pairs.method==method].set_index(['model','asset'])
        values=np.array([[{'Green':0,'Yellow':1,'Red':2}[frame.loc[(m,a),'TL']] for a in assets] for m in MODELS])
        ax.imshow(values,cmap=ListedColormap(colors),vmin=-.5,vmax=2.5,aspect='auto',interpolation='nearest')
        ax.set_yticks(range(10),[LABELS.get(m,m) for m in MODELS],fontsize=9)
        ax.set_title(title,loc='left',weight='bold',pad=34 if method=='Raw' else 12);ax.set_xticks(np.arange(-.5,24,1),minor=True)
        ax.set_yticks(np.arange(-.5,10,1),minor=True);ax.grid(which='minor',color='white',lw=.55)
        ax.tick_params(which='both',length=0);ax.patch.set_alpha(0)
        for spine in ax.spines.values():spine.set_visible(False)
        pos=0
        for name,group in GROUPS.items():
            if pos:ax.axvline(pos-.5,color='white',lw=2.4)
            if method=='Raw':ax.text(pos+(len(group)-1)/2,-.98,name,ha='center',va='bottom',fontsize=9)
            pos+=len(group)
    axes[-1].set_xticks(range(24),assets,rotation=55,ha='right',fontsize=8.5)
    fig.legend(handles=[Patch(color=c,label=s) for c,s in zip(colors,['At most 4','Above 4 to 9','Above 9'])],
               title='Violations per 250 observations (scaled)',loc='lower center',bbox_to_anchor=(.57,.008),ncol=3,frameon=False,fontsize=9,title_fontsize=9)
    fig.subplots_adjust(left=.23,right=.99,top=.91,bottom=.22,hspace=.32);fig.patch.set_alpha(0);return fig


def review_frontier(summary,intervals,sens):
    plots=load_plots();plots.style();fig,(ax,bx)=plt.subplots(2,1,figsize=(7,7.1))
    BLUE,GREEN,PINK,ORANGE,GREY=plots.BLUE,plots.GREEN,plots.PINK,plots.ORANGE,plots.GREY
    full=intervals[(intervals.reference=='Shift-CP')&(intervals.block_calendar_days==60)].set_index('method')
    ex=sens[(sens.sensitivity=='without_crypto')&(sens.block_calendar_days==60)].set_index('method')
    names=['Raw','Vol-ERM','State-L1','POT-Shift','POT-Vol','DtACI-projected-expected','Loss-gate','Past-minimum']
    labels=['Raw','Vol-ERM','Regularised state','POT-Shift','POT-Vol','Projected DtACI (diagnostic)','Loss gate','Past-loss minimum']
    for offset,frame,c in [(-.14,full,BLUE),(.14,ex,PINK)]:
        for j,name in enumerate(names):
            row=frame.loc[name];v=row['difference']
            ax.errorbar(v,j+offset,xerr=[[v-row.simultaneous_lower],[row.simultaneous_upper-v]],fmt='o',color=c,ms=4,capsize=2,lw=1.5)
    ax.axvline(0,color=GREY,lw=.7,ls='--');ax.set_yticks(range(len(names)),labels);ax.invert_yaxis()
    ax.set_xlabel('QS difference from Shift-CP ×10⁴')
    ax.set_title('A. One reporting family, including raw and Vol-ERM',loc='left',weight='bold')
    mapping={'Raw':('Raw',BLUE,(8,0)),'Shift-CP':('Shift-CP',BLUE,(8,-15)),
             'Vol-ERM':('Vol-ERM',BLUE,(-4,9)),'Rolling500':('Rolling 500',BLUE,(8,-8)),
             'Loss-gate':('Loss gate',ORANGE,(8,1)),
             'Past-minimum':('Past-loss minimum',GREEN,(-101,9))}
    for name,(label,c,offset) in mapping.items():
        row=summary.loc[name];x=row.QS_x10000;y=row.violation_rate*100
        bx.scatter(x,y,color=c,s=30);bx.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=9,color=c)
    bx.axhline(1,color=GREY,ls='--',lw=.8)
    plotted=summary.loc[list(mapping)]
    xmin,xmax=plotted.QS_x10000.min(),plotted.QS_x10000.max()
    padding=max(.04,(xmax-xmin)*.32)
    bx.set_xlim(xmin-padding,xmax+padding)
    bx.text(.98,1.02,'Nominal 1%',transform=bx.get_yaxis_transform(),ha='right',fontsize=9,color=GREY)
    bx.set_ylim(min(.87,plotted.violation_rate.min()*100-.12),max(1.98,plotted.violation_rate.max()*100+.12))
    bx.set_xlabel('Mean QS ×10⁴');bx.set_ylabel('Mean violation rate (%)')
    bx.set_title('B. Selection can retain under-protection',loc='left',weight='bold')
    fig.legend(handles=[Line2D([],[],color=c,marker='o',lw=1.6,label=label)
              for c,label in [(BLUE,'All 24 assets'),(PINK,'Without Bitcoin and Ethereum')]],
              loc='lower center',bbox_to_anchor=(.54,.002),ncol=2,frameon=False,fontsize=9.5)
    fig.subplots_adjust(left=.32,right=.975,top=.945,bottom=.14,hspace=.69)
    plots.finish(fig);return fig


def external_bands(intervals):
    plots=load_plots();plots.style();fig,axes=plt.subplots(2,1,figsize=(7,5.7))
    configs=[('correction',['State-L1','POT-Shift','POT-Vol','DtACI-projected-expected'],'A. Correction relative to Shift-CP'),
             ('indication',['Gate-selected-rolling','Loss-gate','Past-minimum'],'B. Indication relative to selected rolling')]
    for ax,(family,names,title) in zip(axes,configs):
        for offset,block,color in [(-.12,20,'#0075FF'),(.12,60,'#F47812')]:
            g=intervals[(intervals.units=='normalised')&(intervals.family==family)&(intervals.block_calendar_days==block)].set_index('method')
            for i,name in enumerate(names):
                r=g.loc[name];v=r.difference*1000
                ax.errorbar(v,i+offset,xerr=[[(r.difference-r.simultaneous_lower)*1000],[(r.simultaneous_upper-r.difference)*1000]],fmt='o',color=color,ms=4,capsize=2,lw=1.5)
        ax.axvline(0,color='#536070',lw=.8,ls='--');ax.set_yticks(range(len(names)),[LABELS.get(n,n) for n in names]);ax.invert_yaxis()
        ax.set_title(title,loc='left',weight='bold');ax.set_xlabel('Normalised QS difference ×10³')
    fig.legend(handles=[Line2D([],[],color=c,marker='o',lw=1.5,label=s) for c,s in [('#0075FF','20-calendar-day blocks'),('#F47812','60-calendar-day blocks')]],
               loc='lower center',bbox_to_anchor=(.58,.005),ncol=2,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.28,right=.98,top=.94,bottom=.16,hspace=.82);plots.finish(fig);return fig


def build(dest):
    inputs=[TEN/'validation.json',TEN/'aggregation_validation.json',TEN/'results/complete.json',
            NATIVE/'complete.json',NATIVE/'validation.json',EXT/'completion.json',EXT/'aggregation_validation.json',
            PROJECT/'research/r8_integration/paper_figures.py',REV/'aggregation_validation.json',REV/'results/complete.json',
            REV/'results/intervals.csv',REV/'results/sensitivity_intervals.csv',PROJECT/'research/r8_referee_revision/PROTOCOL.md']
    inputs += [GBM/'validation.json',GBM/'evaluation/complete.json',GBM/'evaluation/summary.csv',
               GBM/'evaluation/intervals.csv',GBM/'run/configuration.json',PROJECT/'research/r8_referee_revision/GBM_PROTOCOL.md']
    inputs += [TEN/'results'/f'{n}.csv' for n in ['summary','pairs','intervals','sensitivity_intervals','decisions']]
    inputs += [NATIVE/f'{n}.csv' for n in ['summary','intervals']]
    inputs += [EXT/'results'/f'{n}.csv' for n in ['summary','intervals','decisions']]
    for p in [TEN/'validation.json',TEN/'aggregation_validation.json',NATIVE/'validation.json',EXT/'aggregation_validation.json']:assert json.loads(p.read_text())['status']=='passed'
    assert json.loads((EXT/'completion.json').read_text())['status']=='complete'
    for folder,report,key in [(TEN/'results',TEN/'aggregation_validation.json','complete_sha256'),
                              (NATIVE,NATIVE/'validation.json','complete_sha256'),
                              (EXT/'results',EXT/'aggregation_validation.json','result_complete_sha256'),
                              (REV/'results',REV/'aggregation_validation.json','complete_sha256'),
                              (GBM/'evaluation',GBM/'validation.json','evaluation_complete_sha256')]:
        assert json.loads(report.read_text())['status']=='passed'
        assert json.loads(report.read_text())[key]==sha(folder/'complete.json')
        for path,h in json.loads((folder/'complete.json').read_text())['outputs'].items():assert sha(folder/path)==h,path
    d=pd.read_csv(TEN/'results/summary.csv').set_index('method');pairs=pd.read_csv(TEN/'results/pairs.csv')
    ci=pd.read_csv(REV/'results/intervals.csv');sens=pd.read_csv(REV/'results/sensitivity_intervals.csv');dec=pd.read_csv(TEN/'results/decisions.csv')
    native=pd.read_csv(NATIVE/'summary.csv').set_index(['model','method']);nci=pd.read_csv(NATIVE/'intervals.csv')
    e=pd.read_csv(EXT/'results/summary.csv').set_index('method');eci=pd.read_csv(EXT/'results/intervals.csv');edec=pd.read_csv(EXT/'results/decisions.csv')
    gbm=pd.read_csv(GBM/'evaluation/summary.csv').set_index('method');gci=pd.read_csv(GBM/'evaluation/intervals.csv')
    macros={};written=[];section=dest/'sections_r8';figdir=dest/'figures';section.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    macros['nGbmVersion']=json.loads((GBM/'run/configuration.json').read_text())['lightgbm']
    def macro(name,value,precision=4):macros[name]=f'{value:.{precision}f}'
    def write(name,text):
        p=section/name;p.write_text(text+'\n');written.append(p)
    for prefix,frame in [('nTen',d),('nExt',e)]:
        for key,tag in [('Raw','Raw'),('Shift-CP','Static'),('Vol-ERM','VolErm'),('POT-Shift','Pot'),('POT-Vol','PotVol'),('State-L1','Lone'),('DtACI-projected-expected','Dtaci'),('Loss-gate','Gate'),('Past-minimum','Past')]:
            macro(prefix+tag+'QS',frame.loc[key,'QS_x10000']);macro(prefix+tag+'Pi',frame.loc[key,'violation_rate']*100,3)
            macro(prefix+tag+'Kup',frame.loc[key,'kupiec_rejections'],0)
        macro(prefix+'Pairs',frame.loc['Raw','pairs'],0)
    macro('nTenGateRaw',(dec.gate_selected=='Raw').sum(),0);macro('nTenGateApply',(dec.gate_selected!='Raw').sum(),0);macro('nTenGateWorse',d.loc['Loss-gate','worse_than_raw'],0)
    macro('nTenStaticCC',native.xs('Static',level='method').conditional_rejections.sum(),0)
    macro('nTenStaticCCN',native.xs('Static',level='method').conditional_available.sum(),0)
    for block,word in [(20,'Twenty'),(60,'Sixty')]:
        for method,tag in [('Raw','StaticRaw'),('Vol-ERM','VolStatic')]:
            r=ci[(ci.method==method)&(ci.reference=='Shift-CP')&(ci.block_calendar_days==block)].iloc[0]
            values=[r.difference,r.simultaneous_lower,r.simultaneous_upper]
            if method=='Raw':values=[-values[0],-values[2],-values[1]]
            for name,value in zip(['Delta','Lo','Hi'],values):macro('nTen'+tag+word+name,value)
    for b,word in [(20,'Twenty'),(60,'Sixty')]:
        r=ci[(ci.method=='POT-Shift')&(ci.reference=='Shift-CP')&(ci.block_calendar_days==b)].iloc[0]
        for col,tag in [('difference','Delta'),('simultaneous_lower','Lo'),('simultaneous_upper','Hi')]:macro('nTenPot'+word+tag,r[col])
    r=sens[(sens.method=='POT-Shift')&(sens.sensitivity=='without_crypto')&(sens.block_calendar_days==60)].iloc[0]
    for col,tag in [('difference','Delta'),('simultaneous_lower','Lo'),('simultaneous_upper','Hi')]:macro('nTenExCrypto'+tag,r[col])
    for model,tag in [('Chronos-2','Chronos'),('PatchTST-FM','Patch'),('TS-ICL','Tsicl')]:
        for method,label in [('Raw','Raw'),('Static','Static'),('Rolling250','Rolling')]:macro('nTen'+tag+label+'QS',native.loc[(model,method),'QS_x10000'])
        static=nci[(nci.lhs==model+'/Static')&(nci.rhs==model+'/Raw')]
        assert len(static)==2 and (static.simultaneous_lo_x10000<=0).all() and (static.simultaneous_hi_x10000>=0).all()
    for model in ['Chronos-2','TS-ICL']:
        r=nci[(nci.lhs==model+'/Rolling250')&(nci.rhs==model+'/Raw')];assert len(r)==2 and (r.simultaneous_lo_x10000>0).all()
    macro('nExtGateRaw',(edec.loss_gate=='Raw').sum(),0);macro('nExtGateApply',(edec.loss_gate!='Raw').sum(),0)
    assert (eci.simultaneous_lower<0).all() and (eci.simultaneous_upper>0).all()
    for method,tag in [('Raw','Raw'),('Static','Static'),('Rolling250','Rolling')]:
        macro('nGbm'+tag+'QS',gbm.loc[method,'QS_x10000']);macro('nGbm'+tag+'Pi',gbm.loc[method,'pi_mean']*100,3)
        macro('nGbm'+tag+'Kup',gbm.loc[method,'kupiec_rejections'],0)
    for block,word in [(20,'Twenty'),(60,'Sixty')]:
        r=gci[(gci.lhs=='GBM/Static')&(gci.rhs=='GBM/Raw')&(gci.block_length==block)].iloc[0]
        for col,tag in [('estimate','Delta'),('simultaneous_lower','Lo'),('simultaneous_upper','Hi')]:macro('nGbmStaticRaw'+word+tag,r[col])
    corrected=gci[(gci.lhs=='GBM/Static')&(gci.rhs!='GBM/Raw')]
    assert len(corrected)==20 and (corrected.simultaneous_lower<0).all() and (corrected.simultaneous_upper>0).all()
    write('numbers_ten_external.tex','% Generated by research/r8_ten_integration/build.py.\n'+'\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name,value in sorted(macros.items())))
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Forecaster & QS raw & QS static & QS rolling & $\pi$ raw & $\pi$ static & UC raw/static \\',r'\midrule']
    for model in MODELS:
        raw=native.loc[(model,'Raw')];stat=native.loc[(model,'Static')];roll=native.loc[(model,'Rolling250')]
        lines.append(f'{LABELS.get(model,model)} & {raw.QS_x10000:.4f} & {stat.QS_x10000:.4f} & {roll.QS_x10000:.4f} & {raw.pi_mean*100:.3f} & {stat.pi_mean*100:.3f} & {int(raw.kupiec_rejections)}/{int(stat.kupiec_rejections)}'+r' \\')
    lines += [r'\midrule',r'\multicolumn{7}{l}{\emph{Additional supervised benchmark, same asset dates}} \\']
    raw=gbm.loc['Raw'];stat=gbm.loc['Static'];roll=gbm.loc['Rolling250']
    lines.append(f'Direct-quantile GBM & {raw.QS_x10000:.4f} & {stat.QS_x10000:.4f} & {roll.QS_x10000:.4f} & {raw.pi_mean*100:.3f} & {stat.pi_mean*100:.3f} & {int(raw.kupiec_rejections)}/{int(stat.kupiec_rejections)}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];write('tab_native_forecasters.tex','\n'.join(lines))
    selected=['Raw','Shift-CP','Vol-ERM','State-L1','POT-Shift','POT-Vol','Rolling500','Loss-gate','Past-minimum']
    lines=[r'\begin{tabular}{lrrrrl}',r'\toprule',r'Method & QS & Viol. (\%) & UC & Worse & Simultaneous band \\',r'\midrule']
    for name in selected:
        r=d.loc[name];uc=str(int(r.kupiec_rejections)) if r.kupiec_available else '--'
        band='--'
        rows=ci[(ci.method==name)&(ci.reference=='Shift-CP')&(ci.block_calendar_days==60)]
        if len(rows) and np.isfinite(rows.iloc[0].simultaneous_lower):
            row=rows.iloc[0];band=f'$[{row.simultaneous_lower:.4f}, {row.simultaneous_upper:.4f}]$'
        lines.append(f'{LABELS.get(name,name)} & {r.QS_x10000:.4f} & {100*r.violation_rate:.3f} & {uc} & {int(r.worse_than_raw)} & {band}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];write('tab_ten_strong.tex','\n'.join(lines))
    selected=selected[:6]+['DtACI-projected-expected']+selected[6:-2]+['Selected-rolling','Gate-selected-rolling','Loss-gate','Past-minimum']
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Method & QS & Normalised QS & Viol. (\%) & UC & Worse \\',r'\midrule']
    for name in selected:
        r=e.loc[name];uc=str(int(r.kupiec_rejections)) if r.kupiec_available else '--'
        lines.append(f'{LABELS.get(name,name)} & {r.QS_x10000:.4f} & {r.normalised_QS:.6f} & {100*r.violation_rate:.3f} & {uc} & {int(r.worse_than_raw)}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];write('tab_external.tex','\n'.join(lines))
    plots=load_plots();frontier=review_frontier(d,ci,sens)
    for annotation in frontier.axes[1].texts:
        if annotation.get_text()=='Shift-CP':annotation.set_position((8,-5))
    figures=[('fig_ten_traffic',traffic(pairs)),('fig_ten_strong',frontier),('fig_external',external_bands(eci))]
    for name,fig in figures:
        for ext in ['pdf','png','svg']:
            p=figdir/f'{name}.{ext}';kw={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else ({'metadata':{'Date':None}} if ext=='svg' else {})
            with matplotlib.rc_context({'svg.hashsalt':'irfa-ten-external'}):fig.savefig(p,transparent=True,bbox_inches='tight',pad_inches=.06,dpi=210,**kw)
            if ext=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
            written.append(p)
        plt.close(fig)
    return dict(producer_sha256=sha(__file__),inputs={str(p.relative_to(PROJECT)):sha(p) for p in inputs},
                outputs={str(p.relative_to(dest)):sha(p) for p in written},macros=macros)


def check():
    old=json.loads((OUT/'displays.json').read_text())
    def verify(m):
        assert m['producer_sha256']==sha(__file__)
        for p,h in m['inputs'].items():assert sha(PROJECT/p)==h,p
        for p,h in m['outputs'].items():assert sha(SOURCE/p)==h,p
    verify(old);bad=json.loads(json.dumps(old));bad['outputs'][next(iter(bad['outputs']))]='0'*64
    try:verify(bad)
    except AssertionError:pass
    else:raise AssertionError('Corruption negative control did not fail')
    with tempfile.TemporaryDirectory(prefix='irfa-ten-displays-') as d:assert build(Path(d))==old
    return dict(exact_display_replay=len(old['outputs']),negative_control=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');a=ap.parse_args()
    if a.check:print(json.dumps(check()))
    else:
        m=build(SOURCE);OUT.mkdir(exist_ok=True);(OUT/'displays.json').write_text(json.dumps(m,indent=2)+'\n');print('Built',len(m['outputs']),'displays')
