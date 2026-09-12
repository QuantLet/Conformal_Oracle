"""Scientific figures: transparent canvases, vivid colours, bottom legends."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/irfa-mechanism-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/irfa-mechanism-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import engine as e

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#95A1B1','xtick.color':'#39445B','ytick.color':'#39445B',
    'text.color':'#17243B','axes.labelcolor':'#17243B','savefig.transparent':True,
    'svg.fonttype':'none','figure.dpi':150})


def finish(fig,name):
    for ax in fig.axes:
        ax.patch.set_alpha(0);ax.grid(axis='y',color='#98A5B9',alpha=.22,linewidth=.7)
        ax.set_axisbelow(True)
    fig.patch.set_alpha(0)
    for ext in ('png','svg'):
        fig.savefig(e.OUT/f'{name}.{ext}',transparent=True,bbox_inches='tight',pad_inches=.16,dpi=210)
    plt.close(fig)


def dependence(df):
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),sharex=True)
    colors={0.:'#0075FF',.5:'#00A878',.8:'#E6007E'}
    for i,alpha in enumerate(e.ALPHAS):
        for j,kind in enumerate(('normal','t5')):
            ax=axes[i,j];g=df[(df.module=='ar')&(df.innovation==kind)&(df.alpha==alpha)&(df.truth=='constant')]
            for phi,c in colors.items():
                for method,ls,marker in [('Shift-CP','-','o'),('POT80-Shift','--','s')]:
                    x=g[(g.phi==phi)&(g.method==method)].sort_values('n_cal')
                    y=x['difference_vs_Raw'].to_numpy()*10000
                    se=x['MCSE_vs_Raw'].to_numpy()*10000
                    ax.plot(x.n_cal,y,color=c,linestyle=ls,marker=marker,markersize=4,linewidth=1.9)
                    ax.fill_between(x.n_cal,y-1.96*se,y+1.96*se,color=c,alpha=.07,linewidth=0)
            ax.axhline(0,color='#536070',linewidth=.8)
            ax.set_xscale('log',base=2);ax.set_xticks(e.SIZES);ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_title(f'{"Normal" if kind=="normal" else "Student t(5)"} margins · {alpha:.0%} tail',loc='left',fontweight='bold')
            if j==0:ax.set_ylabel('QS change from raw × 10⁴')
            if i==1:ax.set_xlabel('Calibration observations')
    handles=[Line2D([],[],color=c,lw=2,label=f'AR coefficient {phi:g}') for phi,c in colors.items()]
    handles += [Line2D([],[],color='#39445B',lw=2,ls=ls,marker=mk,label=label)
                for ls,mk,label in [('-','o','Empirical conformal shift'),('--','s','POT shift · 80% threshold')]]
    fig.legend(handles=handles[:3],loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,
               columnspacing=2.4,handlelength=2.6)
    fig.legend(handles=handles[3:],loc='lower center',bbox_to_anchor=(.5,.02),ncol=2,frameon=False,
               columnspacing=2.4,handlelength=2.6)
    fig.suptitle('When dependent calibration costs more than it corrects',x=.075,y=.995,ha='left',fontsize=17,fontweight='bold')
    fig.text(.075,.951,'Within each panel: same marginal law and raw bias. Independent evaluation; below zero means lower loss.',fontsize=9.8)
    fig.text(.075,.14,'Shading: ±1.96 paired Monte Carlo standard errors; 500 histories. Bands are pointwise.',fontsize=8.5,color='#536070')
    fig.subplots_adjust(left=.075,right=.98,top=.87,bottom=.25,wspace=.19,hspace=.35)
    finish(fig,'dependence_cost')


def complexity(df,alpha):
    fig,axes=plt.subplots(2,3,figsize=(12.2,7.4),sharex=True)
    style={'Raw':('#536070','--'), 'Shift-CP':('#0075FF','-'),'Vol-ERM':('#00A878','-'),
           'State-L1':('#E6007E','-'),'State-L1-clipped':('#823DDB','--'),'POT80-Vol':('#F47812','-')}
    labels={'Raw':'Raw forecast','Shift-CP':'Conformal shift','Vol-ERM':'Volatility-scaled ERM',
            'State-L1-clipped':'Same L1 fit · clipped predictor','State-L1':'Selected L1 state correction','POT80-Vol':'Volatility-scaled POT80'}
    for i,kind in enumerate(('normal','t5')):
        for j,truth in enumerate(('none','constant','state')):
            ax=axes[i,j];g=df[(df.module=='garch')&(df.innovation==kind)&(df.alpha==alpha)&(df.truth==truth)]
            for method,(color,ls) in style.items():
                if method=='Raw' and truth=='none':continue
                x=g[g.method==method].sort_values('n_cal');y=x.mean_excess_QS*10000
                assert (y>0).all()
                ax.plot(x.n_cal,y,color=color,linestyle=ls,marker='o',markersize=3.8,linewidth=1.8)
            ax.set_yscale('log');ax.set_xscale('log',base=2);ax.set_xticks(e.SIZES)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            title={'none':'Already correct','constant':'Constant bias','state':'Bias follows log volatility'}[truth]
            ax.set_title(title,loc='left',fontweight='bold')
            if j==0:ax.set_ylabel(f'{"Normal" if kind=="normal" else "Student t(5)"} innovations\nExcess QS × 10⁴ · log scale')
            if i==1:ax.set_xlabel('Calibration observations')
            if truth=='none':ax.text(.04,.05,'Raw = oracle: zero regret',transform=ax.transAxes,fontsize=8.4,color='#536070')
    order=['Raw','Shift-CP','Vol-ERM','POT80-Vol','State-L1','State-L1-clipped']
    handles=[Line2D([],[],color=style[m][0],ls=style[m][1],lw=2,label=labels[m]) for m in order]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False,
               columnspacing=1.7,handlelength=2.7)
    fig.suptitle(f'Correction shape and estimation cost at the {alpha:.0%} tail',x=.075,y=.995,
                 ha='left',fontsize=17,fontweight='bold')
    fig.text(.075,.951,'GARCH with known volatility; 500 paired histories. Lower regret is better. Predictor clipping is a separate sensitivity.',fontsize=9.8)
    fig.subplots_adjust(left=.075,right=.985,top=.87,bottom=.19,wspace=.28,hspace=.37)
    finish(fig,f'correction_shape_{alpha:g}')


if __name__=='__main__':
    data=pd.read_csv(e.OUT/'results/summary.csv')
    dependence(data)
    for alpha in e.ALPHAS:complexity(data,alpha)
    print('Three figures, PNG and SVG; transparent backgrounds and external bottom legends.')
