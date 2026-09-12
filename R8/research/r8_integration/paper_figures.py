"""Layouts sized for the manuscript's narrow text measure, not a slide deck."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator

BLUE='#0075FF';GREEN='#00A878';PINK='#E6007E';ORANGE='#F47812';PURPLE='#823DDB';GREY='#536070'


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10.5,
        'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#95A1B1',
        'text.color':'#17243B','axes.labelcolor':'#17243B','svg.fonttype':'none'})


def finish(fig):
    for ax in fig.axes:
        ax.patch.set_alpha(0);ax.grid(alpha=.18,linewidth=.6);ax.set_axisbelow(True)
    fig.patch.set_alpha(0)


def dependence(df):
    style();fig,axes=plt.subplots(2,2,figsize=(7,5.9),sharex=True)
    for i,alpha in enumerate([.01,.05]):
        for j,kind in enumerate(['normal','t5']):
            ax=axes[i,j];g=df[(df.module=='ar')&(df.alpha==alpha)&(df.innovation==kind)&(df.truth=='constant')]
            for phi,c in [(0,BLUE),(.5,GREEN),(.8,PINK)]:
                for method,ls,marker in [('Shift-CP','-','o'),('POT80-Shift','--','s')]:
                    x=g[(g.phi==phi)&(g.method==method)].sort_values('n_cal')
                    y=x.difference_vs_Raw.to_numpy()*1e4;se=x.MCSE_vs_Raw.to_numpy()*1e4
                    ax.plot(x.n_cal,y,color=c,ls=ls,marker=marker,ms=3,lw=1.5)
                    ax.fill_between(x.n_cal,y-1.96*se,y+1.96*se,color=c,alpha=.07,lw=0)
            ax.axhline(0,color=GREY,lw=.7)
            ax.set_xscale('log',base=2);ax.set_xticks([125,250,500,1000]);ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_title(f'{"Normal" if kind=="normal" else "Student t(5)"} · {alpha:.0%}',loc='left',weight='bold')
            if j==0:ax.set_ylabel('QS change from raw ×10⁴')
            if i==1:ax.set_xlabel('Calibration observations')
    handles=[Line2D([],[],color=c,lw=2,label=f'AR coefficient {phi:g}') for phi,c in [(0,BLUE),(.5,GREEN),(.8,PINK)]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.061),ncol=3,frameon=False,fontsize=9.5,columnspacing=1.2)
    fig.legend(handles=[Line2D([],[],color=GREY,ls=ls,marker=mk,lw=1.5,label=label)
               for ls,mk,label in [('-','o','Conformal shift'),('--','s','POT80 shift')]],
               loc='lower center',bbox_to_anchor=(.5,.012),ncol=2,frameon=False,fontsize=9.5)
    fig.subplots_adjust(left=.13,right=.98,bottom=.245,top=.94,wspace=.28,hspace=.38)
    finish(fig);return fig


def complexity(df,alpha):
    style();fig,axes=plt.subplots(3,2,figsize=(7,7.4),sharex=True)
    styles={'Raw':(GREY,'--'),'Shift-CP':(BLUE,'-'),'Vol-ERM':(GREEN,'-'),
            'POT80-Vol':(ORANGE,'-'),'State-L1':(PINK,'-'),'State-L1-clipped':(PURPLE,'--')}
    labels={'Raw':'Raw','Shift-CP':'Conformal shift','Vol-ERM':'Volatility-scaled ERM',
            'POT80-Vol':'Volatility-scaled POT80','State-L1':'Selected L1',
            'State-L1-clipped':'L1 with clipped predictor'}
    for i,truth in enumerate(['none','constant','state']):
        for j,kind in enumerate(['normal','t5']):
            ax=axes[i,j];g=df[(df.module=='garch')&(df.alpha==alpha)&(df.truth==truth)&(df.innovation==kind)]
            for method,(c,ls) in styles.items():
                if method=='Raw' and truth=='none':continue
                x=g[g.method==method].sort_values('n_cal')
                ax.plot(x.n_cal,x.mean_excess_QS*1e4,color=c,ls=ls,marker='o',ms=3,lw=1.5)
            ax.set_yscale('log');ax.set_xscale('log',base=2)
            ax.set_xticks([125,250,500,1000]);ax.xaxis.set_major_formatter(ScalarFormatter())
            label={'none':'already correct','constant':'constant bias','state':'state-dependent bias'}[truth]
            ax.set_title(f'{"Normal" if kind=="normal" else "t(5)"} · {label}',loc='left',weight='bold',fontsize=10)
            if j==0:ax.set_ylabel('Excess QS ×10⁴')
            if i==2:ax.set_xlabel('Calibration observations')
            if truth=='none':ax.text(.04,.06,'Raw regret = 0',transform=ax.transAxes,fontsize=9,color=GREY)
    handles=[Line2D([],[],color=c,ls=ls,lw=1.6,label=labels[m]) for m,(c,ls) in styles.items()]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.002),ncol=2,frameon=False,fontsize=9.5,columnspacing=1.3)
    fig.subplots_adjust(left=.12,right=.98,top=.95,bottom=.21,wspace=.29,hspace=.44)
    finish(fig);return fig


def frontier(summary,intervals,sens):
    style();fig,(ax,bx)=plt.subplots(2,1,figsize=(7,6.5))
    full=intervals[(intervals.reference=='Shift-CP')&(intervals.block_calendar_days==60)].set_index('method')
    ex=sens[(sens.sensitivity=='without_crypto')&(sens.block_calendar_days==60)].set_index('method')
    names=['State-L1','POT-Shift','POT-Vol','DtACI-projected-expected','Loss-gate','Past-minimum']
    labels=['Regularised state','POT-Shift','POT-Vol','Projected DtACI','Loss gate','Past-loss minimum']
    for offset,frame,c in [(-.14,full,BLUE),(.14,ex,PINK)]:
        for j,name in enumerate(names):
            row=frame.loc[name];v=row['difference']
            ax.errorbar(v,j+offset,xerr=[[v-row.simultaneous_lower],[row.simultaneous_upper-v]],fmt='o',color=c,ms=4,capsize=2,lw=1.5)
    ax.axvline(0,color=GREY,lw=.7,ls='--');ax.set_yticks(range(6),labels);ax.invert_yaxis()
    ax.set_xlabel('QS difference from Shift-CP ×10⁴')
    ax.set_title('A. The asset mix changes the evidence',loc='left',weight='bold')
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
    fig.subplots_adjust(left=.255,right=.975,top=.945,bottom=.14,hspace=.69)
    finish(fig);return fig
