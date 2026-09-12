"""Transparent research figure, using only verified result matrices."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/irfa-decision-mpl')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/r8_decision'
R=OUT/'results'


def main():
    summary=pd.read_csv(R/'summary.csv').set_index('method')
    intervals=pd.read_csv(R/'intervals.csv')
    sensitivity=pd.read_csv(R/'sensitivity_intervals.csv')
    full=intervals[(intervals.reference=='Shift-CP')&(intervals.block_calendar_days==60)].set_index('method')
    ex=sensitivity[(sensitivity.sensitivity=='without_crypto')&(sensitivity.block_calendar_days==60)].set_index('method')
    names=['State-L1','POT-Shift','POT-Vol','DtACI-projected-expected','Loss-gate','Past-minimum']
    labels=['Regularised state','POT: constant shift','POT: volatility scale','Projected DtACI*','Loss-based gate','Past-loss minimum']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#243044','text.color':'#243044',
                         'xtick.color':'#465267','ytick.color':'#465267','svg.fonttype':'none'})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13.7,6.7),gridspec_kw={'width_ratios':[1.18,1]})
    blue='#1261F0';purple='#CC28C7';orange='#F07412';teal='#009C89'
    for offset,frame,color in [(-.13,full,blue),(.13,ex,purple)]:
        for j,name in enumerate(names):
            row=frame.loc[name];center=row['difference']
            lo=row.simultaneous_lower;hi=row.simultaneous_upper
            ax.errorbar(center,j+offset,xerr=[[center-lo],[hi-center]],fmt='o',color=color,
                        capsize=3,markersize=5,linewidth=1.8)
    ax.axvline(0,color='#718096',lw=1,linestyle='--')
    ax.set_yticks(range(len(names)),labels);ax.invert_yaxis()
    ax.set_xlabel('Change in Quantile Score relative to Shift-CP (×10⁴)\nNegative values favour the candidate')
    ax.set_title('A. Gains depend on the asset mix',loc='left',fontweight='bold',pad=17)
    ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
    mapping={'Raw':('Raw',blue,(9,4)),'Shift-CP':('Shift-CP',blue,(8,-18)),
             'Vol-ERM':('Vol-ERM',blue,(-3,13)),'Rolling500':('Rolling 500',blue,(8,-14)),
             'Loss-gate':('Loss-based gate',orange,(9,3)),'Past-minimum':('Past-loss minimum',teal,(-90,10))}
    for name,(label,color,offset) in mapping.items():
        row=summary.loc[name];x=row.QS_x10000;y=row.violation_rate*100
        bx.scatter(x,y,c=color,s=58,zorder=4)
        bx.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=9,color=color)
    bx.axhline(1,color='#718096',lw=1,linestyle='--')
    bx.text(5.28,1.013,'Nominal 1%',ha='right',va='bottom',fontsize=8,color='#718096')
    bx.set_xlim(4.99,5.33);bx.set_ylim(.90,1.94)
    bx.set_xlabel('Mean Quantile Score (×10⁴; lower is better)')
    bx.set_ylabel('Mean violation rate (%)')
    bx.set_title('B. Cautious selection leaves undercoverage',loc='left',fontweight='bold',pad=17)
    bx.grid(alpha=.13);bx.set_axisbelow(True)
    fig.suptitle('Stronger comparators change the recalibration argument',fontsize=16,fontweight='bold',y=.97)
    handles=[Line2D([0],[0],color=blue,marker='o',lw=1.8,label='All 24 assets'),
             Line2D([0],[0],color=purple,marker='o',lw=1.8,label='Without Bitcoin and Ethereum')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False,fontsize=10)
    fig.text(.5,.035,'Panel A: simultaneous 95% bands for six comparisons; common 60-calendar-day blocks. '
             'Intervals condition on fitted forecasts and selections.',ha='center',fontsize=8,color='#536176')
    fig.text(.5,.012,'* DtACI uses projected levels and finite empirical thresholds; loss is averaged over expert randomisation. '
             'This adaptation does not inherit the original theorem.',ha='center',fontsize=8,color='#536176')
    fig.subplots_adjust(left=.15,right=.98,bottom=.25,top=.84,wspace=.30)
    for suffix in ['png','svg']:
        fig.savefig(OUT/f'comparison_frontier.{suffix}',dpi=190,transparent=True)
    plt.close(fig)


if __name__=='__main__':main()
