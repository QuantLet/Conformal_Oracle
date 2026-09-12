"""Transparent research figures with legends outside the plotting area."""
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/irfa-shape-cost-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import engine as e


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#23262B','text.color':'#23262B',
        'axes.edgecolor':'#69717B','xtick.color':'#414853','ytick.color':'#414853',
        'svg.fonttype':'none','savefig.transparent':True})
    plots=e.ROOT/'artifacts/r8_shape_cost/figures'
    plots.mkdir(exist_ok=True)
    data=pd.read_csv(e.OUT/'contrasts.csv')
    data=data[data.family=='primary']
    colors={.5:'#E64B35',2.:'#6633FF'}
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.4),sharey=True)
    for ax,kind,title in zip(axes,['normal','t5'],['Normal innovations','Student-t(5) innovations']):
        for factor in [.5,2.]:
            d=data[(data.kind==kind)&(data.h_factor==factor)].sort_values('n')
            x=d.n.to_numpy()
            y=d.n_scaled_delta.to_numpy()
            low=x*d.simultaneous_lower.to_numpy();high=x*d.simultaneous_upper.to_numpy()
            ax.errorbar(x,y,yerr=np.vstack([y-low,high-y]),color=colors[factor],
                        marker='o',markersize=5.5,linewidth=1.8,capsize=4)
            ax.plot(x,d.n_scaled_prediction,color=colors[factor],linestyle='--',linewidth=1.7)
        ax.axhline(0,color='#7A8088',linewidth=.8)
        ax.set_xscale('log');ax.set_xticks([250,1000,4000],labels=['250','1,000','4,000'])
        ax.set_xlim(200,5000);ax.set_title(title,pad=12,weight='medium')
        ax.set_xlabel('Calibration observations')
        ax.grid(axis='y',color='#D5DAE0',alpha=.55,linewidth=.5)
        ax.patch.set_alpha(0)
    axes[0].set_ylabel(r'$n\times$ expected loss difference'+'\nVol-ERM minus Shift-ERM')
    handles=[];labels=[]
    for factor,name in [(.5,'Smaller error'),(2.,'Larger error')]:
        handles.extend([Line2D([0],[0],color=colors[factor],marker='o',linewidth=1.8),
                        Line2D([0],[0],color=colors[factor],linestyle='--',linewidth=1.7)])
        labels.extend([name+': estimated',name+': first-order prediction'])
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.02),
               ncol=2,frameon=False,fontsize=9,handlelength=3)
    fig.subplots_adjust(left=.11,right=.985,top=.9,bottom=.27,wspace=.12)
    fig.patch.set_alpha(0)
    for suffix in ['png','svg']:
        fig.savefig(plots/f'shape_cost.{suffix}',dpi=220,bbox_inches='tight',transparent=True)
    plt.close(fig)
    (plots/'FIGURE_NOTES.md').write_text(
        '# Figure scope\n\nEach solid point averages 5,000 independent calibration histories. '
        'Error bars are approximate simultaneous 95% Monte Carlo intervals across all '
        '12 primary cells. Dashed lines show the first-order prediction, not a fitted curve. '
        'Negative values favour Vol-ERM; positive values favour Shift-ERM. The n=1,000 '
        'Normal cells are the two registered primary conditions. The other cells are '
        'fixed sensitivities. Both coefficients remain fixed over a contiguous horizon. '
        'Backgrounds are transparent and the legend is outside, below the plots.\n'
    )
    record={'files':{str(p.relative_to(e.ROOT)):e.sha(p) for p in sorted(plots.iterdir())
                    if p.is_file() and p.name!='receipt.json'},
            'input_sha256':e.sha(e.OUT/'contrasts.csv'),'producer_sha256':e.sha(__file__)}
    (plots/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
