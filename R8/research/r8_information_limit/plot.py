"""Transparent scientific figure from the complete fixed output grid."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/irfa-information-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/irfa-information-cache')
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from run import ROOT, OUT, sha


def main():
    data=pd.read_csv(OUT/'run/finite.csv')
    part=data[(data.alpha==.01)&(data.epsilon==.2)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.hashsalt':'irfa-information-limit'})
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.8))
    colors=['#0069D9','#F28C00','#BF28C9','#009B72']
    for r,color in zip([0.,.25,.5,.75],colors):
        block=part[part.retention==r].sort_values('n')
        label='Independent draws' if r==0 else f'Retention probability {r:g}'
        axes[0].plot(block.n,100*block.best_average_error,marker='o',color=color,label=label,lw=2.4,ms=6)
        axes[1].plot(block.n,block.scalar_bayes_regret_over_alpha,marker='o',color=color,lw=2.4,ms=6)
    axes[0].set_title('A. Choosing raw or a fixed correction',loc='left',fontweight='bold',pad=14)
    axes[1].set_title('B. Allowing any scalar correction',loc='left',fontweight='bold',pad=14)
    axes[0].set_ylabel('Best equal-prior error (%)');axes[0].set_ylim(0,50)
    axes[1].set_ylabel('Lower bound on excess loss / α');axes[1].set_ylim(0,.018)
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([125,250,500,1000],labels=['125','250','500','1,000'])
        ax.set_xlabel('Calibration observations');ax.grid(axis='y',alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.065),
               ncol=2,frameon=False,columnspacing=2.8)
    fig.text(.5,.005,'Specified two-law score experiment • α = 1% • ε = 0.2 • all history observed',ha='center',fontsize=9)
    fig.subplots_adjust(left=.085,right=.98,top=.86,bottom=.31,wspace=.32)
    outputs=[]
    for ext in ['png','svg']:
        path=OUT/('information_limit.'+ext)
        kwargs={'metadata':{'Date':None}} if ext=='svg' else {}
        fig.savefig(path,transparent=True,dpi=210,bbox_inches='tight',pad_inches=.12,**kwargs)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        outputs.append(path)
    plt.close(fig)
    record=dict(producer_sha256=sha(__file__),input_sha256=sha(OUT/'run/finite.csv'),
                outputs={p.name:sha(p) for p in outputs},transparent=True,legend='outside, bottom')
    (OUT/'figures.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
