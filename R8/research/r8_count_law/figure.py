"""Transparent explanatory figure for the exact witness, outside the manuscript."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/irfa-count-mpl')
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/r8_count_law'
record=json.loads((OUT/'exact_witness.json').read_text())
values=[record['iid_loss_change_decimal']*1e7,record['renewal_loss_change_decimal']*1e7]
colours=['#E6007E','#006BFF']
with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'irfa-count-law',
                    'axes.spines.top':False,'axes.spines.right':False}):
    fig,ax=plt.subplots(figsize=(8.7,5.6))
    ax.bar([0,1],values,color=colours,width=.5)
    ax.axhline(0,color='#283445',lw=.9)
    ax.set_xticks([0,1],['Independent scores','Pairwise independent\nrenewal scores'])
    ax.set_ylabel('Expected loss change after correction (×10⁻⁷)')
    ax.set_ylim(-1.65,1.2)
    ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    for x,value in enumerate(values):
        ax.text(x,value+(.08 if value>0 else -.08),f'{value:+.4f}',ha='center',
                va='bottom' if value>0 else 'top',weight='bold')
    fig.suptitle('Same pairwise diagnostics, different correction costs',fontsize=16,weight='bold',x=.105,ha='left',y=.97)
    fig.text(.105,.9,'Same marginal distribution · 1% tail · 125 calibration scores · same conformal rank',fontsize=10)
    fig.legend(handles=[Patch(color=colours[0],label='Correction increases loss'),
                        Patch(color=colours[1],label='Correction lowers loss')],loc='lower center',
               bbox_to_anchor=(.52,.06),ncol=2,frameon=False)
    fig.text(.52,.012,'Exact mathematical witness near break-even; the loss effects are small.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.105,right=.98,top=.82,bottom=.26)
    fig.savefig(OUT/'exact_witness.png',transparent=True,dpi=200,bbox_inches='tight')
    fig.savefig(OUT/'exact_witness.svg',transparent=True,bbox_inches='tight',metadata={'Date':None})
    plt.close(fig)
result={'inputs':{'exact_witness.json':hashlib.sha256((OUT/'exact_witness.json').read_bytes()).hexdigest()},
        'outputs':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['exact_witness.png','exact_witness.svg']},
        'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'figure.json').write_text(json.dumps(result,indent=2)+'\n')
