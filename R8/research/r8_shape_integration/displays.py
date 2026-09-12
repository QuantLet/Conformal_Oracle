"""Build the R8 shape-cost macros and vector figure from immutable results.

--check rederives macros using Decimal CSV parsing and reproduces the vector
figure in a temporary directory. It never runs a study or changes old files.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/irfa-shape-integration-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_shape_integration'
SIM = ROOT / 'artifacts/r8_shape_cost/simulation'
FIN = ROOT / 'artifacts/r8_shape_cost/financial'
MACROS = ROOT / 'source/sections_r8/numbers_shape.tex'
FIGURE = ROOT / 'source/figures/fig_shape_cost.pdf'
RECEIPT = OUT / 'display_validation.json'
INPUTS = [SIM/'contrasts.csv', SIM/'findings.json', FIN/'summary.csv',
          FIN/'support.csv', FIN/'inference_status.json', FIN/'receipt.json',
          FIN/'validation.json', ROOT/'artifacts/r8_shape_cost/validation/simulation_validation.json',
          ROOT/'research/r8_shape_cost/plot.py']
COLORS = {.5: '#F2384A', 2.: '#6633FF'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value):
    return f'{int(value):,}'.replace(',', '{,}')


def inputs():
    results = pd.read_csv(SIM/'contrasts.csv')
    findings = json.loads((SIM/'findings.json').read_text())
    financial = pd.read_csv(FIN/'summary.csv')
    support = pd.read_csv(FIN/'support.csv')
    inference = json.loads((FIN/'inference_status.json').read_text())
    receipt = json.loads((FIN/'receipt.json').read_text())
    assert sha(SIM/'contrasts.csv') == findings['outputs']['contrasts.csv']
    for name in ['summary.csv', 'support.csv', 'inference_status.json']:
        assert sha(FIN/name) == receipt['outputs'][name], name
    assert json.loads((FIN/'validation.json').read_text())['status'] == 'passed'
    numerical = json.loads((ROOT/'artifacts/r8_shape_cost/validation/simulation_validation.json').read_text())
    assert numerical['status'] == numerical['output_replay']['status'] == 'passed'
    assert numerical['output_replay']['findings_receipt_sha256'] == sha(SIM/'findings.json')
    assert findings['primary_sign_crossing'] == 'supported'
    assert {x['inference'] for x in inference} == {'aborted_empty_state'}
    assert all(x['no_draws_redrawn'] and x['no_pairs_dropped'] for x in inference)
    return results, findings, financial, support, inference


def macros_from_data(data):
    results, findings, financial, support, inference = data
    primary = results[(results.family == 'primary') & results.primary].set_index('h_factor')
    family = results[results.family == 'primary']
    error = 100 * (family.mean_delta-family.leading_prediction).abs() / family.leading_prediction.abs()
    values = {'nShapeHistories': integer(findings['independent_histories']),
              'nShapeFamily': integer(findings['families']['primary']['contrasts']),
              'nShapeSensitivityFamily': integer(findings['families']['sensitivity']['contrasts']),
              'nShapeDraws': integer(findings['bootstrap_draws']),
              'nShapePrimaryN': integer(primary.iloc[0]['n']),
              'nShapePrimaryH': integer(primary.iloc[0]['horizon']),
              'nShapeDiscrepancyMin': f'{error.min():.1f}',
              'nShapeDiscrepancyMax': f'{error.max():.1f}'}
    for factor, prefix in [(.5,'nShapeLow'), (2.,'nShapeHigh')]:
        row = primary.loc[factor]
        for field, suffix in [('mean_delta','Delta'), ('simultaneous_lower','Lo'), ('simultaneous_upper','Hi')]:
            values[prefix+suffix] = f'{row[field]*1e4:.4f}'
    matched = financial[financial.population == 'matched161'].set_index('state')
    assert support.matched.sum() == matched.loc['all','pairs']
    values.update(nShapeFinMatched=integer(support.matched.sum()),
                  nShapeFinFull=integer(len(support)),
                  nShapeFinUnmatched=integer((~support.matched).sum()))
    for state, label in [('high','High'),('other','Other'),('all','All')]:
        row = matched.loc[state]
        values['nShapeFin'+label+'Delta'] = f'{row.delta_normalized:.8f}'
        values['nShapeFin'+label+'ShiftPi'] = f'{100*row.shift_pi:.3f}'
        values['nShapeFin'+label+'VolPi'] = f'{100*row.vol_pi:.3f}'
        values['nShapeFin'+label+'Rows'] = integer(row.pair_dates)
    values['nShapeFinInteraction'] = f'{matched.loc["high","delta_normalized"]-matched.loc["other","delta_normalized"]:.8f}'
    by_length = {r['block_calendar_days']:r for r in inference}
    values['nShapeFinAbortTwenty'] = integer(by_length[20]['draws_with_empty_cells'])
    values['nShapeFinAbortSixty'] = integer(by_length[60]['draws_with_empty_cells'])
    assert by_length[20]['total_draws'] == by_length[60]['total_draws']
    values['nShapeFinDraws'] = integer(by_length[20]['total_draws'])
    return values


def independent_macro_values():
    # An independent textual reconstruction: csv/Decimal, not pandas floats.
    def read(path):
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))
    rows = read(SIM/'contrasts.csv')
    main = [r for r in rows if r['family']=='primary']
    selected = {Decimal(r['h_factor']):r for r in main if r['primary']=='True'}
    findings = json.loads((SIM/'findings.json').read_text())
    expected = {'nShapeHistories':integer(findings['independent_histories']),
                'nShapeFamily':integer(len(main)),
                'nShapeSensitivityFamily':integer(sum(r['family']=='sensitivity' for r in rows)),
                'nShapeDraws':integer(findings['bootstrap_draws']),
                'nShapePrimaryN':integer(selected[Decimal('.5')]['n']),
                'nShapePrimaryH':integer(selected[Decimal('.5')]['horizon'])}
    errors = [Decimal(100)*abs(Decimal(r['mean_delta'])-Decimal(r['leading_prediction']))/abs(Decimal(r['leading_prediction'])) for r in main]
    expected['nShapeDiscrepancyMin'] = f'{min(errors):.1f}'
    expected['nShapeDiscrepancyMax'] = f'{max(errors):.1f}'
    for factor, prefix in [(Decimal('.5'),'nShapeLow'),(Decimal(2),'nShapeHigh')]:
        for field, suffix in [('mean_delta','Delta'),('simultaneous_lower','Lo'),('simultaneous_upper','Hi')]:
            expected[prefix+suffix] = f'{Decimal(selected[factor][field])*10000:.4f}'
    financial = {r['state']:r for r in read(FIN/'summary.csv') if r['population']=='matched161'}
    support = read(FIN/'support.csv')
    matched = sum(r['matched']=='True' for r in support)
    expected.update(nShapeFinMatched=integer(matched), nShapeFinFull=integer(len(support)),
                    nShapeFinUnmatched=integer(len(support)-matched))
    for state, tag in [('high','High'),('other','Other'),('all','All')]:
        row = financial[state]
        assert int(row['pairs']) == matched
        expected['nShapeFin'+tag+'Delta'] = f'{Decimal(row["delta_normalized"]):.8f}'
        expected['nShapeFin'+tag+'ShiftPi'] = f'{Decimal(row["shift_pi"])*100:.3f}'
        expected['nShapeFin'+tag+'VolPi'] = f'{Decimal(row["vol_pi"])*100:.3f}'
        expected['nShapeFin'+tag+'Rows'] = integer(row['pair_dates'])
    expected['nShapeFinInteraction'] = f'{Decimal(financial["high"]["delta_normalized"])-Decimal(financial["other"]["delta_normalized"]):.8f}'
    statuses = {r['block_calendar_days']:r for r in json.loads((FIN/'inference_status.json').read_text())}
    expected['nShapeFinAbortTwenty'] = integer(statuses[20]['draws_with_empty_cells'])
    expected['nShapeFinAbortSixty'] = integer(statuses[60]['draws_with_empty_cells'])
    expected['nShapeFinDraws'] = integer(statuses[20]['total_draws'])
    return expected


def macro_text(values):
    return '% Generated by research/r8_shape_integration/displays.py; do not edit.\n' + ''.join(
        '\\newcommand{\\'+key+'}{'+values[key]+'}\n' for key in sorted(values))


def make_figure(results, target):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#23262B','text.color':'#23262B',
        'axes.edgecolor':'#69717B','xtick.color':'#414853','ytick.color':'#414853',
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.transparent':True})
    data = results[results.family=='primary']
    fig, axes = plt.subplots(1,2,figsize=(10.4,4.4),sharey=True)
    series = []
    for ax, kind, title in zip(axes, ['normal','t5'], ['Normal innovations','Student-t(5) innovations']):
        for factor in [.5,2.]:
            values = data[(data.kind==kind)&(data.h_factor==factor)].sort_values('n')
            x = values.n.to_numpy()
            y = values.n_scaled_delta.to_numpy()
            low = x*values.simultaneous_lower.to_numpy()
            high = x*values.simultaneous_upper.to_numpy()
            prediction = values.n_scaled_prediction.to_numpy()
            ax.errorbar(x,y,yerr=np.vstack([y-low,high-y]),color=COLORS[factor],
                        marker='o',markersize=5.5,linewidth=1.8,capsize=4)
            ax.plot(x,prediction,color=COLORS[factor],linestyle='--',linewidth=1.7)
            series.append(dict(kind=kind,h_factor=factor,n=x.tolist(),observed=y.tolist(),
                               lower=low.tolist(),upper=high.tolist(),prediction=prediction.tolist()))
        ax.axhline(0,color='#7A8088',linewidth=.8)
        ax.set_xscale('log')
        ax.set_xticks([250,1000,4000],labels=['250','1,000','4,000'])
        ax.set_xlim(200,5000)
        ax.set_title(title,pad=12,weight='medium')
        ax.set_xlabel('Calibration observations')
        ax.grid(axis='y',color='#D5DAE0',alpha=.55,linewidth=.5)
        ax.patch.set_alpha(0)
    axes[0].set_ylabel(r'$n\times$ expected loss difference'+'\nVol-ERM minus Shift-ERM')
    handles, labels = [], []
    for factor, name in [(.5,'Smaller error'),(2.,'Larger error')]:
        handles.extend([Line2D([0],[0],color=COLORS[factor],marker='o',linewidth=1.8),
                        Line2D([0],[0],color=COLORS[factor],linestyle='--',linewidth=1.7)])
        labels.extend([name+': simulated', name+': first-order prediction'])
    legend = fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.014),
                        ncol=2,frameon=False,fontsize=9,handlelength=3)
    fig.subplots_adjust(left=.11,right=.985,top=.90,bottom=.28,wspace=.12)
    fig.patch.set_alpha(0)
    fig.canvas.draw()
    extent = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    assert extent.y1 < min(ax.get_position().y0 for ax in axes)
    assert fig.get_facecolor()[3] == 0 and all(ax.get_facecolor()[3]==0 for ax in axes)
    style = dict(transparent_figure=True,transparent_axes=True,legend_outside_bottom=True,
                 legend_bbox=list(extent.bounds),colors={str(k):v for k,v in COLORS.items()},
                 panels=2,points=12,confidence_family=12,pdf_fonttype=42,
                 units='n times expected loss; Vol-ERM minus Shift-ERM',series=series)
    fig.savefig(target,format='pdf',bbox_inches='tight',transparent=True,
                metadata={'CreationDate':None,'ModDate':None,'Title':'Shape and coefficient-estimation cost',
                          'Subject':'Immutable 12-cell simultaneous Monte Carlo comparison'})
    plt.close(fig)
    reader = PdfReader(target)
    assert len(reader.pages)==1
    resources = reader.pages[0]['/Resources'].get_object()
    objects = resources.get('/XObject',{})
    if hasattr(objects,'get_object'):
        objects=objects.get_object()
    images = [k for k,v in objects.items() if v.get_object().get('/Subtype')=='/Image']
    assert not images, 'Figure must remain vector artwork'
    style['raster_images']=len(images)
    return style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    data = inputs()
    values = macros_from_data(data)
    independent = independent_macro_values()
    assert values == independent, (values,independent)
    expected = macro_text(values)
    bindings = {str(p.relative_to(ROOT)):sha(p) for p in INPUTS}
    if args.check:
        assert MACROS.read_text()==macro_text(independent), 'Generated macro text differs from independent CSV/Decimal replay'
        record = json.loads(RECEIPT.read_text())
        assert record['producer_sha256']==sha(__file__)
        assert record['inputs']==bindings
        assert record['outputs']=={str(p.relative_to(ROOT)):sha(p) for p in [MACROS,FIGURE]}
        with tempfile.TemporaryDirectory(prefix='irfa-shape-display-') as folder:
            rebuilt=Path(folder)/'figure.pdf'
            style=make_figure(data[0],rebuilt)
            assert sha(rebuilt)==sha(FIGURE), 'Vector figure differs from exact result replay'
        assert style==record['figure']
        check=dict(status='passed',macros=len(values),independent_macro_replay='csv and Decimal',
                   exact_vector_pdf_replay=True,display_receipt_sha256=sha(RECEIPT),
                   producer_sha256=sha(__file__),inputs=bindings,figure=style)
        (OUT/'display_check.json').write_text(json.dumps(check,indent=2)+'\n')
        print(json.dumps({k:v for k,v in check.items() if k not in ['inputs','figure']},indent=2))
    else:
        MACROS.write_text(expected)
        style=make_figure(data[0],FIGURE)
        record=dict(status='passed',producer_sha256=sha(__file__),inputs=bindings,macros=values,
                    figure=style,outputs={str(p.relative_to(ROOT)):sha(p) for p in [MACROS,FIGURE]},
                    scientific_results_recomputed=False)
        RECEIPT.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'status':'passed','macros':len(values),'outputs':record['outputs']},indent=2))


if __name__=='__main__':
    main()
