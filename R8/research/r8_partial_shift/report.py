"""Generate the research report directly from validated, complete outputs."""
from pathlib import Path
import json
import pandas as pd
from run import ROOT, OUT, sha


def main():
    validation = json.loads((OUT / 'independent_validation.json').read_text())
    assert validation['status'] == 'passed'
    summary = pd.read_csv(OUT / 'run/summary.csv')
    dec = pd.read_csv(OUT / 'run/decomposition.csv')
    choices = pd.read_parquet(OUT / 'run/decisions.parquet')
    initial = pd.read_parquet(OUT / 'initial_numeric_ties/run/decisions.parquet')
    methods = ['Raw','Full-CP','Half-Full','Inner-CP','Half-Inner','Selected-Inner','Oracle-Grid','Oracle-Continuous']
    overview = summary.groupby(['alpha','method'], sort=True)[['expected_loss','expected_violation']].mean().reset_index()
    overview.to_csv(OUT / 'overview.csv', index=False)
    text = ['# Corecția parțială: beneficiul disponibil și costul selecției', '',
            '10 septembrie 2026. Studiu de dezvoltare pe istorii deja salvate; articolul și suplimentul R8 sunt neschimbate.', '',
            '## Verdict', '',
            'Corecția parțială poate reduce costul unei corecții integrale. Alegerea fracției pe o validare scurtă consumă însă o parte din acest avantaj și nu justifică, în această etapă, promovarea regulii ca metodă financiară nouă. Rezultatul util pentru lucrare este separarea explicită a beneficiului corecției, costului estimării și regretului selecției.', '',
            '## Design', '',
            'Protocolul fixează 144 de configurații: două niveluri de coadă, patru lungimi de calibrare, margini Normal/t(5), control AR și GARCH, cu distorsiunile existente. Sunt 500 de istorii per configurație, dar numai 1.500 de istorii latente independente; reutilizarea lor permite comparații pereche. Nu sunt 72.000 de experimente independente.', '',
            'Regula fezabilă estimează shiftul pe primele m=max(100,floor(0.7n)) observații și alege fracția 0, 0.25, 0.5, 0.75 sau 1 prin pierderea observată pe restul calibrării. Ambele valori rămân apoi fixe. Reperul oracol utilizează legea de evaluare și este separat de selecția fezabilă. La n=125 rămân numai 25 de observații de validare.', '',
            'Evaluarea primară integrează pierderea pe o observație marginală independentă pentru AR și pe stările de volatilitate independente, deja salvate, pentru GARCH. Sensibilitatea contiguă păstrează aceiași parametri și integrează viitorul Normal-AR pe H=floor(3n/7) date. Aceasta nu transformă fracția selectată într-un estimator cu garanție conformală.', '',
            '## Rezultate complete pe fiecare nivel', '',
            'Tabelele următoare folosesc ponderi egale pentru cele 72 de configurații ale fiecărui nivel. Sunt medii descriptive ale grilei artificiale, nu rezultatele panelului financiar. Pierderea pinball este înmulțită cu 10.000; mai mic înseamnă mai bine. Frecvențele sunt probabilități de depășire așteptate, exprimate procentual.', '']
    findings = {}
    for alpha in (.01,.05):
        a = overview[overview.alpha == alpha].set_index('method')
        selected = summary[(summary.alpha == alpha) & (summary.method == 'Selected-Inner')]
        component = dec[dec.alpha == alpha]
        ratio = component.selection_regret.mean() / component.oracle_shrinkage_gain.mean()
        findings[str(alpha)] = dict(selected_better_than_raw=int((selected['difference_vs_Raw']<0).sum()),
            selected_better_than_full=int((selected['difference_vs_Full-CP']<0).sum()),
            selected_better_than_half_inner=int((selected['difference_vs_Half-Inner']<0).sum()),
            configurations=72,selection_regret_to_oracle_gain=float(ratio))
        text += [f'### Coada de {100*alpha:g}%', '', '| Metodă | Pierdere ×10.000 | Depășiri (%) |', '| --- | ---: | ---: |']
        text += [f'| {m} | {a.loc[m,"expected_loss"]*1e4:.5f} | {a.loc[m,"expected_violation"]*100:.5f} |' for m in methods]
        f = findings[str(alpha)]
        text += ['', f'Selected-Inner are pierdere medie mai mică decât Raw în {f["selected_better_than_raw"]}/72 configurații, decât Full-CP în {f["selected_better_than_full"]}/72 și decât Half-Inner în {f["selected_better_than_half_inner"]}/72. Aceste numărători sunt descriptive, nu teste independente.', '',
                 f'Raportul dintre regretul mediu al selecției și câștigul mediu disponibil față de corecția integrală pe același bloc interior este {ratio*100:.2f}%. Este un raport de medii pe grila fixată; nu o constantă universală.', '']
    text += ['## Ce izolează descompunerea', '',
             'Pentru fiecare istorie, diferența de pierdere a regulii fezabile față de Raw este exact:', '',
             '    − pierderea eliminabilă printr-un shift constant',
             '    + costul estimării shiftului interior',
             '    − câștigul disponibil prin alegerea oracol a fracției',
             '    + regretul alegerii fracției din validare.', '',
             'Separăm și efectul rezervării datelor pentru validare: pierderea Inner-CP minus Full-CP. Acesta include atât schimbarea numărului de observații, cât și schimbarea rangului conformal. Creșterea costului între n=125 și n=250 la 1% nu trebuie atribuită unei creșteri generale a varianței: m trece de la 100 la 175, iar rangurile sunt 100 și 175. Cel din urmă folosește maximul la un nivel efectiv mai conservator.', '',
             'În cazul GARCH cu distorsiune dependentă de stare, reperul corecției constante este minimul riscului aceleiași mixturi de stări de test. Distanța rămasă până la cuantila condițională oracol este raportată separat; nu este numită eroare de estimare a shiftului.', '',
             'Corecția fixă pe jumătate este un comparator informativ deoarece evită rezervarea unei validări și alegerea unei intensități. Mediile sale favorabile în acest studiu nu justifică impunerea universală a fracției 0.5.', '',
             'Figuri: `artifacts/r8_partial_shift/selection_cost.png` și `partial_frontier.png`, cu fundal transparent și legende în exterior, jos. Toate configurațiile și erorile Monte Carlo pereche sunt în CSV-urile de sinteză; intervalele punctuale rămân descriptive.', '',
             '## Corectitudinea implementării și reproducerea', '',
             f'- Toate cele {validation["independent_rank_and_selection_checks"]:,} alegeri au fost reconstruite independent. Abaterea maximă între cele două formule de pierdere pe validare este {validation["independent_validation_loss_error"]:.3g}.',
             f'- {validation["exact_rational_tie_comparisons"]} comparații ambigue în virgulă mobilă sunt rezolvate independent prin aritmetică rațională. Obiectivul pinball poate fi exact plat când dimensiunea validării înmulțită cu nivelul țintă este un întreg.',
             f'- Prima implementare a încălcat convenția pentru egalități în {int((initial.selected_fraction != choices.selected_fraction).sum()):,} dintre cele 72.000 de alegeri. Versiunea inițială este arhivată și marcată ca înlocuită. Corecția implementează regula din protocol, fără modificarea grilei sau a criteriului după rezultate.',
             f'- Integrarea independentă a pierderilor Normal și t(5) are abatere maximă {validation["quadrature_error"]:.3g}; gradientul riscului în optimul constant este cel mult {validation["scalar_optimum_gradient_error"]:.3g}.',
             '- Toate cele 144.000 de valori Raw/Full-CP reproduc rezultatele vechi în toleranța declarată. Cele 24.000 de diferențe Full-CP contigue coincid cu implementarea independentă anterioară în toleranță.',
             '- Descompunerea exactă, ordinea reperelor oracol, invarianta față de viitor, controlul care introduce intenționat informație viitoare și cazurile de egalitate trec.',
             '- Toate rezultatele se reproduc exact într-un proces nou. Cele 53 de fișiere protejate, inclusiv sursele canonice, PDF-urile și verificările precedente, sunt neschimbate.', '',
             '## Implicația pentru articol', '',
             'Studiul izolează selecția într-o familie scalară, fără coeficienți de stare sau extrapolare. Astfel, costul identificat nu poate fi atribuit complexității unei regresii flexibile. Demonstrația teoretică motivează comparația, dar formula oracol nu este prezentată ca o regulă estimabilă gratuit.', '',
             'Regula testată nu îndeplinește suficient de bine criteriul de dezvoltare externă stabilit: rămâne mai slabă în medie decât corecția fixă pe jumătate și poate păstra subprotecția. Nu este adăugată ca o nouă metodă principală și nu declanșează o căutare de grile mai favorabile pe aceleași rezultate. Panelul financiar, testul French, textul articolului și PDF-urile rămân neschimbate.', '',
             'Dacă rezultatul este integrat, rolul lui este să susțină mecanismul costului selecției. Formulare de lucru pentru o inserție scurtă, încă neaplicată:', '',
             '> Choosing the strength of a scalar correction also incurs estimation cost. In a fixed development experiment, past-loss selection over five fractions improves on full correction in most configurations at the 1% tail, but often fails to improve the raw forecast. An exact loss decomposition separates the gain available to an oracle from the regret of estimating that choice. The comparison with a fixed half correction shows that flexibility in deployment requires enough validation information even when the correction itself has only one parameter.', '',
             'Această formulare necesită trimiterea la specificația și tabelul studiului dacă este introdusă în manuscris. Nu este o revendicare de prioritate pentru shrinkage sau selecția prin validare. Rezultatele French deja inspectate nu pot deveni confirmare neatinsă pentru o regulă proiectată acum.', '',
             '## Fișiere și arhivă', '',
             '`research/r8_partial_shift/README.md` conține pașii de reproducere. `artifacts/r8_partial_shift/independent_validation.json` este verificarea numerică. Pachetul separat `release/R8_20260910_partial_shift_study.zip` conține codul, intrările, rezultatele, prima execuție înlocuită și verificările. Finalizarea și verificarea exclusiv din arhivă sunt consemnate în `artifacts/r8_partial_shift/package.json`. Pachetul nu înlocuiește arhiva completă a calculelor financiare R8.', '']
    report = ROOT / 'docs/IRFA_PARTIAL_SHIFT_RESULTS.md'
    report.write_text('\n'.join(text))
    (OUT / 'findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    inputs = [OUT/'independent_validation.json',OUT/'run/summary.csv',OUT/'run/decomposition.csv',
              OUT/'run/decisions.parquet',OUT/'initial_numeric_ties/run/decisions.parquet']
    record = dict(producer_sha256=sha(__file__),inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
                  outputs={str(p.relative_to(ROOT)):sha(p) for p in [report,OUT/'overview.csv',OUT/'findings.json']})
    (OUT/'report.json').write_text(json.dumps(record,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    main()
