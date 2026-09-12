"""Generate a bounded scientific assessment from verified study outputs."""
import json
import pandas as pd
from run import ROOT, OUT, sha


def main():
    validation=json.loads((OUT/'independent_validation.json').read_text())
    assert validation['status']=='passed'
    assert validation['proof_sha256']==sha(ROOT/'research/r8_information_limit/PROOF.md')
    finite=pd.read_csv(OUT/'run/finite.csv')
    lengths=pd.read_csv(OUT/'run/lengths.csv')
    contiguous=pd.read_csv(OUT/'contiguous.csv')
    rows=finite[(finite.alpha==.01)&(finite.epsilon==.2)&(finite.n==250)].sort_values('retention')
    lines=['# Câtă informație trebuie pentru a recunoaște o corecție utilă?', '',
           '10 septembrie 2026. Dezvoltare teoretică și calcule deterministe. Articolul R8 validat este păstrat.', '',
           '## Rezultatul obținut', '',
           'Am construit un experiment statistic în care putem calcula exact cât de bine poate decide orice regulă între prognoza brută și o corecție dată dinainte. Regula primește întregul istoric și cunoaște cele două legi posibile. Nu îi rezervăm artificial o validare scurtă și nu îi cerem să estimeze shiftul. Dificultatea rămasă este, prin construcție, una de informație.', '',
           'O a doua demonstrație tratează orice corecție scalară, cu o limită inferioară asupra regretului față de cuantila populațională optimă. În regimul în care nivelul cozii scade și numărul așteptat de observații în coadă rămâne finit, eroarea optimă de selecție are o limită strict pozitivă. Regretul scalar împărțit la nivelul cozii rămâne de asemenea pozitiv; regretul absolut poate tinde la zero.', '',
           'Aceasta completează asimptotica la nivel fix din articol. Nu o contrazice: familia nouă schimbă nivelul cozii și densitatea locală odată cu dimensiunea eșantionului.', '',
           '## Exemplu fixat înainte de calcul', '',
           f"La nivelul de 1%, cele două legi au probabilități brute de depășire de {100*rows.iloc[0].theta0:.4f}% și {100*rows.iloc[0].theta1:.4f}%. Scorurile sunt un amestec uniform pe două intervale; corecția oferită este jumătate din lățimea intervalului pozitiv. Ea crește pierderea în prima lege și o reduce în a doua. Unitățile sunt cele ale modelului construit, nu praguri recomandate pentru active.", '',
           '| Persistența prin repetare | Eroarea medie minimă, n=250 | Observații pentru eroare medie ≤10% |',
           '| --- | ---: | ---: |']
    for row in rows.itertuples():
        limit=lengths[(lengths.alpha==.01)&(lengths.epsilon==.2)&(lengths.retention==row.retention)&(lengths.target_error==.1)].iloc[0]
        label='0 — observații independente' if row.retention==0 else f'{row.retention:g}'
        lines.append(f'| {label} | {100*row.best_average_error:.2f}% | {int(limit.minimum_n):,} |')
    lines += ['', 'Eroarea este media cu ponderi egale pe cele două legi, nu o probabilitate estimată pentru panelul financiar. Lungimile sunt necesarul de discriminare în acest experiment favorabil, nu recomandări de ferestre și nici garanții suficiente de eroare uniformă.', '',
              '## Legătura cu evaluarea contiguă', '',
              'Am derivat apoi o extensie pentru un shift ținut fix pe următorul bloc de test, fără gap. Nucleul Markov dă o formulă exactă a termenului de frontieră. Pentru corecții nenegative și limitate, numai partea rară pozitivă a scorurilor contribuie la termenul neliniar al pierderii; penalizarea de transfer este de ordinul α/H.', '',
              'În regimul rar, aceasta dispare față de regretul de ordinul α pentru orice H care crește, inclusiv proporția 70/30 din articol. Rezultatul pentru toate corecțiile scalare cere aici intervalul [0,1]; proiecția nu poate elimina această restricție la frontiera dependentă. Rezultatul binar păstrează cele două acțiuni originale.', '',
              f"Pe toate cele 64 de configurații, limita contiguă este pozitivă în {validation['contiguous']['positive_binary_bounds']} cazuri pentru alegerea binară și în {validation['contiguous']['positive_scalar_bounds']} pentru corecțiile din [0,1]. Limitele negative sunt păstrate. Aceste numărători nu sunt teste statistice sau probabilități de succes.", '',
              '## Verificări', '',
              f"- {validation['enumeration']['cases']} comparații cu aritmetică rațională, prin enumerarea a {validation['enumeration']['enumerated_states']:,} istorii de reîmprospătare și semne; eroare numerică maximă {validation['enumeration']['maximum_error']:.2e}.",
              f"- {validation['integration']['risk_checks']} integrări ale pierderii și {validation['integration']['oracle_checks']} verificări ale optimului; abatere maximă {validation['integration']['maximum_error']:.2e}.",
              f"- {validation['contiguous']['markov_transfer_cases']} verificări ale transferului prin matricea de tranziție; abatere maximă {validation['contiguous']['maximum_transfer_error']:.2e}.",
              '- Reproducere exactă într-un proces nou, 32 de lungimi minime verificate împreună cu predecesoarea fiecăreia și verificarea limitei Poisson.',
              f"- Toate cele {validation['validation']['protected_files']} fișiere canonice, PDF-uri și verificări precedente sunt neschimbate. Nu am generat traiectorii sau prognoze financiare noi.", '',
              '## Noutate și decizie editorială', '',
              'Principiile Le Cam, raportul de verosimilitate și limita Bayes–minimax sunt clasice. Riscul suprapotrivirii la selecție este discutat de [Cawley și Talbot](https://www.jmlr.org/beta/papers/v11/cawley10a.html), iar [Liang, Zhu și Barber](https://arxiv.org/abs/2408.07066) studiază validitatea și eficiența după selecția modelelor conformale. [Ma, Verchand și Samworth](https://arxiv.org/abs/2406.13447) tratează instrumente generale pentru limite inferioare. Nu revendicăm inventarea lor.', '',
              'Contribuția posibilă este conexiunea explicită dintre pierderea recalibrării, informația din coadă, persistență și evaluarea contiguă. Aceasta depășește constatarea că un anumit gate funcționează slab. Totuși, exemplul are repetări exacte și legi cunoscute; nu dovedește că performanța gate-ului financiar este aproape de limita optimă.', '',
              'Verdict: rezultatul trece verificarea matematică și numerică internă și merită evaluat pentru integrare compactă. Nu justifică singur eticheta „top paper”. Înainte de a înlocui revendicarea centrală, trebuie stabilit cât din această conexiune este nou față de literatura completă și ce concluzie financiară poate susține fără a generaliza modelul construit.', '',
              '## Integrare propusă, fără extindere mecanică a manuscrisului', '',
              'Păstrarea unei singure întrebări: când avem suficientă informație pentru a învăța o corecție care merită aplicată? Rezultatul la nivel fix descrie costul, noua limită rară descrie dificultatea identificării, iar studiul fracției arată costul unei implementări fezabile. Teorema de coverage rămâne distinctă.', '',
              'O integrare bună ar înlocui explicații repetate din secțiunea despre selecție și ar concentra în supliment demonstrația. Nu recomand adăugarea tuturor celor 64 de rânduri sau promovarea regulii Bayes cu legi cunoscute drept estimator financiar. Panelul principal și testul French își păstrează concluziile și statutul.', '',
              'Fișierele de lucru sunt `research/r8_information_limit/PROOF.md`, `PROTOCOL.md`, `CONTIGUOUS_ADDENDUM.md` și rezultatele din `artifacts/r8_information_limit`. Figura `information_limit.png` are fundal transparent și legendă în exterior, jos. Arhiva separată și verificarea ei sunt documentate în `package.json` după finalizare.']
    path=ROOT/'docs/IRFA_INFORMATION_LIMIT_RESULTS.md'
    path.write_text('\n'.join(lines)+'\n')
    result=dict(producer_sha256=sha(__file__),inputs={str(p.relative_to(ROOT)):sha(p) for p in
                [OUT/'run/finite.csv',OUT/'run/lengths.csv',OUT/'contiguous.csv',OUT/'independent_validation.json']},
                report_sha256=sha(path))
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
