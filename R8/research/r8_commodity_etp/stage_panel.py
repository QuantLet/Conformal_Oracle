"""Stage the fixed replacement panel; preserve every original input archive."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from panel_scope import PROJECT, OLD, NEW, ART, ROOT, MODELS, ASSETS, REPLACEMENTS, source


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    inputs, copies = {}, {}
    def read(p):
        inputs[str(p.relative_to(PROJECT))] = sha(p)
        return pd.read_csv(p)
    def copy(p, name):
        inputs[str(p.relative_to(PROJECT))] = sha(p)
        dest = ROOT/name; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): assert sha(dest) == sha(p), ('changed staged input', name)
        else: shutil.copy2(p, dest)
        copies[str(name)] = {'source':str(p.relative_to(PROJECT)), 'sha256':sha(dest)}
    for name in ['validation.json', 'decision_validation.json', 'classical_fresh_replay.json']:
        p = NEW/'quality'/name; inputs[str(p.relative_to(PROJECT))] = sha(p)
        assert json.loads(p.read_text())['status'] == 'passed'
    for asset in ASSETS:
        src = source(asset)
        copy(src/'data/returns'/f'{asset}.csv', Path('data/returns')/f'{asset}.csv')
        for model, (folder, suffix) in MODELS.items():
            name = f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet'
            copy(src/'data'/folder/name, Path('data')/folder/name)
            for ext in ['json', 'parquet']:
                name = f'{model}__{asset}.{ext}'
                copy(src/'posthoc'/name, Path('posthoc')/name)
        for model in ['CAViaR-SAV','CAViaR-AS','GAS-t']:
            name = f'{asset}_{model}.parquet'
            copy(src/'data/dynamic'/name, Path('data/dynamic')/name)
        copy(src/'evt_fhs'/f'{asset}.parquet', Path('evt_fhs')/f'{asset}.parquet')
    for name in ['posthoc.csv', 'indication.csv']:
        old, new = read(OLD/'results'/name), read(NEW/'results'/name)
        old = old[old.model.isin(MODELS) & ~old.asset.isin(REPLACEMENTS)]
        assert set(new.asset) == set(REPLACEMENTS.values()) and set(new.model) == set(MODELS)
        result = pd.concat([old,new], ignore_index=True)
        result = result.sort_values(['model','asset','method' if name=='posthoc.csv' else 'alpha'])
        (ROOT/'results').mkdir(exist_ok=True)
        result.to_csv(ROOT/'results'/name, index=False)
    inventory = read(OLD/'quality/asset_inventory.csv')
    inventory = inventory[~inventory.asset.isin(REPLACEMENTS)]
    support = read(NEW/'quality/support.csv'); rows = []
    for item in support.itertuples():
        rows.append(dict(asset=item.asset, ticker=item.asset, instrument={
            'USO':'United States Oil Fund', 'GLD':'SPDR Gold Shares', 'UNG':'United States Natural Gas Fund'}[item.asset],
            currency='USD', exchange='NYSE Arca', n_returns=item.returns, first_date=item.first_return,
            last_date=item.last_date, n_exclusions_after_initial=0, n_missing_prices=0,
            sha256=item.input_sha256, n_closed_market_rows=0, overlap_comparison='different instrument; no splice'))
    inventory = pd.concat([inventory,pd.DataFrame(rows)], ignore_index=True).sort_values('asset')
    assert set(inventory.asset) == set(ASSETS)
    (ROOT/'quality').mkdir(exist_ok=True)
    inventory.to_csv(ROOT/'quality/asset_inventory.csv',index=False)
    # These simulations and two-asset sampling diagnostics do not involve
    # the replaced commodities. The calendar audit is retained as historical.
    for name in ['results/monte_carlo/grid.csv','results/predictive_sampling/draws.csv',
                 'results/predictive_sampling/propagation.csv','results/review/complexity_mc.csv',
                 'results/review/calendar_scores.csv']:
        copy(OLD/name, name)
    record = dict(producer_sha256=sha(__file__), scope_sha256=sha(Path(__file__).with_name('panel_scope.py')),
                  replacements=REPLACEMENTS, models=list(MODELS), assets=ASSETS, inputs=inputs, copies=copies,
                  historical_return_and_forecast_assets_unchanged=21, replacement_assets_reestimated=3)
    ART.mkdir(exist_ok=True)
    (ART/'stage.json').write_text(json.dumps(record,indent=2)+'\n')
    (ROOT/'primary_ready.json').write_text(json.dumps(dict(
        stage_sha256=sha(ART/'stage.json'), all_calculable_returns_retained=True,
        data_selection_before_new_forecast_evaluation=True, replacements=REPLACEMENTS),indent=2)+'\n')
    print('Staged 24 assets, 168 pairs:', len(copies), 'byte-identical input copies')


if __name__ == '__main__': main()
