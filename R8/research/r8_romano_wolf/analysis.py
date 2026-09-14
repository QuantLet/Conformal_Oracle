"""Romano-Wolf stepdown and model confidence set on the stored eight-comparison draws (PROTOCOL.md)."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
REV = PROJECT / 'artifacts/r8_referee_revision/results'
TEN = PROJECT / 'artifacts/r8_ten_comparators/results'
OUT = PROJECT / 'artifacts/r8_romano_wolf'
FAMILY = ['Raw', 'Vol-ERM', 'State-L1', 'POT-Shift', 'POT-Vol', 'DtACI-projected-expected', 'Loss-gate', 'Past-minimum']
SIX = ['State-L1', 'POT-Shift', 'POT-Vol', 'DtACI-projected-expected', 'Loss-gate', 'Past-minimum']
REF = 'Shift-CP'
BLOCKS = (20, 60)
LEVEL = .05


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(folder, block):
    z = np.load(folder / f'bootstrap_{block}.npz', allow_pickle=True)
    methods = [str(m) for m in z['methods']]
    return pd.DataFrame(z['draws'] * 1e4, columns=methods), pd.Series(z['point'] * 1e4, index=methods)


def contrasts(draws, point, family):
    theta = point[family] - point[REF]
    d = draws[family].to_numpy() - draws[[REF]].to_numpy()
    sd = d.std(axis=0, ddof=1)
    tstar = (d - theta.to_numpy()) / sd
    return theta, sd, tstar


def single_step(theta, sd, tstar):
    crit = float(np.quantile(np.abs(tstar).max(axis=1), 1 - LEVEL))
    return crit, theta - crit * sd, theta + crit * sd


def stepdown(theta, sd, tstar):
    t = (theta / sd).to_numpy()
    names = list(theta.index)
    remaining = list(range(len(names)))
    step, rejected_at, crits = 0, {}, []
    while remaining:
        step += 1
        c = float(np.quantile(np.abs(tstar[:, remaining]).max(axis=1), 1 - LEVEL))
        crits.append(dict(step=step, critical_value=c, remaining=len(remaining)))
        new = [j for j in remaining if abs(t[j]) > c]
        for j in new:
            rejected_at[j] = step
        if not new:
            break
        remaining = [j for j in remaining if j not in new]
    return t, rejected_at, crits


def mcs(draws, point, methods):
    """Model confidence set with the T_max statistic and its elimination rule (Hansen, Lunde and Nason, 2011):
    t_i = (mean_j d_ij)/sd, eliminate argmax t_i while the bootstrap p-value of max_i t_i is below LEVEL."""
    kept = list(methods)
    trace = []
    while len(kept) > 1:
        p = point[kept].to_numpy(); D = draws[kept].to_numpy()
        dev = p - (p.sum() - p) / (len(kept) - 1)                       # d_i. = mean over j of (p_i - p_j)
        ddev = D - (D.sum(axis=1, keepdims=True) - D) / (len(kept) - 1)
        sd = (ddev - dev).std(axis=0, ddof=1)
        t = dev / sd
        tstar = ((ddev - dev) / sd).max(axis=1)
        stat = float(t.max()); crit = float(np.quantile(tstar, 1 - LEVEL)); pval = float((tstar >= stat).mean())
        worst = int(np.argmax(t))
        trace.append(dict(size=len(kept), T_max=stat, critical_value=crit, p_value=pval, eliminated=kept[worst] if pval < LEVEL else '', members=';'.join(kept)))
        if pval < LEVEL:
            kept.pop(worst)
        else:
            break
    return kept, trace


def run():
    t0 = time.monotonic()
    published = pd.read_csv(REV / 'intervals.csv')
    rows, crit_rows, mcs_rows, final = [], [], [], {}
    for block in BLOCKS:
        draws, point = load(REV, block)
        theta, sd, tstar = contrasts(draws, point, FAMILY)
        crit8, lo, hi = single_step(theta, sd, tstar)
        for name in FAMILY:
            r = published[(published.method == name) & (published.reference == REF) & (published.block_calendar_days == block)].iloc[0]
            assert abs(r.difference - theta[name]) < 1e-9 and abs(r.simultaneous_lower - lo[name]) < 1e-9 and abs(r.simultaneous_upper - hi[name]) < 1e-9, (block, name)
        draws6, point6 = load(TEN, block)
        theta6, sd6, tstar6 = contrasts(draws6, point6, SIX)
        crit6, lo6, hi6 = single_step(theta6, sd6, tstar6)
        six_published = pd.read_csv(TEN / 'intervals.csv')
        for name in SIX:
            r = six_published[(six_published.method == name) & (six_published.reference == REF) & (six_published.block_calendar_days == block)].iloc[0]
            assert abs(r.simultaneous_lower - lo6[name]) < 1e-9 and abs(r.simultaneous_upper - hi6[name]) < 1e-9, (block, name, 'six')
        # The Raw row under the six-member critical value (Raw was not a member of that family).
        raw_lo6, raw_hi6 = theta['Raw'] - crit6 * sd[0], theta['Raw'] + crit6 * sd[0]
        t, rejected_at, crits = stepdown(theta, sd, tstar)
        for k, name in enumerate(FAMILY):
            rows.append(dict(block_calendar_days=block, method=name, difference=theta[name], bootstrap_sd=sd[k], t=t[k],
                             single_step_lower=lo[name], single_step_upper=hi[name],
                             rejected_single_step=bool(abs(t[k]) > crit8), stepdown_step=rejected_at.get(k, 0),
                             rejected_stepdown=k in rejected_at, favours=('method' if theta[name] < 0 else 'Shift-CP')))
        crit_rows.append(dict(block_calendar_days=block, eight_member_critical=crit8, six_member_critical=crit6,
                              raw_band_lower_eight=lo['Raw'], raw_band_upper_eight=hi['Raw'],
                              raw_band_lower_six_critical=raw_lo6, raw_band_upper_six_critical=raw_hi6,
                              stepdown_steps=json.dumps(crits)))
        kept, trace = mcs(draws, point, [REF] + FAMILY)
        for tr in trace:
            mcs_rows.append(dict(block_calendar_days=block, **tr))
        final[block] = kept
    return dict(results=pd.DataFrame(rows), critical=pd.DataFrame(crit_rows), mcs=pd.DataFrame(mcs_rows), final=final, elapsed=time.monotonic() - t0)


def write(res):
    OUT.mkdir(parents=True, exist_ok=True)
    res['results'].to_csv(OUT / 'results.csv', index=False)
    res['critical'].to_csv(OUT / 'critical_values.csv', index=False)
    res['mcs'].to_csv(OUT / 'mcs.csv', index=False)
    (OUT / 'run.json').write_text(json.dumps(dict(protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')), producer_sha256=sha(__file__),
        inputs={f'artifacts/r8_referee_revision/results/bootstrap_{b}.npz': sha(REV / f'bootstrap_{b}.npz') for b in BLOCKS} |
               {f'artifacts/r8_ten_comparators/results/bootstrap_{b}.npz': sha(TEN / f'bootstrap_{b}.npz') for b in BLOCKS},
        level=LEVEL, elapsed_seconds=res['elapsed'], python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        mcs_final={str(k): v for k, v in res['final'].items()}), indent=2) + '\n')
    r = res['results']; c = res['critical']
    lines = ['# Stepdown and model confidence set: results', '', 'Exploratory; protocol fixed before computation. Level 5%, two-sided, 999 stored draws.', '']
    for block in BLOCKS:
        cc = c[c.block_calendar_days == block].iloc[0]
        lines += [f'## {block}-day blocks', '', f'Single-step critical values: eight-member {cc.eight_member_critical:.6f}, six-member {cc.six_member_critical:.6f}. Stepdown steps: {cc.stepdown_steps}', '',
                  '| method | difference | sd | t | rejected single-step | stepdown step |', '|---|---|---|---|---|---|']
        for _, x in r[r.block_calendar_days == block].iterrows():
            lines.append(f'| {x.method} | {x.difference:.4f} | {x.bootstrap_sd:.4f} | {x.t:.3f} | {x.rejected_single_step} | {x.stepdown_step or "none"} |')
        lines += ['', f'Model confidence set at 95%: {res["final"][block]}', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    res = run()
    if a.check:
        for name, fn in [('results', 'results'), ('critical', 'critical_values'), ('mcs', 'mcs')]:
            saved = pd.read_csv(OUT / f'{fn}.csv'); fresh = res[name]
            assert list(saved.columns) == list(fresh.columns), name
            for col in saved.columns:
                if saved[col].dtype.kind in 'fi':
                    assert np.allclose(saved[col].to_numpy(float), fresh[col].to_numpy(float), rtol=0, atol=1e-10, equal_nan=True), (name, col)
                else:
                    assert (saved[col].fillna('').astype(str).to_numpy() == fresh[col].fillna('').astype(str).to_numpy()).all(), (name, col)
        print(f'CHECK PASSED: results.csv, critical_values.csv, mcs.csv reproduced; published bands reproduced; elapsed {res["elapsed"]:.1f}s')
    else:
        write(res); print('written', OUT)


if __name__ == '__main__':
    main()
