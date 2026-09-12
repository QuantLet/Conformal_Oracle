"""Validate and index the actual reviewer dispatch/completion ledger."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/team_review_20260910'
OUT = ROOT / 'artifacts/r8_team_review'


def collect(complete=False):
    ledger = json.loads((DOC / 'LEDGER.json').read_text())
    tasks = json.loads((DOC / 'TASKS.json').read_text())
    assert [x['id'] for x in tasks] == list(range(1, 101))
    assert ledger['requested'] == 100
    started, finished = ledger['started'], ledger['completed']
    assert set(finished) <= set(started) <= {str(i) for i in range(1, 101)}
    assert len(set(started.values())) == len(started)
    assert len(set(finished.values())) == len(finished)
    if complete:
        assert len(started) == len(finished) == 100, 'Reviews are incomplete'
    records = []
    lines = ['# 100-agent review: index', '',
             'These are focused AI checks in the same team, not external human endorsements.', '',
             f"Requested: 100; started: {len(started)}; completed: {len(finished)}.", '',
             '| Agent | Question | Report |', '|---|---|---|']
    for task in tasks:
        key = str(task['id'])
        row = dict(task_id=task['id'], agent=started.get(key), question=task['question'])
        if key in finished:
            path = DOC / finished[key]
            data = path.read_bytes()
            assert len(data) > 100, (key, 'Empty report')
            row.update(report=str(path.relative_to(ROOT)),
                       sha256=hashlib.sha256(data).hexdigest(), status='completed')
            link = f'[{path.name}]({path.name})'
        else:
            row['status'] = 'running' if key in started else 'queued'
            link = row['status'].capitalize()
        records.append(row)
        lines.append(f"| {task['id']:03} | {task['question']} | {link} |")
    (DOC / 'INDEX.md').write_text('\n'.join(lines) + '\n')
    result = dict(requested=100, started=len(started), completed=len(finished),
                  externally_independent_human_reviews=0, records=records)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'review_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    result = collect(args.complete)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
