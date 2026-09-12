"""Force a fresh replay using a temporary view without cached replay receipts."""
import json
from pathlib import Path
import sys
import tempfile
import lag_calendar as cal
import replay_calendar


def main():
    original=cal.OUT
    with tempfile.TemporaryDirectory(prefix='irfa-calendar-native-replay-') as tmp:
        view=Path(tmp)
        for asset in sorted((original/'native').iterdir()):
            folder=view/'native'/asset.name;folder.mkdir(parents=True)
            for p in asset.iterdir():
                if p.name!='actual_replay.json':(folder/p.name).symlink_to(p.resolve())
        cal.OUT=view;sys.argv=[sys.argv[0]];replay_calendar.main()
        record=json.loads((view/'actual_full_replay.json').read_text())
        for row in record['rows']:
            row['native_directory']=str((original/'native'/row['asset']).relative_to(cal.PROJECT))
        (cal.OLD/'quality/native_replay_lagllama_primary.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
