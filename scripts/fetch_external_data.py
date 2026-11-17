"""Download pinned Metrica sample CSVs locally; do not redistribute the data."""
import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path('data/external/metrica-sample-data')
DEFAULT_REVISION = 'e706dd506b360d69d9d123d5b8026e7294b13996'
SOURCE = 'https://github.com/metrica-sports/sample-data'
LICENSE_NOTE = ('README requests attribution and responsible use; standard redistribution '
                'and derived checkpoint terms are unspecified')


def select_revision(explicit=None, manifest_path=None):
    """Explicit revision wins, then local manifest, then verified fixed default."""
    revision = explicit
    if revision is None and manifest_path is not None and manifest_path.exists():
        revision = json.loads(manifest_path.read_text(encoding='utf-8'))['revision']
    revision = revision or DEFAULT_REVISION
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('revision must be a full lowercase Git commit SHA')
    return revision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', help='full source commit SHA (default: manifest or pinned revision)')
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.data_dir
    revision = select_revision(args.revision, root / 'manifest.json')

    def fetch(item):
        game, suffix = item
        stem = f'Sample_Game_{game}'
        name = f'{stem}_{suffix}.csv'
        output = root / stem / name
        output.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/metrica-sports/sample-data/{revision}/data/{stem}/{name}'
        with urlopen(url, timeout=180) as response:
            content = response.read()
        digest = hashlib.sha256(content).hexdigest()
        if output.exists():
            if hashlib.sha256(output.read_bytes()).hexdigest() != digest:
                raise ValueError(f'existing file differs from source: {output}')
        else:
            output.write_bytes(content)
        return {'game':game, 'file':name, 'bytes':len(content), 'sha256':digest, 'url':url}

    items = [(game, suffix) for game in (1,2) for suffix in
             ('RawTrackingData_Home_Team','RawTrackingData_Away_Team','RawEventsData')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        entries = list(pool.map(fetch, items))
    manifest = {'source':SOURCE, 'revision':revision, 'license':LICENSE_NOTE,
                'default_proxy_config':{'fps':25.0, 'boundary_margin_frames':5, 'max_event_frames':100},
                'files':entries}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Fetched and verified {len(entries)} files for 2 matches; source revision {revision}.')


if __name__ == '__main__':
    main()
