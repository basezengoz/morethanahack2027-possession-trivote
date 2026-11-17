"""Fetch selected public IDSSE matches into the private run directory."""
import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
import time
from pathlib import Path

ARTICLE = 'https://api.figshare.com/v2/articles/28196177'


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = Path('data/runs/idsse'); root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARTICLE, timeout=60) as response: files = json.load(response)['files']
    tasks = []
    for game in args.matches:
        for kind, marker in (('info','matchinformation'), ('positions','positions_raw_observed')):
            matches = [f for f in files if game in f['name'] and marker in f['name']]
            if len(matches) != 1: raise ValueError(f'expected one {kind} file for {game}')
            tasks.append((matches[0], root / (game+'_'+kind+'.xml')))

    def download(task):
        metadata, target = task
        if target.exists():
            if target.stat().st_size != metadata['size'] or digest(target, 'md5') != metadata['supplied_md5']:
                raise ValueError(f'existing file differs: {target}')
        else:
            tmp = target.with_suffix('.download')
            for attempt in range(6):
                offset = tmp.stat().st_size if tmp.exists() else 0
                if offset > metadata['size']: raise ValueError(f'partial file too large: {tmp}')
                if offset == metadata['size']: break
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                request = urllib.request.Request(metadata['download_url'], headers=headers)
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if offset and (response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                            raise ValueError('server did not honor resume offset')
                        with tmp.open('ab' if offset else 'xb') as out:
                            while part := response.read(1024*1024): out.write(part)
                    break
                except (OSError, TimeoutError) as exc:
                    print(json.dumps({'retry':str(target), 'attempt':attempt+1, 'error':str(exc)}), flush=True)
                    if attempt == 5: raise
                    time.sleep(min(2**attempt, 10))
            if tmp.stat().st_size != metadata['size'] or digest(tmp, 'md5') != metadata['supplied_md5']:
                raise ValueError(f'download integrity failure: {tmp}')
            tmp.replace(target)
        result = {'path':str(target), 'size':target.stat().st_size, 'sha256':digest(target,'sha256')}
        print(json.dumps(result), flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(download, tasks))
    print(json.dumps({'downloaded':results}), flush=True)


if __name__ == '__main__': main()
