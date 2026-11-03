"""Save official HS reference editions separately from the immutable trade snapshot."""

import json
from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from oinq.collectors.raw import utc_now, write_json
from oinq.preprocessing.customs import sha256


def main():
    folder = ROOT / 'data/reference/comtrade/hs_20261103'
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for record in manifest['files']:
            if sha256(ROOT / record['path']) != record['sha256']:
                raise ValueError(f"Reference checksum mismatch: {record['edition']}")
        print('Verified existing HS references; snapshot unchanged.')
        return
    records = []
    for edition in ('H2', 'H3', 'H4', 'H5', 'H6'):
        path = folder / f'{edition}.json'
        url = f'https://comtradeapi.un.org/files/v1/app/reference/{edition}.json'
        if not path.exists():
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            document = response.json()
            if not isinstance(document.get('results'), list):
                raise ValueError(f'Invalid reference response: {edition}')
            path.write_bytes(response.content)
        records.append({'edition': edition, 'url': url, 'path': path.relative_to(ROOT).as_posix(),
                        'sha256': sha256(path), 'bytes': path.stat().st_size})
    write_json(manifest_path, {'collected_at': utc_now(), 'files': records})
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
