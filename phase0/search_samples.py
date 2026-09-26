"""Fetch public sample provenance only; never uploads local images."""
import concurrent.futures
import json
import pathlib
import urllib.parse
import urllib.request
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
QUERIES = [
    'elderly woman photograph -painting -drawing -artwork',
    'old man glasses photograph -painting -drawing',
    'elderly sleeping -painting -drawing',
    'elderly eating -painting -drawing',
]

def search(pair):
    index, query = pair
    params = dict(action='query', generator='search', gsrsearch=query,
                  gsrnamespace=6, gsrlimit=8, prop='imageinfo',
                  iiprop='url|extmetadata', iiurlwidth=800, format='json')
    url = 'https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(params)
    target = ROOT / 'artifacts' / ('samples-search-%d.json' % index)
    if not target.exists():
        subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '90', url, '-o', str(target)], check=True)
    data = json.loads(target.read_text())
    return [(p['title'], p.get('imageinfo', [{}])[0].get('extmetadata', {}).get('LicenseShortName', {}).get('value')) for p in data.get('query', {}).get('pages', {}).values()]

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for rows in pool.map(search, enumerate(QUERIES)):
            print(json.dumps(rows, ensure_ascii=False), flush=True)
