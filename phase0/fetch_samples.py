"""Download selected Commons photos with machine-readable attribution."""
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
SELECT = [
    ('Elderly Gambian woman face portrait.jpg', ['portrait']),
    ('Elderly man in Rhodes, Greece (black and white).jpg', ['portrait']),
    ('Old man in a tweed jacket and glasses (1527580).jpg', ['glasses']),
    ('Old man in a tweed jacket and glasses (1527585).jpg', ['glasses']),
    ('Elderly Lahu woman.jpg', ['natural_scene']),
    ('Sleeping elderly man sitting on a park bench under an umbrella, Alameda Afonso Henriques, Lisbon, Portugal julesvernex2.jpg', ['sleeping_candidate']),
    ('This elderly gentleman, with a wheelchair loaded with bags of valuables, has found a warm discrete corner to sleep though a cold day (50757201653).jpg', ['sleeping_candidate']),
    ('Elderly Couple Eating.jpg', ['eating_candidate', 'multiple_faces']),
    ('Elderly woman with curly blonde hair smiling while eating a donut with pink frosting.jpg', ['eating_candidate']),
    ('Washington, D.C. Elderly couple eating dinner at their home on Lamont Street, N.W. (LOC).jpg', ['eating_candidate', 'multiple_faces']),
    ('Elderly woman, with book - DPLA - 0839b2b0f4e85bb95a037b7caf519c8c.jpg', ['portrait']),
    ('Elderly woman - DPLA - 1d68e8f0fc229e79cdf6b5fb05dca63a.jpg', ['portrait']),
]

def main():
    pages = {}
    for path in [ROOT/'artifacts/commons-search.json'] + sorted((ROOT/'artifacts').glob('samples-search-*.json')):
        for page in json.loads(path.read_text()).get('query', {}).get('pages', {}).values():
            pages[page['title'].removeprefix('File:')] = page
    rows = []
    for index, (name, tags) in enumerate(SELECT, 1):
        info = pages[name]['imageinfo'][0]
        meta = info['extmetadata']
        license_name = meta.get('LicenseShortName', {}).get('value', '')
        if license_name not in ['CC0', 'Public domain', 'CC BY 2.0', 'CC BY 3.0', 'CC BY-SA 2.0', 'CC BY-SA 4.0']:
            raise ValueError('Unreviewed license: '+license_name)
        rows.append(dict(id='s%02d'%index, title=name, tags=tags, source_page=info['descriptionurl'],
                         download_url=info.get('thumburl', info['url']), license=license_name,
                         metadata=meta, path='data/originals/s%02d.jpg'%index,
                         transformed=False, source_group='s%02d'%index))
    (ROOT/'data/originals').mkdir(exist_ok=True, parents=True)
    def fetch(row):
        path = ROOT/row['path']
        subprocess.run(['curl','-fsSL','--retry','3','--max-time','120',row['download_url'],'-o',str(path)],check=True)
        row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(row['id'], path.stat().st_size, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, rows))
    (ROOT/'data/sources.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
