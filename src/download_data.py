"""Acquire an immutable local NCBI Pathogen Detection metadata/AST snapshot.

No genome reads, assembled sequences, patient identifiers, or credentials are used.
Raw JSONL preserves every returned field, including absent vs present-null fields.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ENDPOINT = 'https://www.ncbi.nlm.nih.gov/pathogens/pathogens-srv/'
SORT = json.dumps([{'property': 'id', 'direction': 'ASC'}], separators=(',', ':'))
COHORT = 'taxgroup_name:"Salmonella enterica"'
QUERIES = {
    'ast': COHORT,
    'isolates': COHORT + ' AND number_drugs_tested:[1 TO *]',
}
SOURCES = [
    'https://www.ncbi.nlm.nih.gov/pathogens/docs/ast/',
    'https://www.ncbi.nlm.nih.gov/pathogens/ast/',
    'https://www.ncbi.nlm.nih.gov/pathogens/pathogens_help/',
    'https://www.ncbi.nlm.nih.gov/pathogens/docs/data_processing/',
    'https://www.ncbi.nlm.nih.gov/pathogens/faq/',
]

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def retrieve(collection, query, start, limit):
    params = dict(action='retrieve', collection=collection, fq=query,
                  start=start, limit=limit, fl='*', sort=SORT)
    url = ENDPOINT + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        'User-Agent': 'InternshipResearchPrime/1.0 reproducible public research',
        'Accept': 'application/json',
    })
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                body = response.read()
            obj = json.loads(body)
            if not obj.get('success'):
                raise RuntimeError(f'NCBI returned unsuccessful response: {obj}')
            data = obj['ngout']['data']
            return data['content'], int(data['totalCount']), {
                'url': url, 'retrieved_at_utc': utcnow(),
                'response_sha256': hashlib.sha256(body).hexdigest(),
                'response_bytes': len(body),
            }
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def acquire(collection, query, raw_dir, page_size):
    output = raw_dir / f'ncbi_salmonella_{collection}.jsonl'
    if output.exists():
        raise FileExistsError(f'Preserving existing snapshot: {output}')
    partial = output.with_suffix('.jsonl.partial')
    _, expected, probe = retrieve(collection, query, 0, 0)
    record_ids = set()
    previous_id = None
    pages = []
    n = 0
    with partial.open('w', encoding='utf-8', newline='\n') as out:
        while n < expected:
            rows, total, audit = retrieve(collection, query, n, page_size)
            if total != expected:
                raise RuntimeError(f'Live database count changed: {expected} -> {total}; preserve partial and restart snapshot')
            if not rows:
                raise RuntimeError(f'Empty page before total at {n}/{expected}')
            for row in rows:
                key = row.get('id')
                if not key or key in record_ids:
                    raise RuntimeError(f'Duplicate or missing API record ID: {key}')
                if previous_id is not None and key <= previous_id:
                    raise RuntimeError('NCBI response not ordered by ID as requested')
                record_ids.add(key)
                previous_id = key
                out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
            n += len(rows)
            audit.update(start=n-len(rows), records=len(rows), first_id=rows[0]['id'], last_id=rows[-1]['id'])
            pages.append(audit)
            print(f'{collection}: {n:,}/{expected:,}', flush=True)
            time.sleep(0.35)
    _, final_total, final_probe = retrieve(collection, query, 0, 0)
    if n != expected or final_total != expected:
        raise RuntimeError(f'Count inconsistent after acquisition: {n}, {expected}, {final_total}')
    partial.replace(output)
    return {
        'collection': collection, 'query': query, 'fields': '*',
        'sort': json.loads(SORT), 'page_size': page_size,
        'records': n, 'distinct_ids': len(record_ids),
        'file': str(output.relative_to(raw_dir.parent.parent)),
        'sha256': sha256(output), 'bytes': output.stat().st_size,
        'initial_probe': probe, 'final_probe': final_probe, 'pages': pages,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--page-size', type=int, default=5000)
    args = parser.parse_args()
    raw_dir = args.root / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / 'acquisition_manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'Preserving existing manifest: {manifest_path}')
    manifest = {
        'dataset': 'NCBI Pathogen Detection Salmonella enterica AST and tested-isolate metadata',
        'started_at_utc': utcnow(), 'endpoint': ENDPOINT,
        'official_documentation': SOURCES,
        'cohort': COHORT,
        'limitations': [
            'Public convenience/surveillance collection; not a representative prevalence sample.',
            'Phenotypes are supplied by BioSample submitters and receive only basic NCBI QC.',
            'API is a live service; stable sorting and equal before/after counts do not guarantee a transactionally consistent snapshot.',
            'Only isolates with submitted AST are acquired; missing AST is not susceptibility.',
        ],
        'downloads': [],
    }
    for collection, query in QUERIES.items():
        result = acquire(collection, query, raw_dir, args.page_size)
        manifest['downloads'].append(result)
        manifest_path.with_suffix('.json.partial').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    manifest['completed_at_utc'] = utcnow()
    manifest_path.with_suffix('.json.partial').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    manifest_path.with_suffix('.json.partial').replace(manifest_path)
    print(json.dumps({d['collection']: {'records': d['records'], 'sha256': d['sha256'], 'bytes': d['bytes']} for d in manifest['downloads']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
