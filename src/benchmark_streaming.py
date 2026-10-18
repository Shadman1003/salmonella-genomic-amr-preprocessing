"""Benchmark real frozen AST prefixes in isolated child processes.

This measures row canonicalization plus measurement-interval parsing, not the
whole join/group/model pipeline. No records are replicated or synthesized.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


def canonical_row(row):
    from preprocess import canonical, clean_text, measurement
    sign = clean_text(row.get('measurement_sign'))
    sign = '==' if sign == '=' else sign
    out = {name: canonical(row.get(name)) for name in ('antibiotic', 'phenotype')}
    out.update({name: clean_text(row.get(name)) for name in (
        'id', 'biosample_acc', 'target_acc', 'standard', 'platform', 'vendor',
        'reagent', 'bioproject_acc')})
    out['measurement_sign'] = sign
    mic_value = row.get('mic')
    secondary = clean_text(row.get('mic_secondary'))
    if secondary:
        mic_value = f'{clean_text(mic_value)}/{secondary}'
    out['mic_secondary_raw'] = secondary
    out.update(measurement(mic_value, sign, 'mic'))
    out.update(measurement(row.get('disk_diffusion'), sign, 'disk'))
    return out


def worker(path, limit, chunk):
    # Import before timing: initialization remains included in parent wall time
    # and peak RSS, while transform_seconds describes the measured loop.
    from preprocess import canonical, measurement  # noqa: F401
    start = time.perf_counter()
    hashed = hashlib.sha256()
    count = 0
    with path.open(encoding='utf-8') as source:
        lines = (line for line in source if line.strip())
        if limit:
            lines = itertools.islice(lines, limit)
        while True:
            batch = [json.loads(line) for line in itertools.islice(lines, chunk)]
            if not batch:
                break
            for row in batch:
                payload = json.dumps(canonical_row(row), ensure_ascii=False,
                                     sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')
                hashed.update(payload)
                hashed.update(b'\n')
            count += len(batch)
    elapsed = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({'records_processed': count, 'chunk_size': chunk,
                      'requested_prefix_rows': limit or 'full',
                      'canonical_sha256': hashed.hexdigest(),
                      'transform_seconds': elapsed,
                      'rows_per_second': count / elapsed if elapsed else None,
                      'peak_rss_mib': rss / (1024 * 1024 if sys.platform == 'darwin' else 1024)}))


def benchmark(root, chunks):
    raw = root / 'data/raw'
    path = raw / 'ncbi_salmonella_ast.jsonl'
    manifest = json.loads((raw / 'acquisition_manifest.json').read_text())
    source = next(x for x in manifest['downloads'] if x['collection'] == 'ast')
    source_hash = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            source_hash.update(block)
    if source_hash.hexdigest() != source['sha256']:
        raise RuntimeError('AST raw SHA256 differs from frozen manifest')
    runs = []
    for prefix in (1000, 10000, 50000, 0):
        for chunk in chunks:
            start = time.perf_counter()
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--path', str(path), '--limit', str(prefix), '--chunk', str(chunk)]
            try:
                process = subprocess.run(command, capture_output=True, text=True,
                                         check=False, timeout=900)
                if process.returncode:
                    raise RuntimeError(process.stderr[-4000:] or process.stdout[-4000:])
                result = json.loads(process.stdout)
                result.update(status='completed', process_wall_seconds=time.perf_counter() - start)
            except Exception as error:
                result = {'status': 'failed', 'requested_prefix_rows': prefix or 'full',
                          'chunk_size': chunk, 'error': f'{type(error).__name__}: {error}',
                          'process_wall_seconds': time.perf_counter() - start}
            runs.append(result)
            print(json.dumps(result), flush=True)
    consistency = []
    for prefix in (1000, 10000, 50000, 'full'):
        subset = [x for x in runs if x['requested_prefix_rows'] == prefix]
        completed = [x for x in subset if x['status'] == 'completed']
        counts = sorted({x['records_processed'] for x in completed})
        hashes = sorted({x['canonical_sha256'] for x in completed})
        expected_rows = source['records'] if prefix == 'full' else min(prefix, source['records'])
        passed = (len(completed) == len(chunks) and counts == [expected_rows] and len(hashes) == 1)
        consistency.append({'requested_prefix_rows': prefix, 'expected_rows': expected_rows,
                            'observed_rows': counts, 'distinct_canonical_hashes': len(hashes),
                            'all_chunks_same_output': passed})
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Frozen-source AST row canonicalization and MIC/disk interval parsing only; no joins, deduplication, model fitting or predictive-quality comparison.',
              'data_policy': 'Only real ordered source prefixes; no duplication, synthetic expansion or extrapolated timings.',
              'timing_policy': 'One child process per run. process_wall_seconds includes startup/imports; transform_seconds measures JSON loading, canonicalization, parsing and SHA256 hashing. Single run per configuration on a shared workstation; other project work may be active. OS file-cache, concurrent load and run order can affect comparisons.',
              'memory_policy': 'Child process ru_maxrss high-water mark, including Python and pandas/numpy imported by preprocess. Linux reports KiB; converted to MiB.',
              'source_file': source['file'], 'source_sha256': source['sha256'],
              'frozen_source_records': source['records'], 'chunks': chunks,
              'python_executable': sys.executable, 'python_version': sys.version,
              'preprocess_source_sha256': hashlib.sha256((Path(__file__).parent / 'preprocess.py').read_bytes()).hexdigest(),
              'benchmark_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runs': runs, 'chunk_consistency': consistency,
              'all_checks_passed': all(x['all_chunks_same_output'] for x in consistency)}
    destination = root / 'results/streaming_benchmark.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--path', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=5000)
    parser.add_argument('--chunks', type=int, nargs='+', default=[500, 5000, 25000])
    args = parser.parse_args()
    if args.limit < 0 or args.chunk < 1 or any(x < 1 for x in args.chunks):
        parser.error('Limits must be nonnegative and chunk sizes positive')
    if args.worker:
        if not args.path:
            parser.error('--worker requires --path')
        worker(args.path, args.limit, args.chunk)
    else:
        report = benchmark(args.root, args.chunks)
        raise SystemExit(0 if report['all_checks_passed'] else 1)


if __name__ == '__main__':
    main()


