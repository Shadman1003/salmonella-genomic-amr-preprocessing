"""Validate frozen-source integrity, hand-calculated semantics and output contracts.

Model feature-matrix leakage, fitted preprocessing and split-disjointness checks
belong to the modeling audit. Passing this script does not establish those claims.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import contextlib
import io
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import pandas as pd
from preprocess import canonical, clean_text, date_interval, digest, json_rows, measurement, preprocess


class Checks:
    def __init__(self):
        self.results = []
    def run(self, name, function):
        try:
            details = function()
            self.results.append({'check': name, 'passed': True, 'details': details})
        except Exception as error:
            self.results.append({'check': name, 'passed': False,
                                 'error': f'{type(error).__name__}: {error}'})
    @staticmethod
    def require(condition, message):
        if not condition:
            raise AssertionError(message)


def fixture_measurements():
    q = Checks.require
    a = measurement('4', '<=', 'mic')
    q(a['mic_1_lower'] == 0 and a['mic_1_upper'] == 4 and
      not a['mic_1_lower_closed'] and a['mic_1_upper_closed'] and a['mic_censored'],
      'MIC <=4 must denote (0,4], not exact 4')
    a = measurement('8', '>', 'mic')
    q(a['mic_1_lower'] == 8 and a['mic_1_upper'] is None and
      not a['mic_1_lower_closed'], 'MIC >8 must retain an open lower bound')
    a = measurement('0.25', '==', 'mic')
    q(a['mic_1_lower'] == a['mic_1_upper'] == .25 and
      a['mic_1_lower_closed'] and a['mic_1_upper_closed'] and not a['mic_censored'],
      'Exact MIC must be a closed singleton')
    a = measurement('32/4', '==', 'mic')
    q(a['mic_components'] == 2 and a['mic_1_lower'] == 32 and a['mic_2_lower'] == 4,
      'Combination 32/4 must stay two concentrations, not quotient 8')
    a = measurement('32/4', '<=', 'mic')
    q(a['mic_1_lower'] == 0 and a['mic_1_upper'] == 32 and
      a['mic_2_lower'] == a['mic_2_upper'] == 4 and
      a['mic_secondary_semantics'] == 'reported_secondary_concentration_no_independent_comparator',
      'Assay comparator must not be invented independently for the secondary concentration')
    a = measurement('20', '==', 'disk')
    q(a['disk_1_lower'] == a['disk_1_upper'] == 20 and 'mic_1_lower' not in a,
      'Disk diffusion must retain its own measurement namespace')
    q(measurement('5', '==', 'disk')['disk_status'] == 'outside_submission_range',
      '5mm is outside documented disk submission range')
    q(measurement('0', '==', 'mic')['mic_status'] == 'outside_submission_range',
      'Zero MIC must not be accepted as an exact concentration')
    q(measurement('32/4/2', '==', 'mic')['mic_status'] == 'unparsed_combination',
      'Three-component input must be explicitly flagged')
    q(measurement('4', '?', 'mic')['mic_status'] == 'unknown_sign',
      'Unknown comparator must not imply equality')
    q(measurement(None, '==', 'mic')['mic_status'] == 'structural_missing',
      'Missing MIC must remain missing')
    return {'hand_calculated_cases': 11,
            'semantics': 'Censor bounds, combination components and MIC/disk namespaces checked independently.'}


def fixture_dates_and_text():
    q = Checks.require
    expected = [('2019', 'year', 2019, 2019, 2019),
                ('2019-05', 'month', 2019, 2019, 2019),
                ('2020-02-29', 'day', 2020, 2020, 2020),
                ('2018/2020', 'range', 2018, 2020, None),
                ('2020-02-30', 'invalid', None, None, None),
                ('2019-99/2020-99', 'invalid', None, None, None),
                ('2020/2018', 'invalid', None, None, None),
                ('2020-05/2020-04', 'invalid', None, None, None),
                ('05/06/2020', 'unparsed', None, None, None),
                (None, 'missing', None, None, None),
                ('2030', 'out_of_scope', None, None, None)]
    for raw, precision, lower, upper, year in expected:
        parsed = date_interval(raw)
        q((parsed['date_precision'], parsed['year_lower'], parsed['year_upper'], parsed['collection_year']) ==
          (precision, lower, upper, year), f'Unexpected date parse: {raw!r}: {parsed}')
    q(canonical('  ReSistant\u00a0') == 'resistant', 'Unicode whitespace/case normalization failed')
    q(clean_text('Not Provided') == '', 'Missing sentinel normalization failed')
    return {'date_cases': len(expected), 'text_cases': 2,
            'scope': 'Accepted date cases tested; not an exhaustive date grammar verification.'}


def fixture_pipeline(root):
    q = Checks.require
    fixture_parent = root / 'work'
    fixture_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='preprocess_validation_', dir=fixture_parent) as folder:
        test_root = Path(folder)
        raw = test_root / 'data/raw'
        raw.mkdir(parents=True)
        base = {'bioproject_acc': 'PRJNA_TEST', 'collection_date': '2019', 'species_taxid': 28901,
                'asm_stats_contig_n50': 100, 'asm_stats_length_bp': 1000,
                'asm_stats_n_contig': 5}
        odd_tokens = ['aac(6\')-Ib-cr=COMPLETE', 'blaCTX-M-15=COMPLETE', 'gene/with,comma=PARTIAL']
        isolates = [dict(base, id='i1', biosample_acc='B1', target_acc='T1',
                         amrfinderplus_applied=1, AMR_genotypes=odd_tokens, number_amr_genes=3),
                    dict(base, id='i2', biosample_acc='B2', target_acc='T2',
                         amrfinderplus_applied=1, AMR_genotypes=['<NONE>'], number_amr_genes=0),
                    dict(base, id='i3', biosample_acc='B3', target_acc='T3',
                         amrfinderplus_applied=0, AMR_genotypes=[], number_amr_genes=0)]
        def assay(i, bio, target, label, **extra):
            return dict(id=i, biosample_acc=bio, target_acc=target,
                        antibiotic=' Ampicillin ', phenotype=label,
                        measurement_sign='==', mic=4, standard='CLSI', **extra)
        assays = [assay('a1', 'B1', 'T1', ' Resistant '),
                  assay('a2', 'B1', 'T1', 'resistant'),
                  assay('a3', 'B2', 'T2', 'susceptible'),
                  assay('a4', 'B2', 'T2', 'resistant'),
                  assay('a5', 'B3', 'T3', None),
                  assay('a6', 'B4', 'T4', 'susceptible')]
        # API supplies combination drugs through a separate mic_secondary field.
        for entry in assays[:2]:
            entry.update(mic=32, mic_secondary=4, measurement_sign='<=')
        files = []
        for name, records in [('isolates', isolates), ('ast', assays)]:
            path = raw / f'ncbi_salmonella_{name}.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
            files.append({'collection': name, 'file': str(path.relative_to(test_root)),
                          'sha256': digest(path), 'records': len(records)})
        (raw / 'acquisition_manifest.json').write_text(json.dumps({'downloads': files}))
        with contextlib.redirect_stdout(io.StringIO()):
            summary = preprocess(test_root)
        iso = pd.read_csv(test_root / 'data/processed/isolate_features.csv.gz', keep_default_na=False).set_index('biosample_acc')
        q(json.loads(iso.loc['B1', 'gene_tokens']) == sorted(odd_tokens),
          'Gene symbols must remain intact across apostrophes, parentheses, slash and comma')
        q(float(iso.loc['B1', 'gene_count']) == 3, 'Gene count must count annotations, not punctuation fragments')
        q(iso.loc['B2', 'amr_analysis_available'] == 1 and float(iso.loc['B2', 'gene_count']) == 0,
          'Explicit completed zero-hit analysis must be available with zero genes')
        q(iso.loc['B3', 'amr_analysis_available'] == 0 and iso.loc['B3', 'gene_count'] == '',
          'Unperformed analysis must remain unavailable rather than zero-hit')
        pairs = pd.read_csv(test_root / 'data/processed/phenotype_pairs.csv.gz', keep_default_na=False).set_index('biosample_acc')
        q(pairs.loc['B1', 'status'] == 'eligible_binary' and pairs.loc['B1', 'assay_count'] == 1,
          'Exact duplicate assay must not inflate evidence')
        q(pairs.loc['B2', 'status'] == 'conflicting', 'Opposing repeated labels must be quarantined')
        q(pairs.loc['B3', 'status'] == 'missing', 'Missing phenotype must not become susceptible')
        q(pairs.loc['B4', 'status'] == 'unmatched_metadata', 'Unmatched isolate join must be flagged')
        q(summary['exact_duplicate_assays'] == 1, 'Known fixture has exactly one duplicate')
        clean_assays = pd.read_csv(test_root / 'data/processed/ast_clean.csv.gz', keep_default_na=False).set_index('source_id')
        combined = clean_assays.loc['a1']
        q(combined['mic_raw'] == '32/4' and float(combined['mic_secondary_raw']) == 4 and
          float(combined['mic_1_upper']) == 32 and float(combined['mic_2_lower']) == 4 and
          float(combined['mic_2_upper']) == 4,
          'Separate API mic_secondary field must survive the full pipeline as a second component')
        return {'isolates': len(isolates), 'assays': len(assays), 'gene_tokens_preserved': odd_tokens,
                'pair_status_counts': summary['pair_status_counts']}


def source_integrity(root):
    q = Checks.require
    manifest = json.loads((root / 'data/raw/acquisition_manifest.json').read_text())
    q(bool(manifest.get('completed_at_utc')), 'Acquisition manifest is not marked complete')
    output = []
    for source in manifest['downloads']:
        path = root / source['file']
        q(path.is_file(), f'Missing source file: {path}')
        actual = digest(path)
        q(actual == source['sha256'], f'SHA256 mismatch for {source["file"]}')
        records = 0
        ids = set()
        for row in json_rows(path):
            records += 1
            q(bool(row.get('id')), f'Missing raw ID at {source["file"]}:{records}')
            q(row['id'] not in ids, f'Duplicate raw source ID: {row["id"]}')
            ids.add(row['id'])
        q(records == source['records'], f'Row count differs from manifest for {source["file"]}')
        q(path.stat().st_size == source['bytes'], f'Byte count differs for {source["file"]}')
        output.append({'file': source['file'], 'records': records, 'distinct_ids': len(ids),
                       'sha256': actual, 'bytes': path.stat().st_size})
    return output


def output_contracts(root):
    q = Checks.require
    processed = root / 'data/processed'
    summary = json.loads((root / 'results/preprocessing_summary.json').read_text())
    iso = pd.read_csv(processed / 'isolate_features.csv.gz', dtype=str, keep_default_na=False)
    pairs = pd.read_csv(processed / 'phenotype_pairs.csv.gz', dtype=str, keep_default_na=False)
    q(not iso.biosample_acc.eq('').any() and iso.biosample_acc.is_unique,
      'Retained isolate BioSample keys must be nonempty and unique')
    q(not iso.target_acc.eq('').any() and iso.target_acc.is_unique,
      'Retained isolate target accession keys must be nonempty and unique')
    q(not pairs.duplicated(['biosample_acc', 'antibiotic']).any(),
      'Phenotype pair keys must be unique')
    eligible = pairs[pairs.status.eq('eligible_binary')]
    q(eligible.phenotype.isin(['resistant', 'susceptible']).all(),
      'Eligible labels must contain only explicit binary target values')
    q(eligible.biosample_acc.isin(iso.biosample_acc).all(), 'Eligible pair has missing isolate metadata')
    forbidden = [x for x in iso.columns if any(s in x.casefold() for s in ('phenotype', 'mic_', 'disk_', 'number_drugs', 'resistance_count'))]
    q(not forbidden, f'Isolate feature export contains phenotype/measurement-derived fields: {forbidden}')
    q(len(iso) == summary['unique_biosamples_metadata'], 'Isolate count disagrees with preprocessing summary')
    q(len(pairs) == summary['unique_biosample_drug_pairs'], 'Pair count disagrees with summary')
    target_map = iso.set_index('biosample_acc').target_acc.to_dict()
    gene_errors = []
    for row in iso.itertuples():
        tokens = json.loads(row.gene_tokens)
        if tokens != sorted(set(tokens)) or any(not isinstance(x, str) or not x or x == '<NONE>' for x in tokens):
            gene_errors.append(row.biosample_acc)
        if row.amr_analysis_available == '0':
            q(not tokens and row.gene_count == '', 'Unavailable genomic calls must not imply zero genes')
        else:
            q(int(float(row.gene_count)) == len(tokens), 'Gene token/count inconsistency')
    q(not gene_errors, f'Malformed gene token lists for {gene_errors[:10]}')
    ids = set()
    secondary_by_id = {}
    dedup_keys = set()
    assay_counts = Counter()
    phenotype_sets = defaultdict(set)
    ast_count = duplicate_count = unmatched = mismatch = invalid = combinations = 0
    dedup_fields = ['biosample_acc', 'target_acc', 'antibiotic', 'phenotype', 'mic_raw', 'disk_raw',
                    'measurement_sign', 'standard', 'platform', 'vendor', 'reagent']
    def truth(value):
        return str(value).casefold() in ('true', '1')
    for chunk in pd.read_csv(processed / 'ast_clean.csv.gz', dtype=str, keep_default_na=False, chunksize=25000):
        for row in chunk.to_dict('records'):
            ast_count += 1
            q(row['source_id'] not in ids, f'Duplicate processed source_id: {row["source_id"]}')
            ids.add(row['source_id'])
            if row.get('mic_secondary_raw'):
                secondary_by_id[row['source_id']] = row['mic_secondary_raw']
            key = tuple(row[x] for x in dedup_fields)
            duplicate = key in dedup_keys
            q(truth(row['exact_duplicate']) == duplicate, f'Incorrect duplicate flag for {row["source_id"]}')
            dedup_keys.add(key)
            duplicate_count += duplicate
            expected_match = row['biosample_acc'] in target_map
            expected_mismatch = bool(expected_match and target_map[row['biosample_acc']] != row['target_acc'])
            q(truth(row['metadata_matched']) == expected_match, 'Incorrect metadata match flag')
            q(truth(row['target_mismatch']) == expected_mismatch, 'Incorrect target accession mismatch flag')
            unmatched += not expected_match
            mismatch += expected_mismatch
            invalid += int(row['measurement_invalid'])
            combinations += int(float(row['mic_components'])) > 1
            expected_type = ('both' if row['mic_raw'] and row['disk_raw'] else
                             'mic' if row['mic_raw'] else 'disk' if row['disk_raw'] else 'missing')
            q(row['measurement_type'] == expected_type, 'MIC and disk measurement namespaces were pooled')
            if not duplicate:
                pair_key = (row['biosample_acc'], row['antibiotic'])
                assay_counts[pair_key] += 1
                if row['phenotype']:
                    phenotype_sets[pair_key].add(row['phenotype'])
    raw_ast_ids = set()
    raw_secondary_by_id = {}
    for source_row in json_rows(root / 'data/raw/ncbi_salmonella_ast.jsonl'):
        raw_ast_ids.add(source_row['id'])
        secondary = clean_text(source_row.get('mic_secondary'))
        if secondary:
            raw_secondary_by_id[source_row['id']] = secondary
    q(secondary_by_id == raw_secondary_by_id,
      'Reported API secondary MIC concentrations changed or disappeared during preprocessing')
    q(ids == raw_ast_ids, 'Raw-to-clean AST ID set changed; expected row preservation')
    q(ast_count == summary['raw_ast_rows'], 'AST count disagrees with summary')
    q(duplicate_count == summary['exact_duplicate_assays'], 'Duplicate count disagrees with summary')
    q(unmatched == summary['unmatched_ast_records'] and mismatch == summary['target_mismatch_records'],
      'Join exception counts disagree with summary')
    q(invalid == summary['measurement_invalid_assays'] and combinations == summary['combination_mic_assays'],
      'Measurement audit counts disagree with summary')
    q(len(assay_counts) == len(pairs), 'Unique assay keys and pair rows disagree')
    for row in pairs.itertuples():
        key = (row.biosample_acc, row.antibiotic)
        q(assay_counts[key] == int(row.assay_count), f'Pair assay_count mismatch for {key}')
        q(sorted(phenotype_sets[key]) == json.loads(row.all_labels), f'Pair labels mismatch for {key}')
    return {'ast_rows': ast_count, 'isolate_rows': len(iso), 'pair_rows': len(pairs),
            'eligible_pairs': len(eligible), 'exact_duplicates': duplicate_count,
            'unmatched_assays': unmatched, 'target_mismatches': mismatch,
            'measurement_flagged_assays': invalid,
            'reported_secondary_mic_rows_preserved': len(secondary_by_id),
            'feature_export_forbidden_fields': forbidden,
            'scope': 'Output export checked for explicit phenotype/measurement fields; actual fitted model X and split leakage require the separate modeling audit.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--fixtures-only', action='store_true')
    args = parser.parse_args()
    checks = Checks()
    checks.run('hand_calculated_measurement_semantics', fixture_measurements)
    checks.run('partial_date_and_text_fixtures', fixture_dates_and_text)
    checks.run('miniature_end_to_end_duplicate_conflict_join_and_gene_fixtures', lambda: fixture_pipeline(args.root))
    if not args.fixtures_only:
        checks.run('frozen_raw_source_sha256_counts_and_ids', lambda: source_integrity(args.root))
        checks.run('processed_output_contracts_and_reconciliation', lambda: output_contracts(args.root))
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Preprocessing semantics, source integrity and output reconciliation. Excludes fitted model feature selection and split validation, which are audited by modeling.',
              'fixtures_only': args.fixtures_only, 'checks': checks.results,
              'all_checks_passed': all(x['passed'] for x in checks.results),
              'passed_checks': sum(x['passed'] for x in checks.results),
              'total_checks': len(checks.results),
              'interpretation': 'Passing the listed finite checks does not prove all data are error-free or establish clinical validity.'}
    destination = args.root / 'results/preprocessing_validation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report['all_checks_passed'] else 1)


if __name__ == '__main__':
    main()




