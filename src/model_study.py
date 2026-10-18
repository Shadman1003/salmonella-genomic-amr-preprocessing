"""Frozen, leakage-controlled preprocessing ablations for Salmonella AST.

Run from WSL with the project's prepared scientific Python environment. All
data-adaptive operators are fitted inside each training split. Genomic features
are existing AMRFinderPlus calls; this is not discovery from raw genomes.
"""
from __future__ import annotations

import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'

import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time
import warnings
try:
    import resource
except ImportError:  # Windows can run inference; Linux supplies run-memory audit.
    resource = None

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

if __name__ == '__main__':
    # Keep joblib class references importable when the CLI is run as a script.
    sys.modules['model_study'] = sys.modules[__name__]

SEED = 42
DRUGS = ('ampicillin', 'tetracycline', 'ciprofloxacin')
PRIMARY = 'onehot_median_flags'
CATEGORICAL = ['source_type', 'epi_type', 'country', 'host', 'serovar']
BASE_NUMERIC = ['collection_year', 'asm_n50', 'asm_length', 'asm_contigs']
GENOMIC_NUMERIC = ['amr_analysis_available', 'gene_count']
METRIC_NAMES = ['average_precision', 'roc_auc', 'balanced_accuracy', 'f1',
                'resistance_recall', 'precision', 'brier', 'accuracy']


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(safe_json(value), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding='utf-8')


def token_rows(values):
    """Preserve entire symbols, including slash, parentheses and apostrophes."""
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0].tolist()
    else:
        values = np.asarray(values, dtype=object).reshape(-1).tolist()
    result = []
    for value in values:
        if isinstance(value, str):
            parsed = json.loads(value) if value.strip() else []
        elif isinstance(value, (list, tuple)):
            parsed = value
        elif value is None or pd.isna(value):
            parsed = []
        else:
            raise ValueError(f'Invalid gene_tokens type: {type(value)}')
        if not isinstance(parsed, (list, tuple)) or any(
                not isinstance(t, str) for t in parsed):
            raise ValueError('gene_tokens must contain a JSON list of strings')
        result.append(sorted(set(parsed)))
    return result


class GeneEncoder(TransformerMixin, BaseEstimator):
    """Training-vocabulary multi-hot encoding, never count gene copies."""
    def fit(self, X, y=None):
        self.classes_ = np.asarray(sorted({t for row in token_rows(X) for t in row}),
                                   dtype=object)
        self.vocabulary_ = {t: i for i, t in enumerate(self.classes_)}
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        rows, cols = [], []
        parsed = token_rows(X)
        for i, tokens in enumerate(parsed):
            for token in tokens:
                if token in self.vocabulary_:
                    rows.append(i)
                    cols.append(self.vocabulary_[token])
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                 shape=(len(parsed), len(self.classes_)))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(['gene::' + t for t in self.classes_], dtype=object)


GeneEncoder.__module__ = 'model_study'


def feature_columns(mode):
    if mode == 'genes_only':
        return GENOMIC_NUMERIC + ['gene_tokens']
    columns = CATEGORICAL + BASE_NUMERIC
    return columns if mode == 'metadata_only' else columns + GENOMIC_NUMERIC + ['gene_tokens']


def build_pipeline(encoding='onehot', imputation='median_flags', mode='combined'):
    numeric = (GENOMIC_NUMERIC if mode == 'genes_only' else BASE_NUMERIC +
               ([] if mode == 'metadata_only' else GENOMIC_NUMERIC))
    if imputation == 'iterative_flags':
        imputer = IterativeImputer(random_state=SEED, max_iter=15, tol=1e-3,
            add_indicator=True, keep_empty_features=True, initial_strategy='median',
            sample_posterior=False, skip_complete=False)
    else:
        imputer = SimpleImputer(strategy='median',
            add_indicator=(imputation == 'median_flags'), keep_empty_features=True)
    # StandardScaler preserves NaN. Scaling before iterative imputation prevents
    # assembly lengths (millions) from dominating its global stopping tolerance.
    # The same two scaling stages are used for every imputation configuration.
    transformers = [('numeric', Pipeline([('prescale', StandardScaler()), ('impute', imputer),
                                          ('scale', StandardScaler())]), numeric)]
    if mode != 'genes_only':
        encoder = (OneHotEncoder(handle_unknown='ignore', sparse_output=True)
                   if encoding == 'onehot' else
                   OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                  encoded_missing_value=-2, dtype=float))
        steps = [('impute', SimpleImputer(strategy='constant',
                 fill_value='__MISSING__', keep_empty_features=True)),
                 ('encode', encoder)]
        if encoding == 'ordinal':
            steps.append(('scale', StandardScaler()))
        transformers.append(('categorical', Pipeline(steps), CATEGORICAL))
    if mode != 'metadata_only':
        transformers.append(('genes', GeneEncoder(), ['gene_tokens']))
    return Pipeline([
        ('features', ColumnTransformer(transformers, remainder='drop', sparse_threshold=1.0)),
        ('classifier', LogisticRegression(C=1.0, class_weight=None, solver='liblinear',
                max_iter=2000, random_state=SEED)),
    ])


def calculate_metrics(y, probability, threshold=0.5, sample_weight=None):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability)
    pred = (probability >= threshold).astype(int)
    weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
    support_r = float(weight[y == 1].sum())
    support_s = float(weight[y == 0].sum())
    valid_both = support_r > 0 and support_s > 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        values = {
            'n': float(weight.sum()), 'n_resistant': support_r, 'n_susceptible': support_s,
            'prevalence': support_r / weight.sum(), 'threshold': float(threshold),
            'average_precision': (average_precision_score(y, probability, sample_weight=weight)
                                  if support_r > 0 else np.nan),
            'roc_auc': (roc_auc_score(y, probability, sample_weight=weight)
                        if valid_both else np.nan),
            'balanced_accuracy': (balanced_accuracy_score(y, pred, sample_weight=weight)
                                  if valid_both else np.nan),
            'f1': f1_score(y, pred, sample_weight=weight, zero_division=0),
            'resistance_recall': (recall_score(y, pred, sample_weight=weight,
                                  zero_division=0) if support_r > 0 else np.nan),
            'precision': precision_score(y, pred, sample_weight=weight, zero_division=0),
            'brier': brier_score_loss(y, probability, sample_weight=weight),
            'accuracy': accuracy_score(y, pred, sample_weight=weight),
        }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1], sample_weight=weight).ravel()
    values.update(tn=tn, fp=fp, fn=fn, tp=tp)
    return values


def split_groups(frame, groups):
    groups = np.asarray(groups).astype(str)
    if len(np.unique(groups)) < 5:
        raise ValueError('At least five independent groups are required')
    train, rest = next(GroupShuffleSplit(n_splits=1, train_size=0.6,
                                        random_state=SEED).split(frame, groups=groups))
    val_local, test_local = next(GroupShuffleSplit(n_splits=1, train_size=0.5,
        random_state=SEED + 1).split(frame.iloc[rest], groups=groups[rest]))
    splits = np.full(len(frame), '', dtype=object)
    splits[train], splits[rest[val_local]], splits[rest[test_local]] = 'train', 'validation', 'test'
    sets = {name: set(groups[splits == name]) for name in ('train', 'validation', 'test')}
    assert not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or
                sets['validation'] & sets['test'])
    return splits


def bioproject_groups(frame):
    """Union overlapping BioProjects; empty project is a BioSample singleton."""
    parent = list(range(len(frame)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for i, value in enumerate(frame['bioproject_acc'].fillna('')):
        for project in re.findall(r'PRJ[A-Z]+\d+', str(value)):
            if project in owner:
                a, b = find(i), find(owner[project])
                parent[a] = b
            else:
                owner[project] = i
    return np.asarray(['project_component_' + str(find(i)) for i in range(len(frame))])


def validation_threshold(y, probability):
    if len(np.unique(y)) < 2:
        return 0.5, 'Validation lacks both classes; default threshold retained.'
    p, r, thresholds = precision_recall_curve(y, probability)
    score = np.divide(2 * p[:-1] * r[:-1], p[:-1] + r[:-1],
                      out=np.zeros(len(thresholds)), where=(p[:-1] + r[:-1]) > 0)
    indices = np.flatnonzero(np.isclose(score, np.max(score), rtol=0, atol=1e-12))
    return float(thresholds[indices[-1]]), 'Maximum validation F1; highest threshold breaks ties.'


def fit_model(model, X, y):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(X, y)
    return time.perf_counter() - start, [str(w.message) for w in caught]


def calibrate_table(y, probability):
    frame = pd.DataFrame({'observed': np.asarray(y), 'predicted': probability})
    if frame['predicted'].nunique() <= 1:
        frame['bin'] = 0
    else:
        frame['bin'] = pd.qcut(frame['predicted'], min(10, frame['predicted'].nunique()),
                               duplicates='drop')
    return (frame.groupby('bin', observed=True).agg(n=('observed', 'size'),
            mean_probability=('predicted', 'mean'), resistance_rate=('observed', 'mean'))
            .reset_index().assign(bin=lambda x: x['bin'].astype(str)))


def diagnostic_figure(y, probability, destination, title):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), constrained_layout=True)
    p, r, _ = precision_recall_curve(y, probability)
    axes[0, 0].plot(r, p, color='#117c83')
    axes[0, 0].axhline(np.mean(y), color='#888', linestyle='--', label='Prevalence')
    axes[0, 0].set(xlabel='Resistance recall', ylabel='Precision', title='Precision–recall')
    axes[0, 0].legend()
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, probability)
        axes[0, 1].plot(fpr, tpr, color='#117c83')
    axes[0, 1].plot([0, 1], [0, 1], color='#aaa', linestyle='--')
    axes[0, 1].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC')
    calibration = calibrate_table(y, probability)
    axes[1, 0].plot(calibration['mean_probability'], calibration['resistance_rate'], 'o-',
                    color='#117c83')
    axes[1, 0].plot([0, 1], [0, 1], color='#aaa', linestyle='--')
    axes[1, 0].set(xlabel='Mean predicted probability', ylabel='Observed resistance',
                   title='Calibration (quantile bins)', xlim=(0, 1), ylim=(0, 1))
    matrix = confusion_matrix(y, probability >= 0.5, labels=[0, 1])
    axes[1, 1].imshow(matrix, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        axes[1, 1].text(j, i, f'{value:,}', ha='center', va='center',
                         color='white' if value > matrix.max() * 0.55 else '#222')
    axes[1, 1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['S', 'R'],
                   yticklabels=['S', 'R'], xlabel='Predicted', ylabel='Observed',
                   title='Confusion matrix (threshold 0.50)')
    fig.suptitle(title, fontsize=14)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def extract_encoding(model, encoding, mode):
    transformer = model.named_steps['features']
    numeric = transformer.named_transformers_['numeric']
    imputer, scale = numeric.named_steps['impute'], numeric.named_steps['scale']
    prescale = numeric.named_steps['prescale']
    result = {'encoding': encoding, 'mode': mode, 'feature_names':
              transformer.get_feature_names_out().tolist(),
              'numeric_columns': [c for name, _, cols in transformer.transformers_
                                  if name == 'numeric' for c in cols],
              'numeric_scale_mean': scale.mean_, 'numeric_scale_std': scale.scale_,
              'numeric_preimputation_mean': prescale.mean_,
              'numeric_preimputation_std': prescale.scale_,
              'numeric_imputation_coordinate_system': 'Training-standardized numerical variables.',
              'numeric_imputation_statistics': getattr(imputer, 'statistics_', None),
              'indicator_feature_indices': getattr(getattr(imputer, 'indicator_', None),
                                                   'features_', None)}
    if mode != 'genes_only':
        categorical = transformer.named_transformers_['categorical'].named_steps['encode']
        result['categorical_maps'] = {
            column: {str(value): i for i, value in enumerate(categories)}
            for column, categories in zip(CATEGORICAL, categorical.categories_)}
        result['unknown_category_policy'] = ('ordinal=-1 before scaling; missing=__MISSING__'
            if encoding == 'ordinal' else
            'all-zero encoding for unseen category; no rare pooling in either encoding arm')
    if mode != 'metadata_only':
        result['gene_vocabulary'] = transformer.named_transformers_['genes'].vocabulary_
        result['unknown_gene_policy'] = 'Unseen tokens ignored; counts audited separately.'
    return result


def export_encoded(model, frame, directory):
    transformer = model.named_steps['features']
    names = transformer.get_feature_names_out()
    write_json(directory / 'feature_names.json', names.tolist())
    for split in ('train', 'validation', 'test'):
        part = frame.loc[frame['split'] == split]
        with gzip.open(directory / f'ampicillin_{split}.csv.gz', 'wt', encoding='utf-8', newline='') as handle:
            for start in range(0, len(part), 400):
                chunk = part.iloc[start:start + 400]
                values = transformer.transform(chunk)
                if sparse.issparse(values):
                    values = values.toarray()
                export = pd.DataFrame(values, columns=names)
                export.insert(0, 'y_resistant', chunk['y'].to_numpy())
                export.insert(0, 'group_id', chunk['group_id'].to_numpy())
                export.insert(0, 'biosample_acc', chunk['biosample_acc'].to_numpy())
                export.to_csv(handle, index=False, header=(start == 0), float_format='%.8g')


def bootstrap_comparison(y, groups, predictions, count):
    groups, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    rng = np.random.default_rng(SEED)
    samples = {name: {metric: [] for metric in METRIC_NAMES} for name in predictions}
    differences = {name: {m: [] for m in ('average_precision', 'roc_auc', 'f1')}
                   for name in predictions if name != PRIMARY}
    for _ in range(count):
        cluster_counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        weights = cluster_counts[codes]
        values = {name: calculate_metrics(y, prob, sample_weight=weights)
                  for name, prob in predictions.items()}
        for name in predictions:
            for metric in METRIC_NAMES:
                samples[name][metric].append(values[name][metric])
            if name != PRIMARY:
                for metric in differences[name]:
                    differences[name][metric].append(values[name][metric] - values[PRIMARY][metric])
    intervals, paired = [], []
    def interval(values):
        finite = np.asarray(values)[np.isfinite(values)]
        lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
        return dict(ci_lower=lo, ci_upper=hi, valid_replicates=len(finite))
    for name, by_metric in samples.items():
        for metric, values in by_metric.items():
            point = calculate_metrics(y, predictions[name])[metric]
            intervals.append(dict(model=name, metric=metric, estimate=point,
                                  **interval(values)))
    for name, by_metric in differences.items():
        for metric, values in by_metric.items():
            point = (calculate_metrics(y, predictions[name])[metric] -
                     calculate_metrics(y, predictions[PRIMARY])[metric])
            paired.append(dict(model=name, reference=PRIMARY, metric=metric,
                               difference=point, **interval(values)))
    return pd.DataFrame(intervals), pd.DataFrame(paired)


def support(frame):
    return {str(k): int(v) for k, v in frame['phenotype'].value_counts().items()}


def run(project, bootstrap_count=300, minimum_class=100):
    started = time.perf_counter()
    output = project / 'results' / 'model_study'
    for child in ('figures', 'models', 'encoded'):
        (output / child).mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(project / 'data' / 'processed' / 'isolate_features.csv.gz', low_memory=False)
    pairs = pd.read_csv(project / 'data' / 'processed' / 'phenotype_pairs.csv.gz', low_memory=False)
    required = ['biosample_acc', 'group_id', 'gene_tokens', 'bioproject_acc'] + CATEGORICAL + BASE_NUMERIC + GENOMIC_NUMERIC
    absent = sorted(set(required) - set(features))
    if absent:
        raise ValueError(f'Missing isolate feature columns: {absent}')
    assert features['biosample_acc'].notna().all() and features['biosample_acc'].is_unique
    assert features['group_id'].notna().all()
    assert not pairs.duplicated(['biosample_acc', 'antibiotic']).any()
    for column in CATEGORICAL:
        features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
    for column in BASE_NUMERIC + GENOMIC_NUMERIC:
        features[column] = pd.to_numeric(features[column], errors='raise').replace([np.inf, -np.inf], np.nan)
    token_rows(features[['gene_tokens']])  # Fail early on malformed genomic data.
    features['split'] = split_groups(features, features['group_id'])
    manifest = features[['biosample_acc', 'target_acc', 'group_id', 'bioproject_acc', 'split']].copy()
    manifest.to_csv(output / 'split_manifest.csv.gz', index=False)
    forbidden = {'phenotype', 'measurement', 'mic', 'disk_diffusion', 'AST_phenotypes',
                 'antibiotic', 'group_id', 'bioproject_acc', 'target_acc', 'biosample_acc'}
    assert not (set(feature_columns('combined')) & forbidden)
    pairs['antibiotic'] = pairs['antibiotic'].str.strip().str.lower()
    pairs['phenotype'] = pairs['phenotype'].astype(str).str.strip().str.lower().replace(
        {'r': 'resistant', 's': 'susceptible'})
    invalid = pairs['measurement_invalid'].astype(str).str.lower().isin(['true', '1', '1.0'])
    status_counts = (pairs.groupby(['antibiotic', 'phenotype', 'status'], dropna=False)
                    .size().reset_index(name='n_pairs'))
    status_counts.to_csv(output / 'all_phenotype_status_counts.csv', index=False)
    eligibility = (pairs['status'].eq('eligible_binary') &
                   pairs['phenotype'].isin(['resistant', 'susceptible']) & ~invalid)
    pairs['measurement_excluded_from_model'] = invalid
    summary = {
        'protocol': 'Fixed 60/20/20 SNP/BioSample connected-component split; seed 42/43.',
        'primary_drug': 'ampicillin', 'primary_model': PRIMARY,
        'primary_threshold': 0.5, 'primary_metric': 'average_precision',
        'class_weight': None, 'C': 1.0, 'solver': 'liblinear',
        'genomic_features': 'Training-vocabulary multi-hot of reported unique symbol=quality tokens.',
        'label_definition': 'Source-reported resistant=1, susceptible=0; all other labels excluded.',
        'target_label_mapping': {'susceptible': 0, 'resistant': 1},
        'invalid_measurement_rule': 'Exclude eligible pairs flagged measurement_invalid (no imputation).',
        'missing_target_rule': 'Never impute phenotype or interpret untested drug as susceptible.',
        'imputation_caveat': 'IterativeImputer is single deterministic multivariate imputation, not pooled MI inference.',
        'numeric_scaling': 'Training-only StandardScaler before imputation in every arm; same second scaler after imputation for values/flags.',
        'categorical_comparison': 'Both arms retain identical categories; no rare-category pooling.',
        'bootstrap': {'replicates': bootstrap_count, 'unit': 'SNP/BioSample connected component',
            'paired': True, 'method': 'percentile 95%; frozen models, threshold 0.5',
            'scope': 'Conditional on fitted training models; excludes training/model-selection uncertainty.'},
        'all_isolates': len(features), 'all_groups': features['group_id'].nunique(),
        'invalid_measurement_pairs': int(invalid.sum()), 'skipped_tasks': {},
        'invalid_measurement_pairs_by_task': {drug: int((invalid & pairs['antibiotic'].eq(drug)).sum())
                                             for drug in DRUGS},
        'alternate_split_caveat': 'Alternate splits have different test populations; metric differences do not isolate a causal leakage effect.',
        'tasks': {}, 'warnings': {}, 'packages': {p: importlib.metadata.version(p)
                      for p in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'matplotlib')},
    }
    metrics_rows, prediction_rows, cv_rows, encoding_maps, sensitivity_rows = [], [], [], {}, []
    primary_frame, primary_probabilities = None, {}
    for drug in DRUGS:
        selected = pairs.loc[eligibility & pairs['antibiotic'].eq(drug)].copy()
        selected['y'] = selected['phenotype'].eq('resistant').astype(int)
        frame = selected.merge(features, on='biosample_acc', how='inner', validate='one_to_one')
        counts = support(frame)
        if min(counts.get('resistant', 0), counts.get('susceptible', 0)) < minimum_class:
            summary['skipped_tasks'][drug] = {'reason': 'Insufficient class support', 'support': counts}
            continue
        masks = {name: frame['split'].eq(name).to_numpy() for name in ('train', 'validation', 'test')}
        if any(len(np.unique(frame.loc[mask, 'y'])) < 2 for mask in masks.values()):
            summary['skipped_tasks'][drug] = {'reason': 'Fixed split lacks both classes', 'support': counts}
            continue
        summary['tasks'][drug] = {'n': len(frame), 'unmatched_eligible_pairs': len(selected) - len(frame),
            'splits': {name: {'n': int(mask.sum()), 'groups': frame.loc[mask, 'group_id'].nunique(),
                              'support': support(frame.loc[mask])} for name, mask in masks.items()}}
        configurations = ([(enc + '_' + imp, enc, imp, 'combined')
                           for enc in ('ordinal', 'onehot')
                           for imp in ('median', 'median_flags', 'iterative_flags')]
                          if drug == 'ampicillin' else [(PRIMARY, 'onehot', 'median_flags', 'combined')])
        configurations += [('metadata_only', 'onehot', 'median_flags', 'metadata_only'),
                           ('genes_only', 'onehot', 'median_flags', 'genes_only')]
        for name, encoding, imputation, mode in configurations:
            print(f'Fit {drug}: {name} ({masks["train"].sum():,} training isolates)', flush=True)
            model = build_pipeline(encoding, imputation, mode)
            columns = feature_columns(mode)
            elapsed, caught = fit_model(model, frame.loc[masks['train'], columns], frame.loc[masks['train'], 'y'])
            summary['warnings'][drug + '/' + name] = caught
            probabilities = {}
            for split in ('validation', 'test'):
                part = frame.loc[masks[split]]
                probability = model.predict_proba(part[columns])[:, 1]
                assert np.isfinite(probability).all()
                probabilities[split] = probability
                values = calculate_metrics(part['y'], probability)
                metrics_rows.append(dict(drug=drug, model=name, split=split,
                    threshold_policy='fixed_0.5', fit_seconds=elapsed, **values))
                prediction_rows.append(pd.DataFrame(dict(drug=drug, model=name, split=split,
                    biosample_acc=part['biosample_acc'], group_id=part['group_id'],
                    y=part['y'], probability=probability)))
            threshold, explanation = validation_threshold(frame.loc[masks['validation'], 'y'], probabilities['validation'])
            metrics_rows.append(dict(drug=drug, model=name, split='test',
                threshold_policy='validation_max_f1_secondary', fit_seconds=elapsed,
                **calculate_metrics(frame.loc[masks['test'], 'y'], probabilities['test'], threshold)))
            summary['tasks'][drug].setdefault('validation_thresholds', {})[name] = {
                'threshold': threshold, 'rule': explanation}
            file_path = output / 'models' / f'{drug}_{name}.joblib'
            joblib.dump(model, file_path, compress=3)
            restored = joblib.load(file_path)
            probe = frame.loc[masks['test'], columns].head(8).copy()
            assert np.allclose(restored.predict_proba(probe)[:, 1], probabilities['test'][:len(probe)],
                               atol=1e-12, rtol=0)
            if mode != 'genes_only':
                probe.loc[:, CATEGORICAL] = '__UNSEEN_CATEGORY_INFERENCE_CHECK__'
            for c in BASE_NUMERIC + GENOMIC_NUMERIC:
                if c in probe:
                    probe[c] = np.nan
            if 'gene_tokens' in probe:
                probe['gene_tokens'] = json.dumps(['__UNSEEN_GENE__=COMPLETE'])
            assert np.isfinite(restored.predict_proba(probe)).all()
            if drug == 'ampicillin':
                encoding_maps[name] = extract_encoding(model, encoding, mode)
                primary_probabilities[name] = probabilities['test']
            if name == PRIMARY:
                calibration = calibrate_table(frame.loc[masks['test'], 'y'], probabilities['test'])
                calibration.to_csv(output / f'{drug}_calibration.csv', index=False)
                diagnostic_figure(frame.loc[masks['test'], 'y'], probabilities['test'],
                    output / 'figures' / f'{drug}_diagnostics.png', drug.capitalize() + ': frozen primary model')
                if drug == 'ampicillin':
                    primary_frame = frame
                    export_encoded(model, frame, output / 'encoded')
            if drug == 'ampicillin' and mode == 'combined':
                train = frame.loc[masks['train']].reset_index(drop=True)
                for fold, (fit_idx, cv_idx) in enumerate(StratifiedGroupKFold(
                        n_splits=3, shuffle=True, random_state=SEED).split(train, train['y'], train['group_id']), 1):
                    assert not (set(train.iloc[fit_idx]['group_id']) & set(train.iloc[cv_idx]['group_id']))
                    fold_model = build_pipeline(encoding, imputation, mode)
                    seconds, fold_warnings = fit_model(fold_model, train.iloc[fit_idx][columns], train.iloc[fit_idx]['y'])
                    prob = fold_model.predict_proba(train.iloc[cv_idx][columns])[:, 1]
                    cv_rows.append(dict(drug=drug, model=name, fold=fold,
                        n_training=len(fit_idx), n_validation=len(cv_idx), fit_seconds=seconds,
                        **calculate_metrics(train.iloc[cv_idx]['y'], prob)))
                    summary['warnings'][f'CV/{name}/{fold}'] = fold_warnings
        print(f'Finished {drug}', flush=True)
    if primary_frame is not None:
        test = primary_frame.loc[primary_frame['split'].eq('test')]
        ci, differences = bootstrap_comparison(test['y'].to_numpy(), test['group_id'],
                                               primary_probabilities, bootstrap_count)
        ci.to_csv(output / 'bootstrap_ci.csv', index=False)
        differences.to_csv(output / 'paired_differences.csv', index=False)
        ap = ci.loc[ci['metric'].eq('average_precision')].set_index('model').loc[list(primary_probabilities)]
        fig, ax = plt.subplots(figsize=(9.5, 5.3), constrained_layout=True)
        ax.errorbar(ap['estimate'], np.arange(len(ap)), xerr=np.vstack([
            np.maximum(0, ap['estimate'] - ap['ci_lower']), np.maximum(0, ap['ci_upper'] - ap['estimate'])]),
            fmt='o', color='#117c83', capsize=3)
        ax.set(yticks=np.arange(len(ap)), yticklabels=ap.index,
               xlabel='Test average precision (paired group bootstrap 95% CI)',
               title='Ampicillin: preprocessing ablations and controls')
        ax.invert_yaxis()
        ax.axvline(test['y'].mean(), color='#999', linestyle='--', label='Test prevalence')
        ax.legend(loc='lower right')
        fig.savefig(output / 'figures' / 'ampicillin_ablation_average_precision.png', dpi=170)
        plt.close(fig)
        # Diagnostic alternate splits are never used to select the primary model.
        for name, groups in [('biosample_random_clone_overlap_diagnostic', features['biosample_acc'].to_numpy()),
                             ('bioproject_component_holdout', bioproject_groups(features))]:
            try:
                split_lookup = pd.Series(split_groups(features, groups), index=features['biosample_acc'])
                frame = primary_frame.copy()
                frame['alt_split'] = frame['biosample_acc'].map(split_lookup)
                fit = frame['alt_split'].eq('train')
                evaluation = frame['alt_split'].eq('test')
                if min(frame.loc[fit, 'y'].nunique(), frame.loc[evaluation, 'y'].nunique()) < 2:
                    raise ValueError('Alternate fixed split lacks both classes')
                model = build_pipeline()
                seconds, caught = fit_model(model, frame.loc[fit, feature_columns('combined')], frame.loc[fit, 'y'])
                probability = model.predict_proba(frame.loc[evaluation, feature_columns('combined')])[:, 1]
                fit_clusters = set(frame.loc[fit, 'group_id'])
                overlap = frame.loc[evaluation, 'group_id'].isin(fit_clusters)
                sensitivity_rows.append(dict(analysis=name, status='evaluated', fit_seconds=seconds,
                    n_train=int(fit.sum()), test_rows_with_training_SNP_component=int(overlap.sum()),
                    **calculate_metrics(frame.loc[evaluation, 'y'], probability)))
                summary['warnings'][name] = caught
            except ValueError as exc:
                sensitivity_rows.append(dict(analysis=name, status='not_evaluable', reason=str(exc)))
        summary['inference_checks'] = {
            'serialized_predictions_match': True, 'unknown_categories_do_not_fail': True,
            'unknown_genes_do_not_fail': True, 'missing_numeric_values_do_not_fail': True,
            'split_group_intersections': 0,
            'limitations': 'Ingestion metadata is not patient-level context; SNP grouping is not complete phylogenetic validation.'}
        train_tokens = set(encoding_maps[PRIMARY].get('gene_vocabulary', {}))
        summary['unseen_gene_tokens'] = {split: {
            'n_isolates_with_unseen_token': sum(bool(set(tokens) - train_tokens) for tokens in
                token_rows(primary_frame.loc[primary_frame['split'].eq(split), ['gene_tokens']])),
            'n_unique_unseen_tokens': len({t for row in token_rows(primary_frame.loc[
                primary_frame['split'].eq(split), ['gene_tokens']]) for t in row} - train_tokens)
            } for split in ('validation', 'test')}
    pd.DataFrame(metrics_rows).to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(cv_rows).to_csv(output / 'cv_metrics.csv', index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output / 'split_sensitivities.csv', index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(output / 'predictions.csv.gz', index=False)
    write_json(output / 'encoding_maps.json', encoding_maps)
    summary['elapsed_seconds'] = time.perf_counter() - started
    summary['peak_resident_memory_mib'] = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
        (1024 * 1024 if sys.platform == 'darwin' else 1024)) if resource else None
    write_json(output / 'summary.json', summary)
    print(json.dumps(safe_json({'finished': True, 'output': str(output),
          'tasks': list(summary['tasks']), 'elapsed_seconds': summary['elapsed_seconds']})), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--bootstrap', type=int, default=300)
    parser.add_argument('--minimum-class', type=int, default=100)
    parser.add_argument('--predict', type=Path, help='CSV/CSV.gz of raw isolate feature rows for frozen inference')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.predict:
        if not args.model or not args.output:
            parser.error('--predict requires --model and --output')
        frame = pd.read_csv(args.predict, low_memory=False)
        for column in CATEGORICAL:
            if column in frame:
                frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
        model = joblib.load(args.model)
        result = pd.DataFrame({'biosample_acc': frame['biosample_acc'],
                               'p_resistant': model.predict_proba(frame)[:, 1]})
        result.to_csv(args.output, index=False)
        return
    if args.bootstrap < 20:
        parser.error('--bootstrap must be at least 20')
    run(args.project_root, args.bootstrap, args.minimum_class)


if __name__ == '__main__':
    main()
