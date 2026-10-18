"""Independent saved-artifact audit: every holdout prediction and metric.

This reuses fitted pipelines for inference only; it never refits a model.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import joblib
import model_study as study

ROOT=Path(__file__).resolve().parents[1]
def run():
    out=ROOT/'results/model_study'
    features=pd.read_csv(ROOT/'data/processed/isolate_features.csv.gz')
    pairs=pd.read_csv(ROOT/'data/processed/phenotype_pairs.csv.gz')
    split=pd.read_csv(out/'split_manifest.csv.gz')
    predictions=pd.read_csv(out/'predictions.csv.gz')
    metrics=pd.read_csv(out/'metrics.csv')
    summary=json.loads((out/'summary.json').read_text())
    assert features.biosample_acc.is_unique and split.biosample_acc.is_unique
    joined=features.merge(split[['biosample_acc','split']],on='biosample_acc',validate='one_to_one')
    assert len(joined)==len(features)
    sets={s:set(split.loc[split.split.eq(s),'group_id']) for s in ('train','validation','test')}
    assert not (sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
    indexed=joined.set_index('biosample_acc',drop=False)
    audit=[]
    for (drug,name),block in predictions.groupby(['drug','model'],sort=True):
        labels=pairs[pairs.antibiotic.eq(drug)&pairs.status.eq('eligible_binary')].set_index('biosample_acc').phenotype
        assert np.array_equal(block.y.to_numpy(),labels.loc[block.biosample_acc].eq('resistant').astype(int).to_numpy())
        fitted=joblib.load(out/'models'/f'{drug}_{name}.joblib')
        mode=name if name in ('genes_only','metadata_only') else 'combined'
        selected_columns=study.feature_columns(mode)
        assert not any(c in selected_columns for c in ['AST_phenotypes','mic','disk_diffusion','phenotype','biosample_acc','bioproject_acc','group_id'])
        for part,pred in block.groupby('split',sort=True):
            frame=indexed.loc[pred.biosample_acc].copy()
            assert frame.split.eq(part).all()
            for c in study.CATEGORICAL:
                if c in frame: frame[c]=frame[c].astype(object).where(frame[c].notna(),np.nan)
            probability=fitted.predict_proba(frame[selected_columns])[:,1]
            assert np.allclose(probability,pred.probability,atol=1e-12,rtol=0)
            row=metrics[metrics.drug.eq(drug)&metrics.model.eq(name)&metrics.split.eq(part)&metrics.threshold_policy.eq('fixed_0.5')]
            assert len(row)==1
            derived=study.calculate_metrics(pred.y,probability)
            for key,value in derived.items():
                assert np.isclose(value,row.iloc[0][key],atol=1e-12,rtol=0), (drug,name,part,key)
            audit.append({'drug':drug,'model':name,'split':part,'rows':len(pred),'all_saved_probabilities_match':True,'all_fixed_threshold_metrics_match':True})
        if mode!='metadata_only':
            # Model vocabulary is a subset of training-only source tokens; held-out
            # unknown tokens must not have been added after fitting.
            eligible=pairs[pairs.antibiotic.eq(drug)&pairs.status.eq('eligible_binary')&pairs.measurement_invalid.eq(0)].biosample_acc
            train_features=indexed[indexed.split.eq('train')&indexed.index.isin(eligible)]
            train_vocab={t for row in study.token_rows(train_features[['gene_tokens']]) for t in row}
            assert set(fitted.named_steps['features'].named_transformers_['genes'].classes_)==train_vocab
    cv=pd.read_csv(out/'cv_metrics.csv')
    assert len(cv)==18 and cv.groupby('model').fold.nunique().eq(3).all()
    ci=pd.read_csv(out/'bootstrap_ci.csv'); paired=pd.read_csv(out/'paired_differences.csv')
    assert ci.valid_replicates.eq(300).all() and paired.valid_replicates.eq(300).all()
    assert (ci.ci_lower<=ci.ci_upper).all() and (paired.ci_lower<=paired.ci_upper).all()
    assert all(not value for value in summary['warnings'].values()),summary['warnings']
    notebook=json.loads((ROOT/'notebooks/Research_Walkthrough.ipynb').read_text())
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert len(cells)==11 and all(c['execution_count'] for c in cells)
    assert not any(o['output_type']=='error' for c in cells for o in c.get('outputs',[]))
    reproduction=None
    reference=ROOT/'results/reproduction_reference.json'
    if reference.exists():
        original=pd.DataFrame(json.loads(reference.read_text()))
        keys=['drug','model','split','threshold_policy']
        columns=[c for c in original.columns if c not in keys]
        a=original.sort_values(keys).reset_index(drop=True)
        b=metrics.sort_values(keys).reset_index(drop=True)
        assert a[keys].equals(b[keys])
        assert np.allclose(a[columns].to_numpy(float),b[columns].to_numpy(float),atol=1e-12,rtol=0)
        reproduction={'all_frozen_metric_rows_match_initial_run':True,'rows':len(a),
                      'max_absolute_difference':float(np.max(np.abs(a[columns].to_numpy(float)-b[columns].to_numpy(float)))),
                      'fit_timings_excluded':True}
    result={'all_checks_passed':True,'fitted_models_audited':len(set((x['drug'],x['model']) for x in audit)),
            'validation_and_test_predictions_recomputed':sum(x['rows'] for x in audit),
            'all_saved_probabilities_match_1e_12':True,'all_reported_fixed_threshold_metrics_match_1e_12':True,
            'known_group_intersections':0,'grouped_cv_fits_recorded':len(cv),'bootstrap_replicates':300,
            'final_model_and_fold_warnings':0,'executed_notebook_code_cells':len(cells),
            'notebook_error_outputs':0,'model_audits':audit,
            'reproduction_check':reproduction,
            'limitations':'Finite artifact/inference checks cannot prove external validity or complete biological independence.'}
    (ROOT/'results/model_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
