"""Synthetic missingness stress on observed VALIDATION cells; no target use.

Does not evaluate unknown naturally missing values and is not a clinical model.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
def run():
    started=time.perf_counter()
    f=pd.read_csv(ROOT/'data/processed/isolate_features.csv.gz')
    splits=pd.read_csv(ROOT/'results/model_study/split_manifest.csv.gz')
    # Inspect manifest contract rather than silently guess a different holdout.
    splitmap=splits.drop_duplicates('biosample_acc').set_index('biosample_acc')['split']
    f['split']=f.biosample_acc.map(splitmap)
    cols=['collection_year','asm_n50','asm_length','asm_contigs','gene_count']
    train=f.loc[f.split.eq('train'),cols].to_numpy(float)
    valid=f.loc[f.split.eq('validation'),cols].to_numpy(float)
    scaler=StandardScaler().fit(train); a=scaler.transform(train); b=scaler.transform(valid)
    methods={'training_median':SimpleImputer(strategy='median',keep_empty_features=True),
             'iterative_bayesian_ridge':IterativeImputer(random_state=42,max_iter=20,tol=.001,keep_empty_features=True)}
    fitted={}
    for name,method in methods.items():
        t=time.perf_counter(); method.fit(a); fitted[name]=(method,time.perf_counter()-t)
    rng=np.random.default_rng(430); records=[]
    iqr=np.nanquantile(train,.75,axis=0)-np.nanquantile(train,.25,axis=0)
    iqr=np.where(iqr>0,iqr,1)
    for mechanism,rate in [('MCAR',.1),('MCAR',.3),('MAR_contig_conditioned',.3)]:
        mask=rng.random(b.shape)<rate
        if mechanism.startswith('MAR'):
            # ASM contigs remains observed as anchor; others more often missing
            # in assemblies with more contigs. This is a declared simulation.
            rank=pd.Series(valid[:,3]).rank(pct=True).fillna(.5).to_numpy()
            prob=np.clip(.05+.5*rank,.05,.55)
            mask=rng.random(b.shape)<prob[:,None]; mask[:,3]=False
        mask &= np.isfinite(b)
        corrupted=b.copy(); corrupted[mask]=np.nan
        for name,(method,fit_seconds) in fitted.items():
            t=time.perf_counter(); reconstructed=scaler.inverse_transform(method.transform(corrupted)); seconds=time.perf_counter()-t
            for i,col in enumerate(cols):
                selected=mask[:,i]
                if not selected.any(): continue
                error=reconstructed[selected,i]-valid[selected,i]
                records.append({'mechanism':mechanism,'nominal_mask_rate':rate,'method':name,'feature':col,
                                'observed_validation_cells_masked':int(selected.sum()),
                                'rmse_original_units':float(np.sqrt(np.mean(error**2))),
                                'rmse_over_training_iqr':float(np.sqrt(np.mean(error**2))/iqr[i]),
                                'fit_seconds':fit_seconds,'transform_seconds':seconds})
    output=ROOT/'results/imputation_stress.csv'; pd.DataFrame(records).to_csv(output,index=False)
    (ROOT/'results/imputation_stress_design.json').write_text(json.dumps({
        'synthetic_masking_only':True,'fitted_on':'training isolates only','evaluated_on':'observed validation cells only',
        'target_used':False,'test_set_used':False,'n_training':len(train),'n_validation':len(valid),
        'features':cols,'missingness_mechanisms':['MCAR10%','MCAR30%','MAR conditional on recorded assembly contigs'],
        'limitations':'Stress experiments cannot identify the accuracy of naturally missing values. MAR is a simulation, not an asserted true missingness mechanism.',
        'elapsed_seconds':time.perf_counter()-started},indent=2))
    print(output)
if __name__=='__main__': run()
