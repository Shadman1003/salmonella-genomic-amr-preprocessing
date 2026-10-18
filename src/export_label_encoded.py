"""Explicit training-fitted integer-label exports, before model scaling.

These are educational/inspection artifacts. Use the saved full pipeline for
inference. Numeric columns marked imputed are estimates, not new measurements.
"""
import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
import model_study as study
ROOT=Path(__file__).resolve().parents[1]
def run():
    output=ROOT/'data/processed'
    f=pd.read_csv(output/'isolate_features.csv.gz')
    pairs=pd.read_csv(output/'phenotype_pairs.csv.gz')
    p=pairs[pairs.antibiotic.eq('ampicillin')&pairs.status.eq('eligible_binary')&pairs.measurement_invalid.eq(0)]
    p=p[['biosample_acc','phenotype']].merge(f,on='biosample_acc',validate='one_to_one')
    split=pd.read_csv(ROOT/'results/model_study/split_manifest.csv.gz')[['biosample_acc','split']]
    p=p.merge(split,on='biosample_acc',validate='one_to_one')
    for c in study.CATEGORICAL: p[c]=p[c].astype(object).where(p[c].notna(),np.nan)
    model=joblib.load(ROOT/'results/model_study/models/ampicillin_ordinal_median_flags.joblib')
    branches=model.named_steps['features'].named_transformers_
    categorical=branches['categorical']; encoder=categorical.named_steps['encode']
    numeric=branches['numeric']; pre=numeric.named_steps['prescale']
    numbers=study.BASE_NUMERIC+study.GENOMIC_NUMERIC
    encoded=pd.DataFrame({'biosample_acc':p.biosample_acc,'split':p.split,
                          'phenotype':p.phenotype,'y_resistant':p.phenotype.eq('resistant').astype(int)})
    rawcodes=encoder.transform(categorical.named_steps['impute'].transform(p[study.CATEGORICAL]))
    mapping={}
    for j,c in enumerate(study.CATEGORICAL):
        encoded[c+'_label']=rawcodes[:,j].astype(int)
        mapping[c]={str(v):int(i) for i,v in enumerate(encoder.categories_[j])}
    z=pre.transform(p[numbers]); imputed=numeric.named_steps['impute'].transform(z)
    original_units=pre.inverse_transform(imputed[:,:len(numbers)])
    for j,c in enumerate(numbers):
        encoded[c+'_imputed']=original_units[:,j]
        encoded[c+'_was_missing']=p[c].isna().astype(int)
    genes=branches['genes']; matrix=genes.transform(p[['gene_tokens']]).toarray().astype('int8')
    encoded=pd.concat([encoded,pd.DataFrame(matrix,columns=genes.get_feature_names_out(),index=encoded.index)],axis=1)
    files={}
    for part in ('train','validation','test'):
        path=output/f'ampicillin_label_encoded_{part}.csv.gz'
        encoded[encoded.split.eq(part)].to_csv(path,index=False); files[part]={'file':str(path.relative_to(ROOT)),'rows':int(encoded.split.eq(part).sum())}
    info={'fitted_on':'ampicillin training connected groups only','target_mapping':{'susceptible':0,'resistant':1},
          'categorical_maps':mapping,'unknown_category_code':-1,'missing_category':'__MISSING__ (code listed per map)',
          'categorical_codes_are_unscaled':True,'numeric_values_are_imputed_in_original_units':True,
          'genomic_encoding':'training-vocabulary multi-hot symbol=status, not label integers for an entire gene list',
          'columns':len(encoded.columns),'files':files,
          'limitations':'Integer category distances are artificial. These exports are for inspection; frozen pipeline inference includes learned scaling.'}
    (ROOT/'results/label_encoding_export.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))
if __name__=='__main__':run()
