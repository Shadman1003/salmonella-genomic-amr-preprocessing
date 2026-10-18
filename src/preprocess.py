"""Auditable, deterministic preprocessing of frozen NCBI AST and isolate JSONL.

No AST measurements or phenotype-derived counts are exposed to model features.
Run with the Python environment described in README.txt.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math, re, resource, time, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

MISSING = {'', 'null', 'none', 'nan', 'na', 'n/a', 'not provided', 'missing', 'not collected'}
LABELS = {'susceptible','resistant','intermediate','nonsusceptible','not defined',
          'susceptible-dose dependent','susceptible dose dependent','not determined'}
MAX_YEAR = 2026

def clean_text(value):
    if value is None: return ''
    text = unicodedata.normalize('NFKC', str(value)).strip()
    text = re.sub(r'\s+', ' ', text).replace('\u2013','-').replace('\u2011','-')
    return '' if text.casefold() in MISSING else text

def canonical(value): return clean_text(value).casefold()

def json_rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def date_interval(value):
    """Only accepted unambiguous ISO precision/ranges; never invent a month/day."""
    text = clean_text(value)
    result = {'date_raw':text,'date_precision':'missing','year_lower':None,'year_upper':None,'collection_year':None}
    if not text: return result
    years = re.findall(r'(?<!\d)(?:18|19|20)\d{2}(?!\d)', text)
    if not years:
        result['date_precision']='unparsed'; return result
    if any(int(y)<1900 or int(y)>MAX_YEAR for y in years):
        result['date_precision']='out_of_scope'; return result
    try:
        if re.fullmatch(r'\d{4}',text): precision='year'
        elif re.fullmatch(r'\d{4}-\d{2}',text): pd.Timestamp(text+'-01'); precision='month'
        elif re.fullmatch(r'\d{4}-\d{2}-\d{2}',text): pd.Timestamp(text); precision='day'
        elif '/' in text and len(years)==2:
            ends=text.split('/')
            if len(ends)!=2 or any(date_interval(x)['date_precision'] not in ('year','month','day') for x in ends):
                result['date_precision']='invalid'; return result
            if pd.Timestamp(ends[0])>pd.Timestamp(ends[1]):
                result['date_precision']='invalid'; return result
            precision='range'
        else: result['date_precision']='unparsed'; return result
        lo,hi=min(map(int,years)),max(map(int,years))
        result.update(date_precision=precision,year_lower=lo,year_upper=hi,collection_year=lo if lo==hi else None)
    except (ValueError,TypeError): result['date_precision']='invalid'
    return result

def measurement(value, sign, kind):
    """Retain censored intervals and combination components; do not divide X/Y."""
    raw = clean_text(value)
    output = {kind+'_raw':raw, kind+'_status':'structural_missing' if not raw else 'valid',
              kind+'_components':0,kind+'_censored':False,
              kind+'_secondary_semantics':'not_applicable'}
    for i in (1,2):
        output.update({f'{kind}_{i}_lower':None,f'{kind}_{i}_upper':None,
                       f'{kind}_{i}_lower_closed':False,f'{kind}_{i}_upper_closed':False})
    if not raw: return output
    parts = raw.split('/')
    output[kind+'_components']=len(parts)
    if len(parts)>1: output[kind+'_secondary_semantics']='reported_secondary_concentration_no_independent_comparator'
    if len(parts)>2 or (kind=='disk' and len(parts)>1):
        output[kind+'_status']='unparsed_combination'; return output
    if sign not in ('==','<','<=','>','>='):
        output[kind+'_status']='unknown_sign'; return output
    for i,part in enumerate(parts,1):
        try: x=float(part)
        except ValueError: output[kind+'_status']='nonnumeric'; return output
        if not math.isfinite(x) or (kind=='mic' and not 0<x<=1024) or (kind=='disk' and not 6<=x<=150):
            output[kind+'_status']='outside_submission_range'; return output
        lo,hi=x,x; lc=uc=True
        # Source supplies one sign for the assay; do not extrapolate it to the
        # inhibitor/second component when no independent comparator is given.
        if i==1 and sign in ('<','<='): lo=0 if kind=='mic' else 6; lc=kind=='disk'; uc=sign=='<='
        if i==1 and sign in ('>','>='): hi=None; lc=sign=='>='; uc=False
        output.update({f'{kind}_{i}_lower':lo,f'{kind}_{i}_upper':hi,
                       f'{kind}_{i}_lower_closed':lc,f'{kind}_{i}_upper_closed':uc})
    output[kind+'_censored']=sign!='=='
    return output

class UnionFind:
    def __init__(self): self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        if self.parent[x]!=x: self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b: self.parent[max(a,b)]=min(a,b)

def preprocess(root):
    start=time.perf_counter(); raw=root/'data/raw'; out=root/'data/processed'; audit=root/'results/audit'
    out.mkdir(parents=True,exist_ok=True); audit.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((raw/'acquisition_manifest.json').read_text())
    for source in manifest['downloads']:
        assert digest(root/source['file'])==source['sha256'], 'Frozen raw SHA256 mismatch'
    ast_path=raw/'ncbi_salmonella_ast.jsonl'; iso_path=raw/'ncbi_salmonella_isolates.jsonl'
    # Preserve schema/missingness field counts without loading nested raw arrays in a frame.
    schema={}; field_types=defaultdict(Counter); field_present=Counter(); iso_records=[]
    iso_missing=Counter(); gene_states=Counter(); bad_qc=[]; date_rows=[]
    for row in json_rows(iso_path):
        assert row.get('species_taxid') in (28901,'28901'), 'Organism-group query contains a non-enterica or unverified species; review cohort'
        for k,v in row.items(): field_present[k]+=1; field_types[k][type(v).__name__]+=1
        bio=clean_text(row.get('biosample_acc')); target=clean_text(row.get('target_acc'))
        d=date_interval(row.get('collection_date')); date_rows.append({'biosample_acc':bio,**d})
        # '<NONE>' is an explicit zero-hit analysis; empty arrays have different semantics.
        values=row.get('AMR_genotypes',[])
        if isinstance(values,str): values=[values]
        values=[clean_text(v) for v in values if clean_text(v)]
        explicit_none='<NONE>' in values
        applied=row.get('amrfinderplus_applied') in (1,'1',True)
        available=bool(applied and (values or explicit_none or row.get('number_amr_genes')==0))
        # A zero count is accepted only when the completed-analysis flag is explicit.
        tokens=sorted(set(v for v in values if v!='<NONE>')) if available else []
        for token in tokens: gene_states[token.rsplit('=',1)[-1] if '=' in token else 'UNPARSED']+=1
        feature={'biosample_acc':bio,'target_acc':target,
                 'bioproject_acc':clean_text(row.get('bioproject_acc')),
                 'snp_cluster':clean_text(row.get('erd_group_acc') or row.get('erd_group')),
                 'clustered':row.get('clustered',0), 'source_type':canonical(row.get('source_type')),
                 'epi_type':canonical(row.get('epi_type')), 'country':clean_text(row.get('geo_loc_name')).split(':')[0],
                 'host':canonical(row.get('host')),'serovar':canonical(row.get('serovar') or row.get('serotype')),
                 'collection_year':d['collection_year'],'date_precision':d['date_precision'],
                 'amr_analysis_available':int(available),'gene_count':len(tokens) if available else None,
                 'gene_tokens':json.dumps(tokens,ensure_ascii=False),
                 'amrfinderplus_version':clean_text(row.get('amrfinderplus_version')),
                 'refgene_db_version':clean_text(row.get('refgene_db_version')),
                 'analysis_type':clean_text(row.get('amrfinderplus_analysis_type')),
                 'species_taxid':row.get('species_taxid')}
        for field,name in [('asm_stats_contig_n50','asm_n50'),('asm_stats_length_bp','asm_length'),('asm_stats_n_contig','asm_contigs')]:
            value=row.get(field)
            try:
                value=float(value)
                if not math.isfinite(value) or value<=0: raise ValueError
            except (ValueError,TypeError):
                if row.get(field) is not None: bad_qc.append({'biosample_acc':bio,'field':field,'raw':row.get(field)})
                value=None
            feature[name]=value
        # N50 exceeding total length is an internally impossible assembly summary.
        if feature['asm_n50'] and feature['asm_length'] and feature['asm_n50']>feature['asm_length']:
            bad_qc.append({'biosample_acc':bio,'field':'asm_n50','raw':feature['asm_n50']})
            feature['asm_n50']=None
        iso_records.append(feature)
    iso=pd.DataFrame(iso_records)
    schema['isolates']={'rows':len(iso),'fields':{k:{'present':field_present[k],'absent':len(iso)-field_present[k],
                         'types':dict(v)} for k,v in sorted(field_types.items())}}
    iso.to_csv(audit/'isolate_candidates.csv.gz',index=False)
    multiplicity=iso.groupby('biosample_acc').size(); multi=set(multiplicity[multiplicity>1].index)
    iso[iso.biosample_acc.isin(multi)].to_csv(audit/'ambiguous_biosample_metadata.csv.gz',index=False)
    # Conservative primary joins: quarantine ALL non-unique BioSamples, never choose by label.
    iso=iso[~iso.biosample_acc.isin(multi)&iso.biosample_acc.ne('')].copy()
    uf=UnionFind()
    for row in iso.itertuples():
        uf.find('B:'+row.biosample_acc)
        if row.snp_cluster: uf.union('B:'+row.biosample_acc,'C:'+row.snp_cluster)
    iso['group_id']=[uf.find('B:'+x) for x in iso.biosample_acc]
    iso.to_csv(out/'isolate_features.csv.gz',index=False)
    pd.DataFrame(date_rows).to_csv(audit/'date_precision.csv.gz',index=False)
    pd.DataFrame(bad_qc,columns=['biosample_acc','field','raw']).to_csv(audit/'invalid_assembly_measurements.csv',index=False)
    # Row-level assay records: long format is intentional, not automatically duplication.
    rows=[]; normalization=[]; fields=defaultdict(Counter); present=Counter(); raw_schema=Counter()
    for row in json_rows(ast_path):
        for k,v in row.items(): present[k]+=1; fields[k][type(v).__name__]+=1
        source_id=row['id']; sign=clean_text(row.get('measurement_sign'))
        if sign=='=': normalization.append({'source_id':source_id,'field':'measurement_sign','before':'=','after':'==','reason':'equivalent equality comparator'}); sign='=='
        new={'source_id':source_id,'biosample_acc':clean_text(row.get('biosample_acc')),
             'target_acc':clean_text(row.get('target_acc')),'antibiotic':canonical(row.get('antibiotic')),
             'phenotype':canonical(row.get('phenotype')), 'phenotype_raw':clean_text(row.get('phenotype')),
             'standard':clean_text(row.get('standard')),'platform':clean_text(row.get('platform')),
             'vendor':clean_text(row.get('vendor')),'reagent':clean_text(row.get('reagent')),
             'measurement_sign':sign, 'bioproject_acc':clean_text(row.get('bioproject_acc'))}
        for key in ('antibiotic','phenotype'):
            before=str(row.get(key,''))
            if before!=new[key]: normalization.append({'source_id':source_id,'field':key,'before':before,'after':new[key],'reason':'Unicode/whitespace/case canonicalization'})
        mic_value=row.get('mic')
        secondary=clean_text(row.get('mic_secondary'))
        if secondary: mic_value=f'{clean_text(mic_value)}/{secondary}'
        new['mic_secondary_raw']=secondary
        new.update(measurement(mic_value,sign,'mic'))
        new.update(measurement(row.get('disk_diffusion'),sign,'disk'))
        new['measurement_type']='both' if new['mic_raw'] and new['disk_raw'] else ('mic' if new['mic_raw'] else ('disk' if new['disk_raw'] else 'missing'))
        new['measurement_invalid']=int(new['mic_status'] not in ('valid','structural_missing') or new['disk_status'] not in ('valid','structural_missing') or new['measurement_type']=='both')
        new['label_recognized']=int(new['phenotype'] in LABELS or new['phenotype']=='')
        rows.append(new)
    ast=pd.DataFrame(rows)
    schema['ast']={'rows':len(ast),'fields':{k:{'present':present[k],'absent':len(ast)-present[k],
                    'types':dict(v)} for k,v in sorted(fields.items())}}
    ast['metadata_matched']=ast.biosample_acc.isin(set(iso.biosample_acc))
    target_map=iso.set_index('biosample_acc').target_acc.to_dict()
    ast['target_mismatch']=[bool(target_map.get(b) and target_map[b]!=t) for b,t in zip(ast.biosample_acc,ast.target_acc)]
    ast['exact_duplicate']=ast.duplicated(subset=['biosample_acc','target_acc','antibiotic','phenotype','mic_raw','disk_raw','measurement_sign','standard','platform','vendor','reagent'],keep='first')
    ast.to_csv(out/'ast_clean.csv.gz',index=False)
    ast[ast.exact_duplicate].to_csv(audit/'exact_duplicate_assays.csv.gz',index=False)
    ast[ast.measurement_invalid.eq(1)].to_csv(audit/'measurement_quarantine.csv.gz',index=False)
    ast[~ast.metadata_matched|ast.target_mismatch].to_csv(audit/'join_exceptions.csv.gz',index=False)
    pd.DataFrame(normalization,columns=['source_id','field','before','after','reason']).to_csv(audit/'normalization_changes.csv.gz',index=False)
    unique=ast[~ast.exact_duplicate]
    pairs=[]
    for (bio,drug),group in unique.groupby(['biosample_acc','antibiotic'],sort=True,dropna=False):
        labels=sorted(set(x for x in group.phenotype if x))
        if not bio or not drug: status='invalid_key'
        elif len(labels)>1: status='conflicting'
        elif not labels: status='missing'
        elif labels[0] not in ('resistant','susceptible'): status='nonbinary'
        elif group.phenotype.eq('').any(): status='incomplete_repeats'
        elif not group.metadata_matched.all() or group.target_mismatch.any(): status='unmatched_metadata'
        else: status='eligible_binary'
        pairs.append({'biosample_acc':bio,'antibiotic':drug,'phenotype':labels[0] if len(labels)==1 else '',
                      'status':status,'all_labels':json.dumps(labels),'assay_count':len(group),
                      'standards':json.dumps(sorted(set(group.standard))),
                      'measurement_invalid':int(group.measurement_invalid.any()),
                      'source_ids':json.dumps(group.source_id.tolist())})
    pairs=pd.DataFrame(pairs); pairs.to_csv(out/'phenotype_pairs.csv.gz',index=False)
    pairs[pairs.status.ne('eligible_binary')].to_csv(audit/'phenotype_pair_quarantine.csv.gz',index=False)
    labels=ast.groupby(['antibiotic','phenotype']).size().rename('assays').reset_index()
    labels.to_csv(audit/'drug_phenotype_counts.csv',index=False)
    missing=(iso.replace('',np.nan).isna().mean()*100).rename('missing_percent').rename_axis('field').reset_index()
    missing.to_csv(audit/'isolate_feature_missingness.csv',index=False)
    ast.groupby(['measurement_type','measurement_sign']).size().rename('records').reset_index().to_csv(audit/'censoring_counts.csv',index=False)
    summary={'raw_ast_rows':len(ast),'raw_isolate_records':schema['isolates']['rows'],
             'unique_biosamples_metadata':len(iso),'ambiguous_biosamples_quarantined':len(multi),
             'exact_duplicate_assays':int(ast.exact_duplicate.sum()),'unique_biosample_drug_pairs':len(pairs),
             'pair_status_counts':pairs.status.value_counts().to_dict(),
             'phenotype_counts':ast.phenotype.value_counts(dropna=False).to_dict(),
             'measurement_types':ast.measurement_type.value_counts().to_dict(),
             'comparators':ast.measurement_sign.value_counts().to_dict(),
             'censored_measurements':int((ast.mic_censored|ast.disk_censored).sum()),
             'measurement_invalid_assays':int(ast.measurement_invalid.sum()),
             'combination_mic_assays':int(ast.mic_components.gt(1).sum()),
             'missing_platform':int(ast.platform.eq('').sum()),'missing_vendor':int(ast.vendor.eq('').sum()),
             'unknown_phenotype_rows':int(ast.label_recognized.eq(0).sum()),
             'unmatched_ast_records':int((~ast.metadata_matched).sum()),'target_mismatch_records':int(ast.target_mismatch.sum()),
             'amr_available_isolates':int(iso.amr_analysis_available.sum()),
             'species_taxid_counts':iso.species_taxid.value_counts().to_dict(),
             'gene_status_counts':dict(gene_states),'date_precision_counts':iso.date_precision.value_counts().to_dict(),
             'snp_cluster_available_isolates':int(iso.snp_cluster.ne('').sum()),'connected_groups':iso.group_id.nunique(),
             'largest_group_isolates':int(iso.group_id.value_counts().max()),
             'preprocess_wall_seconds':time.perf_counter()-start,
             'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             'policies':{'labels':'Never imputed; unanimous source R/S only for primary binary cohort.',
                         'assays':'Censoring and dual concentrations preserved; MIC and disk never pooled or used as X.',
                         'groups':'Connected components of BioSample and recorded SNP-cluster links; absent cluster falls back to BioSample.',
                         'genes':'Unique symbol=status annotations, not copy number; unavailable distinguished from completed zero-hit analysis.',
                         'invalid_measurements':'Flagged and separately quarantined for quantitative use; source phenotypes remain usable because measurements never enter X.'}}
    (audit/'schema_audit.json').write_text(json.dumps(schema,indent=2))
    (root/'results/preprocessing_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args()
    preprocess(a.root)
