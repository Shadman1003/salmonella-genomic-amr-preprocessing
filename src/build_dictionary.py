"""Document every exported cleaned field and its model-use policy."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MEANINGS={
'biosample_acc':'Stable source BioSample string; biological sample join key, not a numeric feature.',
'target_acc':'Versioned NCBI PDT isolate/assembly accession; provenance only.',
'source_id':'Original unique AST server record ID; provenance and cell-audit join key.',
'antibiotic':'Conservative whitespace/Unicode/case-normalized source drug name; defines separate tasks.',
'phenotype':'Canonical source phenotype. Only unanimous resistant/susceptible labels define binary y.',
'phenotype_raw':'Source phenotype text retained before normalization; never a predictor.',
'standard':'Source testing standard (such as CLSI); assay provenance, not a model input.',
'platform':'Source AST platform; absent values preserved; provenance only.',
'vendor':'Source AST vendor; absent values preserved; provenance only.',
'reagent':'Source assay version/reagent; provenance only.',
'measurement_sign':'Source comparator; equality variants normalized to ==. Never a model input.',
'mic_secondary_raw':'Raw second drug concentration from API mic_secondary; not a ratio or independently censored value.',
'measurement_type':'mic/disk/both/missing. MIC mg/L and disk mm retain separate measurement domains.',
'measurement_invalid':'Any failed parser/domain check, unknown sign, or both measurement types; quantitative quarantine flag.',
'label_recognized':'Source phenotype belongs to the documented expected vocabulary (or is absent).',
'metadata_matched':'BioSample has exactly one retained isolate-feature row.',
'target_mismatch':'AST PDT differs from the unambiguous metadata PDT for its BioSample.',
'exact_duplicate':'Repeated identical assay payload after first occurrence; source IDs do not establish biological duplication.',
'bioproject_acc':'Source project accession(s); grouping sensitivity/provenance only, never X.',
'snp_cluster':'Recorded NCBI PDS cluster accession; known relatedness grouping only, never X.',
'clustered':'Original source clustered flag; audit/provenance, not a predictor.',
'source_type':'Source harmonized broad source category; nominal model metadata.',
'epi_type':'NCBI epidemiological isolation type; nominal model metadata.',
'country':'Country-level prefix of geo_loc_name before colon; nominal model metadata.',
'host':'Conservatively normalized submitted host text; no invented host or gender mapping.',
'serovar':'Normalized reported serovar/serotype; nominal model metadata, not an ordered integer quantity.',
'collection_year':'Unambiguous collection year from strict partial-date parser. Missing/range ambiguity stays missing until train-fitted imputation.',
'date_precision':'Original collection date precision category; audit only.',
'amr_analysis_available':'1 if applied flag plus reported call list or explicit zero count establishes completed analysis; otherwise0.',
'gene_count':'Number of unique reported symbol=status tokens after completed analysis; not gene copies. Missing if unavailable.',
'gene_tokens':'JSON string array of unique AMR symbol=status annotations. Train-only vocabulary multi-hot X.',
'amrfinderplus_version':'Source annotation tool version; audit/provenance only.',
'refgene_db_version':'Source reference database version; audit/provenance only.',
'analysis_type':'Source annotation mode (COMBINED/NUCLEOTIDE etc.); provenance only.',
'species_taxid':'NCBI species taxid; required28901 in this frozen cohort; provenance/cohort check.',
'asm_n50':'Assembly contig N50 in base pairs; positive with N50<=assembly length; numeric model metadata.',
'asm_length':'Total assembly length in base pairs; positive numeric model metadata.',
'asm_contigs':'Assembly contig count; positive numeric model metadata.',
'group_id':'Deterministic connected BioSample/PDS component. Split unit, never X.',
'status':'Pair eligibility/quarantine disposition: eligible_binary, nonbinary, conflicting, unmatched_metadata, etc.',
'all_labels':'JSON list of all distinct nonmissing source labels for a BioSample-drug pair.',
'assay_count':'Number of retained nondeduplicated source assays for this BioSample-drug pair; audit only.',
'standards':'JSON list of source standards within a pair; audit only.',
'source_ids':'JSON list of all AST source records supporting a pair; audit only.',
}
NUMERIC={'collection_year','asm_n50','asm_length','asm_contigs','amr_analysis_available','gene_count'}
CATEGORICAL={'source_type','epi_type','country','host','serovar'}
def run():
    rows=[]
    for name in ('ast_clean','isolate_features','phenotype_pairs'):
        frame=pd.read_csv(ROOT/f'data/processed/{name}.csv.gz',nrows=10)
        for col in frame.columns:
            description=MEANINGS.get(col)
            if description is None and col.startswith(('mic_','disk_')):
                kind='MIC (mg/L)' if col.startswith('mic_') else 'disk zone (mm)'
                if col.endswith('_raw'): description=f'Raw submitted {kind} value; combination components separated by slash are not divided.'
                elif col.endswith('_status'): description=f'{kind} parser/domain state. structural_missing means that measurement type is absent, not a value to impute.'
                elif col.endswith('_components'): description=f'Number of supplied {kind} concentration/measurement components.'
                elif col.endswith('_censored'): description=f'True if a non-equality comparator constrains the primary {kind} component.'
                elif col.endswith('_secondary_semantics'): description='Second component records its nominal supplied concentration; no independent source comparator is inferred.'
                elif col.endswith(('_lower_closed','_upper_closed')): description=f'Whether the corresponding bound is included. Interpret only together with valid {kind} status/comparator.'
                elif col.endswith('_lower'): description=f'Lower {kind} bound. Missing for absent/invalid measurements; primary lower0 for < or <= MIC.'
                elif col.endswith('_upper'): description=f'Upper {kind} bound. For a valid > or >= primary measurement, missing means no finite upper bound; absent assays also have missing bounds.'
            if description is None: raise ValueError(f'Undocumented field {name}/{col}')
            role=('numeric X' if col in NUMERIC else 'nominal X' if col in CATEGORICAL else 'multi-hot genomic X' if col=='gene_tokens' else 'defines y for eligible pair' if name=='phenotype_pairs' and col=='phenotype' else 'audit/key/provenance; excluded from X')
            if name!='isolate_features' and role.endswith('X'): role='assay/pair audit; excluded from X'
            rows.append({'table':name,'column':col,'model_role':role,'description':description})
    pd.DataFrame(rows).to_csv(ROOT/'DATA_DICTIONARY.csv',index=False)
    print(f'Documented {len(rows)} exported fields')
if __name__=='__main__':run()
