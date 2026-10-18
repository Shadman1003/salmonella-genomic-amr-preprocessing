"""Build and execute the research walkthrough from already-generated artifacts.

This does not redownload data, run preprocessing, fit models or rerun benchmarks.
Run with /tmp/shadman_amr_prime_venv/bin/python in WSL. A temporary python3 kernel spec
pins notebook execution to the interpreter running this script.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def normalized(source):
    return textwrap.dedent(source).strip() + '\n'


def build(root):
    md = lambda s: nbformat.v4.new_markdown_cell(normalized(s))
    code = lambda s: nbformat.v4.new_code_cell(normalized(s))
    cells = [
        md(r'''
        # Internship Research Prime
        ## Auditable preprocessing and resistance prediction in Salmonella

        **Question.** How do missing-value handling and categorical encoding affect
        prediction of source-reported antibiotic resistance from isolate metadata
        and existing AMRFinderPlus annotations?

        The primary task is **ampicillin resistant versus susceptible**. Tetracycline
        and ciprofloxacin are secondary. The fixed main configuration is
        **onehot_median_flags**; controlled ablations hold logistic regression fixed.
        Recorded SNP/BioSample components are separated before data-adaptive operations.

        The frozen source contains **142,338 assays, 9,977 assembled isolate records,
        and 9,976 BioSamples**. These are different denominators. This notebook loads
        and inspects the delivered artifacts, reproduces small transformation examples,
        and checks frozen-model inference. It **does not rerun the full research pipeline**.
        This is computational methodology work using public observations, with no
        claim of novel gene discovery, representative prevalence or clinical validity.
        '''),
        code(r'''
        import sys, json, hashlib, platform
        from pathlib import Path
        from collections import Counter
        from io import BytesIO
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display, Image

        ROOT = Path.cwd()
        if not (ROOT / 'data/raw/acquisition_manifest.json').exists():
            ROOT = ROOT.parent
        assert (ROOT / 'src/preprocess.py').exists(), 'Run this notebook from the project or notebooks directory.'
        sys.path.insert(0, str(ROOT / 'src'))
        from preprocess import measurement, json_rows, digest

        manifest = json.loads((ROOT / 'data/raw/acquisition_manifest.json').read_text())
        prep = json.loads((ROOT / 'results/preprocessing_summary.json').read_text())
        model_dir = ROOT / 'results/model_study'
        study = json.loads((model_dir / 'summary.json').read_text())
        source_checks = []
        for item in manifest['downloads']:
            actual = digest(ROOT / item['file'])
            assert actual == item['sha256'], f"Raw file changed: {item['file']}"
            source_checks.append({'collection': item['collection'], 'records': item['records'],
                                  'bytes': item['bytes'], 'sha256_verified': actual == item['sha256']})
        print('Interpreter:', sys.executable, '| Python:', platform.python_version())
        print('Frozen acquisition UTC:', manifest['started_at_utc'], 'to', manifest['completed_at_utc'])
        display(pd.DataFrame(source_checks))
        print('Raw SHA256 checks passed. Sorted IDs and unchanged counts do not imply a transactional live snapshot.')
        '''),
        md(r'''
        ### 1. Source structure and units of analysis

        NCBI's [AST documentation](https://www.ncbi.nlm.nih.gov/pathogens/docs/ast/)
        defines an AST row as an individual laboratory test. The linked isolate table
        supplies assembly summaries, annotations and collection metadata. The public
        source labels receive basic quality checks, rather than full method/value review
        ([source limitations](https://www.ncbi.nlm.nih.gov/pathogens/ast/)).

        We retain the source ID, check BioSample/assembly agreement, preserve repeated
        assays, and form resolved **BioSample-antibiotic** targets. Repeated antibiotics
        are useful long-format data; repeated labels are not independent votes.
        The one BioSample linked to two assemblies is conservatively quarantined.
        '''),
        code(r'''
        schema = json.loads((ROOT / 'results/audit/schema_audit.json').read_text())
        schema_rows = []
        selected_fields = {'id','biosample_acc','target_acc','antibiotic','phenotype','mic','mic_secondary',
                           'disk_diffusion','measurement_sign','platform','host','AMR_genotypes',
                           'AMR_genotypes_core','amrfinderplus_applied','collection_date','species_taxid'}
        for collection, info in schema.items():
            for field, info_field in info['fields'].items():
                if field in selected_fields:
                    schema_rows.append(dict(collection=collection, field=field, **info_field))
        display(pd.DataFrame(schema_rows))
        raw_biosamples, raw_targets, raw_drugs = set(), set(), set()
        for row in json_rows(ROOT / 'data/raw/ncbi_salmonella_ast.jsonl'):
            raw_biosamples.add(row['biosample_acc']); raw_targets.add(row['target_acc']); raw_drugs.add(row['antibiotic'])
        assert (len(raw_targets), len(raw_biosamples), len(raw_drugs)) == (9977, 9976, 72)
        features = pd.read_csv(ROOT / 'data/processed/isolate_features.csv.gz')
        pairs = pd.read_csv(ROOT / 'data/processed/phenotype_pairs.csv.gz')
        display(pd.DataFrame({'unit': ['Raw AST observations','Raw assembly records','Raw BioSamples',
                                      'Retained unique BioSamples','Resolved BioSample-drug pairs'],
                              'count': [prep['raw_ast_rows'], prep['raw_isolate_records'], len(raw_biosamples),
                                        len(features), len(pairs)]}))
        drug_denominators = pairs.groupby('antibiotic').agg(
            biosample_drug_pairs=('biosample_acc','size'),
            eligible_binary_pairs=('status',lambda x: x.eq('eligible_binary').sum()))
        display(drug_denominators.loc[['ampicillin','tetracycline','ciprofloxacin']])
        '''),
        md(r'''
        ### 2. Preserve assay semantics before building labels

        MIC **<=4 mg/L** is the interval **(0,4]**, while **>8 mg/L** has an open
        lower boundary. A combination such as **32/4** contains two reported
        concentrations; dividing them changes the meaning. The source comparator
        is not independently invented for the second component. Disk diffusion
        uses millimetres and remains separate from MIC.

        The small examples below test transformation semantics. The following
        platform disagreement comes from actual raw observations and is quarantined
        for the binary target rather than resolved by a guessed breakpoint.
        '''),
        code(r'''
        examples = []
        for value, sign, kind in [('4','<=','mic'), ('8','>','mic'), ('32/4','==','mic'),
                                  ('20','==','disk'), (None,'==','mic')]:
            parsed = measurement(value, sign, kind)
            examples.append({'example': f'{kind} {sign} {value}', 'status': parsed[kind+'_status'],
                             'lower': parsed[kind+'_1_lower'], 'upper': parsed[kind+'_1_upper'],
                             'lower_closed': parsed[kind+'_1_lower_closed'],
                             'upper_closed': parsed[kind+'_1_upper_closed'],
                             'components': parsed[kind+'_components']})
        display(pd.DataFrame(examples))
        conflict = [r for r in json_rows(ROOT / 'data/raw/ncbi_salmonella_ast.jsonl')
                    if r['target_acc']=='PDT000555592.1' and r['antibiotic']=='piperacillin-tazobactam']
        display(pd.DataFrame(conflict)[['id','biosample_acc','antibiotic','phenotype',
                                        'measurement_sign','mic','mic_secondary','standard','platform']])
        assert {r['phenotype'] for r in conflict} == {'susceptible','resistant'}
        display(pd.read_csv(ROOT / 'results/audit/censoring_counts.csv'))
        print('Combination-MIC assays:', prep['combination_mic_assays'],
              '| Censored measurements:', prep['censored_measurements'])
        '''),
        md(r'''
        ### 3. Missingness and class encoding have different meanings

        Source-resistant maps to **1** and source-susceptible to **0**. Intermediate,
        nonsusceptible, undefined, missing and contradictory labels stay in the audit,
        outside the primary binary task. Untested antibiotics are never called susceptible.

        Missing laboratory MIC in a disk assay is structural. Missing host/source
        metadata is unrecorded context. A completed zero-hit gene analysis is different
        from unavailable analysis; all raw isolates here have an applied analysis.
        The retained source has **40 missing collection years** and complete numerical
        assembly-quality fields. Numerical imputation is therefore a limited natural
        missingness problem here, rather than a reason to invent defects in the data.
        '''),
        code(r'''
        display(pd.DataFrame([{'source_label':'susceptible','binary_target':0,'policy':'eligible if repeats agree'},
                              {'source_label':'resistant','binary_target':1,'policy':'eligible if repeats agree'},
                              {'source_label':'intermediate / nonsusceptible / undefined','binary_target':None,
                               'policy':'audit; excluded from primary binary target'}]))
        display(pairs.groupby('status').size().rename('pairs').reset_index())
        display(pd.read_csv(ROOT / 'results/audit/phenotype_pair_quarantine.csv.gz').head(8))
        missing = pd.read_csv(ROOT / 'results/audit/isolate_feature_missingness.csv')
        focus = ['host','source_type','serovar','collection_year','asm_n50','asm_length','asm_contigs']
        display(missing[missing['field'].isin(focus)])
        assert int(features['collection_year'].isna().sum()) == 40
        assert features[['asm_n50','asm_length','asm_contigs']].notna().all().all()
        table = missing.set_index('field').loc[focus,'missing_percent']
        fig, ax = plt.subplots(figsize=(8,3.4))
        table.plot.barh(ax=ax, color='#167d8d')
        ax.set(xlabel='Missing values among retained BioSamples (%)', ylabel='',
               title='Observed metadata missingness; assembly summaries are complete')
        fig.tight_layout(); buffer=BytesIO(); fig.savefig(buffer,format='png',dpi=130)
        display(Image(data=buffer.getvalue())); plt.close(fig)
        '''),
        md(r'''
        ### 4. Keep related isolates together before learning transformations

        The primary split assigns recorded SNP/BioSample connected components to
        training, validation and test using fixed seeds. Group proportions are
        approximately 60/20/20; unequal group sizes can make isolate proportions
        differ. Missing SNP links fall back to BioSample grouping, and are not proof
        of biological independence.

        MIC, disk measurements, AST-derived counts and source phenotype summaries
        are excluded from predictors. Identifiers serve linkage and split audit.
        Imputers, scalers, category vocabularies and the gene vocabulary learn from
        training only, including separately within the group-aware CV folds.
        '''),
        code(r'''
        splits = pd.read_csv(model_dir / 'split_manifest.csv.gz')
        split_sets = {s:set(splits.loc[splits['split'].eq(s),'group_id']) for s in ['train','validation','test']}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not split_sets[a].intersection(split_sets[b]), f'Component overlap: {a}/{b}'
        assert splits['biosample_acc'].is_unique
        display(splits.groupby('split').agg(biosamples=('biosample_acc','size'), groups=('group_id','nunique')))
        display(pd.DataFrame([{'drug':drug,'split':split,**info}
                              for drug,task in study['tasks'].items() for split,info in task['splits'].items()]))
        print('Recorded SNP/BioSample components:', prep['connected_groups'],
              '| largest retained component:', prep['largest_group_isolates'])
        print(study['protocol'])
        '''),
        md(r'''
        ### 5. Inspect fitted encodings instead of inventing integer meanings

        Nominal categories have no natural numerical spacing. Ordinal codes are an
        explicit comparison baseline, while one-hot encoding assigns separate columns.
        A categorical missing token records missingness without guessing the value.
        Each arm uses its fitted training vocabulary; the saved maps state unseen-value
        behavior. Gene annotations are **symbol=status multi-hot presence** features,
        not gene copy number or newly inferred sequence mechanisms.

        Median, median plus missing indicators, and single deterministic iterative
        imputation are controlled alternatives. Iterative imputation is not pooled
        multiple-imputation inference, and additional complexity need not improve results.
        '''),
        code(r'''
        import joblib
        import model_study as modeling  # Import custom GeneEncoder before loading the frozen joblib pipeline.
        from model_study import feature_columns, CATEGORICAL, BASE_NUMERIC, GENOMIC_NUMERIC
        primary_name = study['primary_model']
        encoding_maps = json.loads((model_dir / 'encoding_maps.json').read_text())
        mapping = encoding_maps[primary_name]
        forbidden = {'phenotype','AST_phenotypes','mic','disk_diffusion','number_drugs_resistant',
                     'number_drugs_susceptible','number_drugs_tested','biosample_acc','target_acc','group_id','bioproject_acc'}
        assert not set(feature_columns('combined')).intersection(forbidden)
        print('Predictor columns:', feature_columns('combined'))
        display(pd.DataFrame([{'feature':key,'training_categories':len(values),'first_categories':list(values)[:6]}
                              for key,values in mapping['categorical_maps'].items()]))
        print('Unknown-category policy:', mapping['unknown_category_policy'])
        print('Gene vocabulary size:',len(mapping['gene_vocabulary']), '| policy:', mapping['unknown_gene_policy'])
        print('Numeric imputation columns:',mapping['numeric_columns'])
        print('Numeric indicator indices:',mapping['indicator_feature_indices'])
        display(pd.DataFrame(study.get('unseen_gene_tokens',{})).T)
        integer_export = ROOT / 'data/processed/ampicillin_label_encoded_train.csv.gz'
        if integer_export.exists():
            example = pd.read_csv(integer_export,nrows=5)
            print('Explicit unscaled integer-label export (training vocabulary only):')
            display(example[['biosample_acc','y_resistant','source_type_label','country_label',
                             'host_label','serovar_label','collection_year_imputed','collection_year_was_missing']])
            print('Maps and unknown code:',ROOT / 'results/label_encoding_export.json')
        '''),
        md(r'''
        ### 6. Read performance using the declared task, split and threshold

        Ampicillin compares six preprocessing combinations, plus metadata-only and
        genes-only controls. The primary model remains fixed rather than chosen after
        inspecting the test table. Secondary drugs use the declared main configuration
        and controls. Read support, resistance recall and precision alongside average
        precision (AP), ROC AUC, balanced accuracy and calibration.

        The table below selects the **fixed 0.5** threshold. Validation-selected
        thresholds are separately labeled in the results CSV. AP is threshold-free;
        its reference level depends on resistance frequency. Confusion counts and
        calibration reveal behavior that an accuracy percentage can conceal.
        '''),
        code(r'''
        metrics = pd.read_csv(model_dir / 'metrics.csv')
        fixed = metrics[metrics['split'].eq('test') & metrics['threshold_policy'].eq('fixed_0.5')]
        columns = ['drug','model','n','n_resistant','average_precision','roc_auc','balanced_accuracy',
                   'resistance_recall','precision','f1','brier','accuracy']
        display(fixed[columns].round(4))
        assert {'metadata_only','genes_only',primary_name}.issubset(set(fixed['model']))
        display(Image(filename=str(model_dir / 'figures/ampicillin_ablation_average_precision.png')))
        display(Image(filename=str(model_dir / 'figures/ampicillin_diagnostics.png')))
        print('Actual measured results; diagnostics above use the frozen main model and declared threshold.')
        '''),
        md(r'''
        ### 7. Quantify uncertainty and challenge the split

        Training-fold group-aware CV and paired resampling of test groups address
        recorded dependence. Bootstrap confidence intervals are conditional on fitted
        models and the frozen split: they do not capture all training uncertainty or
        repair noisy laboratory labels.

        A BioSample-random diagnostic may place clonal relatives across the boundary.
        BioProject-component holdout probes submission/project shifts and can have
        difficult class support. These analyses answer different questions; none is
        automatic proof of external, temporal or clinical validity. A confidence
        interval crossing zero in a paired difference does not support a clear gain.
        '''),
        code(r'''
        cv = pd.read_csv(model_dir / 'cv_metrics.csv')
        ci = pd.read_csv(model_dir / 'bootstrap_ci.csv')
        differences = pd.read_csv(model_dir / 'paired_differences.csv')
        sensitivities = pd.read_csv(model_dir / 'split_sensitivities.csv')
        display(cv.round(4))
        display(ci[ci['metric'].isin(['average_precision','resistance_recall'])].round(4))
        display(differences[differences['metric'].eq('average_precision')].round(4))
        display(sensitivities.round(4))
        print('Bootstrap design:', study['bootstrap'])
        print('Warnings recorded by the actual run:', {k:v for k,v in study.get('warnings',{}).items() if v})
        '''),
        md(r'''
        ### 8. Missingness stress experiments and processing scale

        Natural numerical missingness is limited here. A separately declared stress
        experiment masks **observed validation cells** under simulated MCAR and
        assembly-contig-conditioned MAR mechanisms, without target/test-set use.
        Reconstruction error on those known values cannot establish the truth of
        naturally missing values or identify the source's real missingness mechanism.

        The streaming benchmark processes real ordered source prefixes and the full
        collection with different chunk sizes. Equal canonical hashes check semantic
        equivalence. It measures row normalization/measurement parsing, not full joins,
        training, distributed Spark performance or a larger biological sample size.
        Single-run timing depends on startup, cache and run order.
        '''),
        code(r'''
        stress_path = ROOT / 'results/imputation_stress.csv'
        if stress_path.exists():
            stress = pd.read_csv(stress_path)
            design = json.loads((ROOT / 'results/imputation_stress_design.json').read_text())
            print('Stress design:', design)
            display(stress[['mechanism','method','feature','observed_validation_cells_masked',
                            'rmse_original_units','rmse_over_training_iqr']].round(4))
        else:
            print('Optional imputation stress results are not present; no values have been invented.')
        benchmark_path = ROOT / 'results/streaming_benchmark.json'
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text())
            assert benchmark['all_checks_passed']
            display(pd.DataFrame(benchmark['runs'])[['requested_prefix_rows','chunk_size','records_processed',
                                                     'transform_seconds','process_wall_seconds','peak_rss_mib']].round(3))
            display(pd.DataFrame(benchmark['chunk_consistency']))
            print('Benchmark scope:', benchmark['scope'])
        else:
            print('Optional streaming benchmark has not been run; no extrapolated timing is shown.')
        '''),
        md(r'''
        ### 9. Check the frozen inference path

        Loading the model also loads its learned preprocessing. We import the custom
        encoder module from this project's src directory before deserialization, and
        predict a small set of already-held-out rows. Stored and reloaded probabilities
        should agree. This check does not refit or improve the model.

        A separate perturbed probe checks supported behavior for unseen categories,
        unknown gene tokens and missing numerical values. Its output is a software
        robustness test on altered inputs, not additional experimental performance.
        '''),
        code(r'''
        model = joblib.load(model_dir / f'models/ampicillin_{primary_name}.joblib')
        stored = pd.read_csv(model_dir / 'predictions.csv.gz')
        heldout = stored[stored['drug'].eq('ampicillin') & stored['model'].eq(primary_name)
                         & stored['split'].eq('test')].head(6).copy()
        probe = heldout[['biosample_acc']].merge(features,on='biosample_acc',validate='one_to_one')
        probability = model.predict_proba(probe[feature_columns('combined')])[:,1]
        assert np.allclose(probability,heldout['probability'].to_numpy(),atol=1e-10,rtol=0)
        heldout['reloaded_probability'] = probability
        display(heldout[['biosample_acc','y','probability','reloaded_probability']].round(7))
        perturbed = probe.copy()
        perturbed.loc[:,CATEGORICAL] = '__UNSEEN_NOTEBOOK_PROBE__'
        for column in BASE_NUMERIC + GENOMIC_NUMERIC:
            perturbed[column] = np.nan
        perturbed['gene_tokens'] = json.dumps(['__UNSEEN_NOTEBOOK_GENE__=COMPLETE'])
        assert np.isfinite(model.predict_proba(perturbed[feature_columns('combined')])).all()
        print('Frozen predictions agree; unknown/missing-input software probe returned finite probabilities.')
        '''),
        md(r'''
        ### 10. What the evidence can and cannot support

        The data are selected surveillance/research submissions with incomplete
        source/host fields. Laboratory standards and editions are not fully harmonized;
        source S/I/R labels are preserved rather than recomputed using guessed
        breakpoints. Recorded clusters do not encode all phylogenetic relationships.
        Existing AMR annotations contain prior biological knowledge and general efflux
        calls, so predictive performance is not novel mechanism discovery or causality.
        Small/uncertain ablation differences are valid findings. An external source
        and an independently specified follow-up protocol would strengthen generalization.

        **Reproduce the full pipeline in WSL from the project directory:**
        ```bash
        bash setup_wsl.sh && bash run_wsl.sh
        ```
        **Re-execute this walkthrough after those artifacts exist:**
        ```bash
        /tmp/shadman_amr_prime_venv/bin/python src/build_notebook.py --project-root .
        ```
        Redownload is a separate opt-in operation: `bash download_wsl.sh`. It refuses
        to replace the frozen snapshot. Exact source URLs, timestamps and hashes are
        retained in the acquisition manifest. The initial downloaded data and earlier
        Internship_Research project are preserved. Read README.txt and STUDY_GUIDE.txt
        for source-use policies, AI-assistance disclosure and the three-day study plan.
        '''),
        code(r'''
        validation = json.loads((ROOT / 'results/preprocessing_validation.json').read_text())
        assert validation['all_checks_passed']
        assert all(source['sha256_verified'] for source in source_checks)
        assert study.get('inference_checks',{}).get('serialized_predictions_match')
        display(pd.DataFrame(validation['checks'])[['check','passed']])
        print('Walkthrough checks complete:', {'source_hashes':True, 'recorded_split_overlap':0,
              'preprocessing_validation':validation['all_checks_passed'], 'frozen_inference_match':True,
              'full_pipeline_rerun_inside_notebook':False})
        print('Use results/model_study/summary.json for actual support, packages, warnings and limitations.')
        '''),
    ]
    assert len(cells) == 22
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        'kernelspec': {'display_name':'Python 3 (Research Prime)', 'language':'python', 'name':'python3'},
        'language_info': {'name':'python', 'version':platform_version()},
        'research_prime_scope': 'Executed artifact walkthrough; no data acquisition, model fitting or full preprocessing rerun.',
    })
    return notebook


def platform_version():
    import platform
    return platform.python_version()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--write-only',action='store_true',help='Prepare unexecuted notebook while experiment artifacts are still being generated.')
    parser.add_argument('--timeout',type=int,default=180,help='Per-cell execution timeout in seconds.')
    args=parser.parse_args()
    root=args.project_root.resolve()
    notebook=build(root)
    destination=root/'notebooks/Research_Walkthrough.ipynb'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if args.write_only:
        notebook.metadata['research_prime_qa']={'status':'prepared_unexecuted','code_cells':11}
        nbformat.write(notebook,destination)
        print(json.dumps({'status':'prepared_unexecuted','file':str(destination),'cells':22,'code_cells':11}))
        return
    required=[
        'data/raw/acquisition_manifest.json','results/preprocessing_summary.json',
        'results/preprocessing_validation.json','results/model_study/summary.json',
        'results/model_study/metrics.csv','results/model_study/cv_metrics.csv',
        'results/model_study/bootstrap_ci.csv','results/model_study/paired_differences.csv',
        'results/model_study/split_sensitivities.csv','results/model_study/encoding_maps.json',
        'results/model_study/predictions.csv.gz',
        'results/model_study/figures/ampicillin_ablation_average_precision.png',
        'results/model_study/figures/ampicillin_diagnostics.png',
    ]
    absent=[name for name in required if not (root/name).exists()]
    if absent:
        raise FileNotFoundError('Finish pipeline artifacts before executing walkthrough: '+', '.join(absent))
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='research-prime-kernel-') as temp:
        kernels=Path(temp)/'kernels'; spec=kernels/'python3';spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({
            'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Python 3 (Research Prime)', 'language':'python',
        }),encoding='utf-8')
        manager=KernelManager(kernel_name='python3',
                              kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        client=NotebookClient(notebook,km=manager,timeout=args.timeout,kernel_name='python3',
                              resources={'metadata':{'path':str(root)}})
        try:
            client.execute(cwd=str(root))
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    code_cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    errors=[output for cell in code_cells for output in cell.get('outputs',[]) if output.output_type=='error']
    assert len(code_cells)==11 and not errors
    assert [cell.execution_count for cell in code_cells] == list(range(1,12))
    images=sum('image/png' in output.get('data',{}) for cell in code_cells for output in cell.get('outputs',[]))
    assert images==3, f'Expected three embedded figures; found {images}'
    notebook.metadata['research_prime_qa']={
        'status':'executed_no_errors','validated_at_utc':datetime.now(timezone.utc).isoformat(),
        'cells':len(notebook.cells),'code_cells':len(code_cells),'executed_code_cells':len(code_cells),
        'error_outputs':len(errors),'embedded_png_outputs':images,'python_executable':sys.executable,
        'kernel_name':'python3','elapsed_seconds':time.perf_counter()-started,
        'full_pipeline_rerun_inside_notebook':False,
        'required_artifact_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in required},
    }
    nbformat.validate(notebook)
    nbformat.write(notebook,destination)
    print(json.dumps({'file':str(destination),**notebook.metadata['research_prime_qa']},indent=2))


if __name__=='__main__':
    main()
