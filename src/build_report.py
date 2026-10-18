"""Build a research report entirely from executed result artifacts.

Run with the bundled document Python (reportlab/pandas/matplotlib). The scientific
analysis itself runs in WSL. No result is hardcoded into the narrative.
"""
from pathlib import Path
import json, html, textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'; MOD=RES/'model_study'; FIG=RES/'figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def fmt(x,n=3):
    if pd.isna(x): return 'NA'
    return f'{float(x):.{n}f}'
def num(x): return f'{int(x):,}'
def esc(x): return html.escape(str(x))

def figures(pre):
    missing=pd.read_csv(RES/'audit/isolate_feature_missingness.csv')
    missing=missing[missing.missing_percent.gt(0)].sort_values('missing_percent').tail(10)
    fig,ax=plt.subplots(figsize=(8.6,3.6),layout='constrained')
    ax.barh(missing.field,missing.missing_percent,color='#167e86'); ax.set(xlabel='Missing (%) before learned imputation',xlim=(0,100),title='Missing isolate metadata')
    fig.savefig(FIG/'metadata_missingness.png',dpi=180); plt.close(fig)
    signs=pd.Series(pre['comparators']).sort_values()
    fig,ax=plt.subplots(figsize=(8.6,3.5),layout='constrained')
    ax.barh(signs.index,signs.values,color='#167e86'); ax.set(xlabel='Assay results',title='Source measurement comparators (equality normalized)')
    for i,x in enumerate(signs.values): ax.text(x+500,i,num(x),va='center',fontsize=9)
    ax.set_xlim(0,signs.max()*1.18); fig.savefig(FIG/'measurement_comparators.png',dpi=180); plt.close(fig)
    bench=read('results/streaming_benchmark.json')
    runs=pd.DataFrame(bench['runs']); full=runs[runs.requested_prefix_rows.eq('full')]
    fig,axs=plt.subplots(1,2,figsize=(8.6,3.2),layout='constrained')
    for ax,col,label in zip(axs,['transform_seconds','peak_rss_mib'],['Transform time (seconds)','Peak process RSS (MiB)']):
        ax.bar(full.chunk_size.astype(str),full[col],color='#167e86'); ax.set(xlabel='Chunk size (real rows)',ylabel=label)
    fig.suptitle('Single-process full-cohort streaming benchmark')
    fig.savefig(FIG/'streaming_benchmark.png',dpi=180); plt.close(fig)
    stress=pd.read_csv(RES/'imputation_stress.csv')
    stress=stress[stress.mechanism.eq('MCAR')&stress.nominal_mask_rate.eq(.3)]
    fig,ax=plt.subplots(figsize=(8.6,3.2),layout='constrained')
    stress.pivot(index='feature',columns='method',values='rmse_over_training_iqr').plot.bar(ax=ax,color=['#167e86','#d6974f'])
    ax.set(xlabel='',ylabel='RMSE / training IQR',title='Synthetic MCAR 30%: observed validation cells only'); ax.tick_params(axis='x',rotation=15)
    ax.legend(fontsize=8); fig.savefig(FIG/'imputation_stress.png',dpi=180); plt.close(fig)

def build():
    pre=read('results/preprocessing_summary.json'); model=read('results/model_study/summary.json')
    metrics=pd.read_csv(MOD/'metrics.csv'); cv=pd.read_csv(MOD/'cv_metrics.csv'); ci=pd.read_csv(MOD/'bootstrap_ci.csv')
    diffs=pd.read_csv(MOD/'paired_differences.csv'); sens=pd.read_csv(MOD/'split_sensitivities.csv')
    primary_name=model['primary_model']
    fixed=metrics[metrics.split.eq('test')&metrics.threshold_policy.eq('fixed_0.5')]
    primary=fixed[fixed.drug.eq('ampicillin')&fixed.model.eq(primary_name)].iloc[0]
    task=model['tasks']['ampicillin']; figures(pre)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CoverTitle',fontName='Helvetica-Bold',fontSize=27,leading=32,textColor=colors.HexColor('#12333e'),spaceAfter=18))
    styles.add(ParagraphStyle(name='SectionPrime',fontName='Helvetica-Bold',fontSize=17,leading=21,textColor=colors.HexColor('#126b75'),spaceAfter=13))
    styles.add(ParagraphStyle(name='BodyPrime',fontName='Helvetica',fontSize=10.2,leading=14.4,spaceAfter=10))
    styles.add(ParagraphStyle(name='SmallPrime',fontName='Helvetica',fontSize=8.3,leading=11,spaceAfter=7))
    styles.add(ParagraphStyle(name='TablePrime',fontName='Helvetica',fontSize=8,leading=10))
    story=[]; transcript=[]
    def p(text,small=False): story.append(Paragraph(text,styles['SmallPrime' if small else 'BodyPrime'])); transcript.append(text)
    def title(text): story.append(Paragraph(text,styles['SectionPrime'])); transcript.append('\n'+text+'\n')
    def page(text): story.append(PageBreak()); title(text)
    def table(headers,rows,widths=None):
        data=[[Paragraph(esc(v),styles['TablePrime']) for v in headers]]+[[Paragraph(esc(v),styles['TablePrime']) for v in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2eff0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#126b75')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8f9')]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([t,Spacer(1,12)]); transcript.extend([' | '.join(map(str,headers))]+[' | '.join(map(str,row)) for row in rows])
    def pic(path,width=490):
        from PIL import Image as PILImage
        path=Path(path); w,h=PILImage.open(path).size
        story.extend([Image(str(path),width=width,height=width*h/w),Spacer(1,10)])
    def ref(no,label,url): return f'[{no}] <link href="{url}" color="#126b75">{label}</link>'
    sources=[('NCBI AST Browser documentation','https://www.ncbi.nlm.nih.gov/pathogens/docs/ast/'),('NCBI BioSample antibiogram specification','https://www.ncbi.nlm.nih.gov/biosample/docs/antibiogram/'),('NCBI Pathogen Detection help','https://www.ncbi.nlm.nih.gov/pathogens/pathogens_help/'),('AMRFinderPlus: interpreting results','https://github.com/ncbi/amr/wiki/Interpreting-results'),('scikit-learn: grouped cross-validation','https://scikit-learn.org/stable/modules/cross_validation.html'),('scikit-learn: missing-value imputation','https://scikit-learn.org/stable/modules/impute.html'),('scikit-learn: categorical preprocessing','https://scikit-learn.org/stable/modules/preprocessing.html'),('scikit-learn: common pitfalls and leakage','https://scikit-learn.org/stable/common_pitfalls.html'),('NCBI Pathogen Detection FAQ','https://www.ncbi.nlm.nih.gov/pathogens/faq/')]
    sources += [('Feldgarden et al. (2021). AMRFinderPlus and the Reference Gene Catalog. Scientific Reports 11:12728. doi:10.1038/s41598-021-91456-0','https://www.nature.com/articles/s41598-021-91456-0'),('Yu, Wheeler and Barquist (2025). Biased sampling driven by bacterial population structure confounds machine learning prediction of antimicrobial resistance. PLOS Biology 23:e3003539. doi:10.1371/journal.pbio.3003539','https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003539')]
    story.append(Spacer(1,24)); p('SHADMAN / COMPUTER SCIENCE AND ENGINEERING / DATA SCIENCE',True)
    story.append(Paragraph('Machine Learning-Based Analysis and Prediction of Antibiotic Resistance',styles['CoverTitle']))
    p('<b>Prime study:</b> Auditable preprocessing, missing values, and categorical encoding for Salmonella phenotype and genomic-annotation data.')
    p('Executed 18 October 2026 | WSL / Ubuntu | Reproducible undergraduate empirical study')
    title('Abstract')
    p(f'This study links {num(pre["raw_ast_rows"])} submitted antimicrobial susceptibility test (AST) results to {num(pre["raw_isolate_records"])} assembled isolate records. It evaluates cleaning and preprocessing as explicit, testable decisions. The workflow preserves censored assay intervals, separates MIC from disk measurements, quarantines conflicting labels and ambiguous joins, and encodes curated genomic annotations without using phenotype-derived predictors. Nominal encoding and numerical imputation are compared in a prespecified factorial experiment. Training, validation and test data are separated by recorded SNP/BioSample connected groups.')
    p(f'For the predeclared ampicillin model, held-out average precision was {fmt(primary.average_precision)}, AUROC {fmt(primary.roc_auc)}, and resistance recall {fmt(primary.resistance_recall)} at threshold 0.5. These are retrospective cohort results, not a clinical validation. The preprocessing audit and paired experiments establish what changed and what did not; complexity is not assumed to improve accuracy.')
    table(['Frozen inputs','Primary evaluation'],[[num(pre['raw_ast_rows'])+' AST results',num(primary.n)+' held-out ampicillin BioSamples'],[num(pre['raw_isolate_records'])+' isolate/assembly records',num(task['splits']['test']['groups'])+' held-out connected groups'],[num(pre['censored_measurements'])+' censored assays','Primary: one-hot + median + indicators']],widths=[245,245])
    p('Prepared with AI assistance for Shadman. All numerical results come from executed artifacts. Review and understand the methods before academic submission.',True)

    page('1 / Research question and contribution')
    p('The earlier project used a limited metadata snapshot. This study changes both the data source and the empirical design: experimental phenotypes are joined to isolate-specific genomic annotations, and alternative preprocessing policies are compared under fixed relatedness-aware splits. Its contribution is a reproducible preprocessing study, not a claim of a novel AMR algorithm.')
    table(['Question','Operational test'],[['What makes cleaning scientifically meaningful?','Audit source censoring, dual concentrations, repeated assays, discordant labels, metadata multiplicity and missingness.'],['Does integer encoding distort nominal predictors?','Compare ordinal integers with one-hot columns using the same numeric and genomic inputs.'],['Does numerical imputation help here?','Cross median / median+indicators / iterative+indicators with both categorical strategies.'],['How much signal comes from genomic annotations?','Compare combined, metadata-only and genomic-only pipelines.'],['Does the result depend on evaluation design?','Use connected-group holdout, grouped CV, BioProject and BioSample split diagnostics.'],['Can processing scale without changing outputs?','Compare canonical hashes and process memory at real row prefixes and chunk sizes.']],widths=[182,308])
    p('The study protocol in docs/STUDY_PROTOCOL.txt was fixed before model results were inspected. Ampicillin is primary; tetracycline and ciprofloxacin are prespecified secondary tasks. Other drug results remain in the cleaned data and do not choose the headline outcome. No hyperparameters or cleaning thresholds are selected from test scores.')
    p('This is analysis of public bacterial isolate records. There are no patient identifiers, prospective clinical decisions, new laboratory experiments, or raw-sequence assembly experiments in this work.')
    p('<b>Related empirical work.</b> Feldgarden et al. describe AMRFinderPlus reference curation, point-mutation detection, and evidence categories for annotation calls. This motivates retaining call status and treating the predictors as existing curated knowledge, rather than newly discovered genomic mechanisms. [10] Yu, Wheeler and Barquist demonstrate that bacterial population structure and biased sampling can confound AMR models, and that increasing sample size alone need not rescue performance. Their result motivates group-separated evaluation and source-transfer sensitivity here; recorded SNP groups are a practical approximation, not a reproduction of their full phylogenetic analysis. [11]')

    page('2 / Acquisition, denominators and provenance')
    p('NCBI stores submitted AST results separately from genomic annotations. A row represents an assay result, while an isolate can have many antibiotic results. The organism-group query and matching tested-isolate metadata were acquired with stable ID ordering; all retrieved species identifiers were verified as Salmonella enterica (28901). [1,3]')
    table(['Stage / denominator','Observed count'],[['Raw AST rows',num(pre['raw_ast_rows'])],['Raw assembled isolate records',num(pre['raw_isolate_records'])],['Unambiguous BioSample feature rows',num(pre['unique_biosamples_metadata'])],['BioSamples with ambiguous metadata, quarantined',num(pre['ambiguous_biosamples_quarantined'])],['Distinct BioSample-drug pairs after exact deduplication',num(pre['unique_biosample_drug_pairs'])],['Known SNP-cluster links retained',num(pre['snp_cluster_available_isolates'])],['Connected groups after conservative metadata cleaning',num(pre['connected_groups'])]],widths=[350,140])
    p('Raw JSONL files are immutable analytical inputs. The acquisition manifest records every request, page count, timestamp and SHA256. Initial and final counts agree and server IDs are unique and ascending. This verifies retrieval completeness under the observed query, but the live browser backend does not provide a transactional snapshot. The endpoint/schema can change; frozen files are the reproducible source.')
    p('NCBI surveillance and research submissions are a convenience cohort. Tested isolates may differ from untested isolates; dates and locations are unevenly reported. These data cannot establish population prevalence. AST row counts must never be presented as independent patient or isolate counts. [1,9]')
    p('Files: data/raw/acquisition_manifest.json; logs/raw_ast_profile.json; logs/raw_isolate_profile.json. All exclusions and transformations retain source accessions for traceability.',True)

    page('3 / Measurement-aware cleaning')
    pic(FIG/'measurement_comparators.png')
    p(f'The workflow identified {num(pre["censored_measurements"])} censored assays and {num(pre["combination_mic_assays"])} combination-MIC results. It retained {num(pre["measurement_types"].get("disk",0))} disk measurements separately from MIC. A missing MIC on a disk assay is structural missingness, not a concentration to estimate. [1,2]')
    p('For an MIC reported as &gt;16 mg/L, the valid information is a lower bound, not an exact value of 16. The cleaned schema stores bound values and open/closed endpoints. An absent upper endpoint means no finite upper bound was supplied. Combination values retain both concentrations; X/Y is never treated as division. The API mic_secondary field is explicitly preserved. The assay sign is not independently extrapolated to the second component.')
    p('MIC values are checked against the submission domain (positive, up to 1024 mg/L); disk diameters use 6-150 mm. Domain violations and unknown signs are flagged rather than mean-imputed or silently corrected. These are schema checks, not independently reconstructed clinical breakpoints. Current laboratory standards are not applied retrospectively to relabel historical tests. [2]')
    p(f'There were {num(pre["measurement_invalid_assays"])} quantitative measurement anomalies. Quantitative quarantine is separate from phenotype-label eligibility; the model study records its stricter flagged-measurement exclusion rule. No MIC, disk, sign, platform or AST-panel counts enter prediction features.')

    page('4 / Repeated assays, conflicts and join integrity')
    table(['Pair disposition','BioSample-drug pairs'],[[k,num(v)] for k,v in pre['pair_status_counts'].items()],widths=[350,140])
    p(f'Exact duplicate biological assay payloads: {num(pre["exact_duplicate_assays"])}. Multiple drugs per isolate are intentional long-format records. Compatible repeats remain auditable, while the analytical target is one BioSample-drug label. A pair is eligible only when known source labels are unanimous R or S, keys are valid, and the metadata join is unambiguous. Missing, intermediate, nonsusceptible and undefined labels are retained outside the binary target; they are never filled with a majority class.')
    p('A real conflict illustrates why this matters: SAMN11179044 has piperacillin-tazobactam results from Microscan (susceptible, &lt;=16/4) and Phoenix (resistant, &gt;64/4), both reported under CLSI. The workflow preserves both records and quarantines the conflicting pair. It does not decide which platform is correct or remove the isolate because its genes disagree with a phenotype.')
    p('A second edge case is BioSample SAMN03894186, associated with two current PDT records. The primary feature join excludes that ambiguous BioSample instead of multiplying outcomes over assemblies or selecting a record based on resistance. This conservative rule may discard usable data; both candidates remain in the audit for a more informed future resolution.')
    p(f'Unmatched AST records after this conservative join: {num(pre["unmatched_ast_records"])}; accession mismatches: {num(pre["target_mismatch_records"])}. Pair-level quarantine and assay-level quarantine have different denominators and are reported separately.')
    p('Evidence: results/audit/phenotype_pair_quarantine.csv.gz, exact_duplicate_assays.csv.gz, ambiguous_biosample_metadata.csv.gz, join_exceptions.csv.gz; logs/raw_ast_edge_cases.json.',True)

    page('5 / Missing values and collection-date precision')
    pic(FIG/'metadata_missingness.png')
    p(f'Assay platform was absent in {num(pre["missing_platform"])} records and vendor in {num(pre["missing_vendor"])}. Those fields are provenance, not model predictors. Isolate host and source type also have substantial missingness. Missing categorical values receive an explicit training-pipeline category; unknown holdout categories are handled without refitting the vocabulary.')
    p('Collection dates are parsed at their supplied precision: year, month, day, explicit range, missing, or rejected/unparsed form. Only an unambiguous in-scope year is used numerically. No day is invented and ingestion dates are not substituted for collection dates. A year-spanning interval remains missing for the scalar year feature.')
    table(['Collection-date precision','Isolates'],[[k,num(v)] for k,v in pre['date_precision_counts'].items()],widths=[350,140])
    p('Numerical predictors are collection year and assembly summaries, plus genomic call count in combined models. They are standardized using training data before imputation. Median imputation is a transparent baseline. Missingness indicators expose whether a value was absent. Iterative imputation estimates a missing feature from other observed features; it cannot prove that naturally missing values have been reconstructed correctly. [6]')

    page('6 / Genomic and categorical encoding')
    p('The genomic feature source is the NCBI AMRFinderPlus annotation, not a new annotation run on raw genomes. Each unique symbol=status token becomes a training-vocabulary multi-hot column. Parentheses, apostrophes, commas, slashes and mutation suffixes are preserved. COMPLETE, POINT, PARTIAL, PARTIAL_END_OF_CONTIG, HMM and MISTRANSLATION calls retain their distinctions. Counts describe unique reported calls, not gene copy number. [3,4]')
    table(['Reported call status','Unique-symbol observations across isolates'],[[k,num(v)] for k,v in pre['gene_status_counts'].items()],widths=[220,270])
    p(f'Completed annotation was available for {num(pre["amr_available_isolates"])} retained isolate records. In the raw API, a completed zero-hit analysis may omit its empty array; the applied flag and explicit zero count distinguish this case from unavailable analysis. Unavailable calls would have a missing count and an availability flag, not an asserted zero resistance burden. In this acquired cohort, annotation availability itself is not a meaningful missingness experiment.')
    p('The inclusive AMR list can contain general efflux/plus annotations. Their presence is not proof of phenotypic resistance and absence is not proof of susceptibility. Curated reference knowledge is external information; possible reference/benchmark overlap and later database versions limit any claim of independent biological discovery. [3,4]')
    p('Nominal metadata use either one-hot categories or alphabetically learned integer codes. Integer codes give a linear model artificial distances; they are a deliberate comparison, not a recommended biological ordering. Both pipelines learn their categories from training only. The binary target mapping is explicitly resistant=1, susceptible=0; all other source labels remain separately auditable. [7]')
    p('Explicit unscaled integer-label datasets are included as ampicillin_label_encoded_train/validation/test.csv.gz under data/processed. Their 217 columns include category codes, imputed numeric values in original units, missingness flags and training-vocabulary genomic multi-hot columns. results/label_encoding_export.json documents the fitted maps, unknown code -1 and intended inspection use; saved-pipeline inference remains the authoritative model path.',True)

    page('7 / Evaluation that respects relatedness')
    table(['Primary ampicillin partition','BioSamples','Groups','R / S'],[[k,num(v['n']),num(v['groups']),str(v['support'])] for k,v in task['splits'].items()],widths=[148,75,70,197])
    p('Connected components link BioSamples with the same recorded SNP-cluster accession. The fixed split allocates approximately 60%/20%/20% of groups to training/validation/test, so row proportions need not be exactly 60/20/20. Missing clusters fall back to BioSample-specific groups, never one shared unknown group. Known groups do not overlap across partitions. Grouping reduces known clone overlap but cannot guarantee patient or complete phylogenetic independence. [3,5]')
    p('Each of the six ampicillin configurations is evaluated in the same three grouped training folds. Every fold rebuilds the imputer, scaler, nominal categories and gene vocabulary. No full-dataset encoding or imputation is permitted. Validation-only max-F1 thresholds are supplementary; the primary result uses a fixed threshold of 0.5. [5,8]')
    p('Logistic regression uses C=1, liblinear, no class weighting, and a fixed seed. The classifier is deliberately held constant so preprocessing comparisons are interpretable. Probabilities are evaluated with average precision, AUROC, Brier score and calibration. Recall, precision, F1, balanced accuracy and confusion counts expose errors hidden by overall accuracy.')
    p('Neither genomic/phenotype disagreement nor test performance is used to remove records. Forbidden X fields include MIC, disk, phenotype, AST_phenotypes, number_drugs_*, IDs, BioProject and SNP-cluster IDs. Project/cluster identifiers are grouping/provenance fields only. The intended use is retrospective prediction after genomic annotation, not real-time prescribing before any testing.')

    page('8 / Controlled preprocessing experiment')
    subset=fixed[fixed.drug.eq('ampicillin')]
    table(['Ampicillin model','AP','AUROC','R recall','Brier'],[[r.model,fmt(r.average_precision),fmt(r.roc_auc),fmt(r.resistance_recall),fmt(r.brier)] for r in subset.itertuples()],widths=[238,63,63,63,63])
    cvmean=cv.groupby('model').average_precision.agg(['mean','std'])
    table(['Grouped training CV','Mean AP','Fold SD'],[[k,fmt(v['mean']),fmt(v['std'])] for k,v in cvmean.iterrows()],widths=[310,90,90])
    p('These are prespecified comparisons on the same frozen holdout, not a test-driven model search. Average precision should be interpreted against the held-out resistance prevalence. The metadata-only and genomic-only controls help distinguish source/lineage associations from information in genomic annotations. More elaborate imputation is not assumed better; identical or worse results are informative.')
    p('The source has complete assembly summaries and only limited numerical missingness compared with its sparse categorical fields. Therefore small numerical-imputation effects are plausible. The masking experiment later in this report evaluates a deliberately harder numerical-missingness scenario without relabeling it as a natural-data result.')
    ordinal=diffs[diffs.model.eq('ordinal_median')&diffs.metric.eq('average_precision')].iloc[0]
    iterative=diffs[diffs.model.eq('onehot_iterative_flags')&diffs.metric.eq('average_precision')].iloc[0]
    meta=subset[subset.model.eq('metadata_only')].iloc[0]
    gene=subset[subset.model.eq('genes_only')].iloc[0]
    p(f'<b>Observed finding.</b> The ordinal-median AP difference versus the primary model is {fmt(ordinal.difference,4)} (paired 95% interval {fmt(ordinal.ci_lower,4)} to {fmt(ordinal.ci_upper,4)}). One-hot iterative imputation changes AP by {fmt(iterative.difference,5)}, also with an interval spanning zero. These experiments do not establish an improvement from more elaborate preprocessing. Genomic-only AP is {fmt(gene.average_precision)}, versus metadata-only {fmt(meta.average_precision)}; curated genomic annotations supply most of the predictive information in this cohort.')

    page('9 / Primary ampicillin performance')
    pic(MOD/'figures/ampicillin_diagnostics.png',width=490)
    p(f'At the fixed threshold 0.5: accuracy {fmt(primary.accuracy)}, balanced accuracy {fmt(primary.balanced_accuracy)}, resistance recall {fmt(primary.resistance_recall)}, precision {fmt(primary.precision)}, F1 {fmt(primary.f1)}, and Brier score {fmt(primary.brier)}. The test set contains {num(primary.n_resistant)} resistant and {num(primary.n_susceptible)} susceptible observations.')
    p(f'Confusion counts: true susceptible {num(primary.tn)}, false resistance {num(primary.fp)}, missed resistance {num(primary.fn)}, correctly identified resistance {num(primary.tp)}. A high aggregate score does not eliminate false susceptibility. Calibration is descriptive; no recalibration has been fitted to the test data.')
    p('Curves and calibration bins come from held-out predictions in results/model_study/predictions.csv.gz. The serialized pipeline reproduces test predictions and accepts unseen categories/genes and missing numeric values without fitting new transformations.')

    page('10 / Uncertainty and secondary drug tasks')
    pic(MOD/'figures/ampicillin_ablation_average_precision.png',width=490)
    p('The 95% intervals resample entire held-out connected groups, using identical bootstrap weights for every frozen model. They are conditional on this training run and cohort, and do not cover all model-selection, source-sampling or annotation-reference uncertainty. Wide or overlapping intervals should temper ranking claims.')
    primaryci=ci[ci.model.eq(primary_name)&ci.metric.eq('average_precision')].iloc[0]
    cip=fixed[fixed.drug.eq('ciprofloxacin')&fixed.model.eq(primary_name)].iloc[0]
    p(f'With 300 paired group resamples, primary ampicillin AP is {fmt(primaryci.estimate)} with a 95% interval of {fmt(primaryci.ci_lower)} to {fmt(primaryci.ci_upper)}. The ciprofloxacin test contains only {num(cip.n_resistant)} resistant outcomes, so its resistance recall is based on much less resistant support than the other tasks.',True)
    table(['Secondary drug / primary preprocessing','AP','AUROC','R recall','R / S'],[[r.drug,fmt(r.average_precision),fmt(r.roc_auc),fmt(r.resistance_recall),f'{num(r.n_resistant)} / {num(r.n_susceptible)}'] for r in fixed[fixed.drug.ne('ampicillin')&fixed.model.eq(primary_name)].itertuples()],widths=[218,60,60,60,92])
    p('Tetracycline and ciprofloxacin use their own observed R/S outcomes with the same isolate split and primary preprocessing. Intermediate and nonsusceptible outcomes are excluded from this binary estimand rather than arbitrarily merged. Differences across drugs reflect distinct biology, class support, and reporting populations; they do not select the primary endpoint.')

    page('11 / Split sensitivity and missingness stress')
    table(['Alternate ampicillin evaluation','Status','AP','AUROC','Test rows sharing training SNP component'],[[r.analysis,r.status,fmt(getattr(r,'average_precision',np.nan)),fmt(getattr(r,'roc_auc',np.nan)),str(getattr(r,'test_rows_with_training_SNP_component','NA'))] for r in sens.itertuples()],widths=[210,66,55,55,104])
    p('The BioSample-random diagnostic can place related isolates on both sides of evaluation. The BioProject sensitivity asks about source transfer. These splits change test populations as well as overlap, so score differences cannot be attributed solely to leakage. A failed or class-degenerate sensitivity is reported as not evaluable, never silently replaced by a favorable split.')
    pic(FIG/'imputation_stress.png',width=490)
    p('The stress experiment fits numerical imputers on training isolates and masks previously observed validation cells under MCAR (10%,30%) and a declared contig-conditioned MAR simulation. Errors are normalized by training IQR. Labels and the test set are not used. This tests reconstruction under specified artificial mechanisms; it cannot identify the correctness of naturally missing values or establish that real missingness is MAR.')
    stress=pd.read_csv(RES/'imputation_stress.csv')
    masked=stress[stress.mechanism.eq('MCAR')&stress.nominal_mask_rate.eq(.3)].set_index(['method','feature'])
    e=lambda method,feature:masked.loc[(method,feature),'rmse_original_units']
    p(f'In the MCAR30% simulation, iterative reconstruction changed N50 RMSE from {num(e("training_median","asm_n50"))} to {num(e("iterative_bayesian_ridge","asm_n50"))} base pairs and collection-year RMSE from {fmt(e("training_median","collection_year"),2)} to {fmt(e("iterative_bayesian_ridge","collection_year"),2)} years. Reconstruction gains under artificial masking did not translate into a detectable AP gain in the naturally near-complete numerical data. Different evaluation questions therefore produce different, compatible findings.',True)

    page('12 / Runtime, memory and reproducibility')
    pic(FIG/'streaming_benchmark.png',width=490)
    p('The engineering benchmark uses actual ordered prefixes (1,000/10,000/50,000/full), without duplicating records. Each configuration runs in a separate process. Canonical hashes agree across chunk sizes for every prefix. This establishes batching invariance for the measured canonicalization/interval-parsing/hash stage, not the full relational pipeline.')
    p(f'The full preprocessing run used {fmt(pre["preprocess_wall_seconds"],1)} seconds and peak process RSS {fmt(pre["peak_rss_mib"],1)} MiB. This full in-memory join/group audit differs from the streaming microbenchmark. Single-run timings are sensitive to file caching, WSL mount performance and run order; they are not a comparison with Spark or a claim about millions of independent samples.')
    p('Reproduction in Ubuntu WSL: bash setup_wsl.sh, then bash run_wsl.sh. Raw data are already frozen. Acquisition is a separate opt-in operation that refuses to overwrite an existing manifest. The run script executes cleaning, validation, modeling, masking and streaming checks; notebook execution and PDF building are documented separately. Frozen requirements, logs, SHA256SUMS and the validated ZIP accompany the study.')
    p('The executed notebook is a readable analytical walkthrough of the frozen artifacts and core transformation functions. The complete computational experiment lives in versioned source scripts. Saved-model inference is available through model_study.py --predict; the README records its exact arguments.')

    page('13 / Interpretation, limitations and next work')
    p('The strongest improvement over a simple cleaning exercise is traceability: censored and combination measurements retain meaning; one ambiguous join and one conflicting pair are visible; annotation absence has an explicit meaning; relatedness influences evaluation; and preprocessing alternatives are measured under a fixed model. Complexity is demonstrated by real data properties and defensible decisions.')
    p('This remains a retrospective study of a selected public cohort. The same public repositories and curated references may have informed annotation development. AMRFinderPlus versions and reference databases vary across isolates. Assembly/genotype summaries omit expression, gene regulation and unrecognized mechanisms. Recorded SNP clusters do not represent a complete phylogenetic partition; clusterless genomes can still be related. Sample, location and source reporting can encode collection bias.')
    p('Nonbinary and missing source outcomes are deliberately excluded from the binary models, potentially changing the represented population. Submission-domain measurement checks cannot adjudicate historical clinical breakpoints or inter-platform disagreement. The conservative duplicate-BioSample policy discards information rather than resolving a biological identity question. Deterministic imputation does not quantify uncertainty in every missing feature.')
    p('The next defensible steps are external source validation, a fully dated reference-annotation benchmark, explicit organism/drug/method/standard breakpoint sensitivity, more complete genomic-relatedness partitions, and repeated independent computational benchmarks. Adding species or samples should follow a scientific generalization question, not merely inflate the row count.')
    p('The report does not claim publication novelty, prospective clinical utility, or independent laboratory confirmation. It supplies a reproducible empirical basis for an undergraduate research submission and a stronger starting point for supervised extension.')

    page('References and artifact map')
    for i,(label,url) in enumerate(sources,1): p(ref(i,label,url),small=True)
    p('All sources were checked on 18 October 2026. Source documentation supports interpretation and method choices; actual cohort counts and performance are computed from the archived data. The full machine-readable primary-source register is docs/SOURCES.json.',True)
    table(['Artifact','Purpose'],[['data/raw/ + acquisition_manifest.json','Frozen submitted records and acquisition integrity'],['data/processed/','Assays, unanimous label pairs, isolate feature table'],['results/audit/','Schema, cell normalization, joins, repeats, quarantine, dates'],['results/model_study/','Frozen splits, fitted pipelines, encoding maps, predictions, metrics, CIs'],['results/imputation_stress* and streaming_benchmark.json','Declared missingness simulation and batching benchmark'],['src/ + logs/ + requirements-lock-wsl.txt','Executable methods and environment evidence'],['notebooks/Research_Walkthrough.ipynb','Executed explanation and exploration'],['STUDY_GUIDE.txt + README.txt','Study/viva preparation and WSL execution']],widths=[240,250])
    p('<b>Assistance disclosure.</b> AI assistance was used for source discovery, implementation, analysis, figures and drafting. Shadman should review all methods and results, understand the code, and follow institutional disclosure rules. No authorship credentials, funding, new ethics approval or laboratory work have been invented.',True)
    def footer(canvas,doc):
        canvas.saveState(); w,h=A4
        canvas.setStrokeColor(colors.HexColor('#dce6e8')); canvas.line(50,40,w-50,40)
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#586e76'))
        canvas.drawString(50,28,'INTERNSHIP RESEARCH PRIME  |  FROZEN NCBI COHORT  |  18 OCT 2026')
        canvas.drawRightString(w-50,28,str(doc.page)); canvas.restoreState()
    pdf=ROOT/'reports/Research_Report_Prime.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=48,bottomMargin=54,
                      title='Antibiotic Resistance - Prime Preprocessing Study',author='Shadman; AI-assisted preparation').build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'reports/Research_Report_Prime.txt').write_text('\n\n'.join(transcript),encoding='utf-8')
    print(pdf)

if __name__=='__main__': build()
