INTERNSHIP RESEARCH PRIME
Auditable preprocessing and antimicrobial resistance prediction in Salmonella

Project location
C:\Users\USER\OneDrive\Desktop\Internship_Research_Prime

This is a separate research project using a new organism and source collection.
The earlier Internship_Research project is preserved. The emphasis here is on
real data-quality decisions, reproducible transformations and defensible model
evaluation. This is an undergraduate research study, with no claim of novel gene
discovery, publication readiness or clinical deployment.

WHAT THE DATA ACTUALLY CONTAIN

The frozen NCBI Pathogen Detection acquisition on 18 October 2026 contains:
  142,338 antibiotic susceptibility test (AST) observations across 72 antibiotics;
    9,977 assembled isolate records, representing 9,976 distinct BioSamples;
  matched isolate metadata, assembly quality summaries and AMRFinderPlus calls.

An AST row is one assay, not an independent bacterium or patient. Most isolates
have multiple antibiotics tested; a small number have repeated assays. The
prediction task uses one resolved BioSample-antibiotic label, and related isolates
are kept together during the primary split. Do not report 142,338 independent
samples. The cleaned modeling sample sizes are recorded separately per drug.

The acquisition queried the NCBI organism group named Salmonella enterica. All
9,977 acquired isolate records were verified as species_taxid 28901. The source
includes mostly environmental/other isolates and uneven submission projects,
locations and years. It is not a representative survey of resistance prevalence.
No raw sequence reads, new whole-genome assemblies or patient outcome data are
used. Genomic predictors are existing annotations, including general efflux
calls in the NCBI AMR_genotypes field, not newly discovered mechanisms.

RUN IN WSL

Open the WSL terminal and enter:

  cd /mnt/c/Users/USER/OneDrive/Desktop/Internship_Research_Prime
  bash setup_wsl.sh && bash run_wsl.sh

The setup/run wrappers and requirements-lock.txt describe the software environment.
The existing prepared interpreter for this session is:
  /tmp/shadman_amr_prime_venv/bin/python
A directory under /tmp may be cleared after a system restart; use setup_wsl.sh
to recreate the environment. The full pipeline reruns derived outputs. It does
not automatically redownload or replace the frozen source files.

Individual stages, after preparing the environment:

  /tmp/shadman_amr_prime_venv/bin/python src/preprocess.py --root .
  /tmp/shadman_amr_prime_venv/bin/python src/validate_preprocessing.py --root .
  /tmp/shadman_amr_prime_venv/bin/python src/model_study.py --project-root .
  /tmp/shadman_amr_prime_venv/bin/python src/benchmark_streaming.py --root .

For supported options:
  /tmp/shadman_amr_prime_venv/bin/python src/model_study.py --help
  /tmp/shadman_amr_prime_venv/bin/python src/benchmark_streaming.py --help

Data download is a separate, explicit operation:
  bash download_wsl.sh
The downloader refuses to overwrite an existing final snapshot or manifest.
Use a new destination for a future acquisition, and retain the original manifest
and raw files when comparing runs. The source is a live service: a fresh download
can differ even with the same query. The browser backend is not a promised stable
public API, so future compatibility may require updating the acquisition code.

WHERE TO START

  STUDY_GUIDE.txt
    Three-day learning sequence, worked examples and viva questions.
  notebooks/
    Readable execution narrative and analysis, using actual project outputs.
  reports/
    Research findings and figures; use the delivered report for interpretation.
  data/raw/acquisition_manifest.json
    Exact queries, source URLs, retrieval timestamps, page checks and file hashes.
  results/preprocessing_summary.json
    Measured raw/clean counts, missingness, exclusions and grouping statistics.
  results/preprocessing_validation.json
    Source-integrity checks, hand-calculated fixtures and reconciliation results.
  results/model_study/metrics.csv and summary.json
    Actual per-drug results, split support, experiment protocol and warnings.
  results/streaming_benchmark.json
    Measured row-transformation timing/memory and cross-chunk output checks.

WHAT MAKES THE PREPROCESSING SUBSTANTIAL

1. Source integrity and schema profiling
   The pipeline verifies SHA256 hashes before transforming the frozen JSONL files.
   Absent fields, observed data types, original row IDs and acquisition provenance
   remain inspectable. Initial and final counts agree and IDs are unique and
   sorted; these checks do not make the live source transactionally consistent.

2. Record linkage and repeated evidence
   The pipeline checks BioSample and target accession agreement. The one BioSample
   associated with two assembly records is quarantined from the primary metadata
   cohort instead of silently picking an assembly. It distinguishes multiple
   antibiotics, exact duplicate assays and repeated measurements. Conflicting
   labels are quarantined rather than resolved by majority vote.

3. Censored and combination measurements
   MIC <=4 is an interval (0,4], not the exact value 4. MIC >8 is an open lower
   bound. Combination measurements such as 32/4 retain two concentrations; they
   are never divided to produce 8. The source provides one comparator, which is
   not invented for the secondary concentration. MIC in mg/L and disk diffusion
   in mm retain separate fields and structural missingness. Invalid values and
   unknown comparators are flagged for audit.

4. Missingness with meaning
   Text sentinels, Unicode and whitespace are normalized. Partial collection dates
   retain year/month/day/range precision. Missing host or source is not guessed.
   A completed zero-hit AMRFinderPlus analysis is distinguished from unavailable
   analysis. Labels and untested antibiotics are never imputed. Numeric feature
   imputation and categorical vocabularies are fitted inside training pipelines.
   Categorical missing values use an explicit token. Median, median plus missing
   indicators, and iterative numeric imputation are compared without assuming
   that the most complicated option must improve performance.

5. Encoding and experimental controls
   Binary labels use source-resistant=1 and source-susceptible=0. Intermediate,
   nonsusceptible, undefined, missing and contradictory labels do not enter the
   primary binary task. Nominal predictors are compared using ordinal codes and
   one-hot encoding; ordinal numbers carry an artificial order and are a baseline.
   Gene symbol=status tokens use a training-only sparse multi-hot vocabulary.
   Symbols and status labels remain intact; token presence does not measure copy
   number. Unknown categories and unseen genes have explicit inference behavior.

6. Leakage controls and uncertainty
   MIC, disk results, source AST summaries and drug resistance counts cannot enter
   X. They would leak evidence used to define the target. Accession/project/cluster
   identifiers are used for linkage, splitting or audit, not as predictors.
   The primary split holds entire recorded SNP/BioSample components together.
   Missing cluster links cannot establish biological independence. Group-aware
   training-fold validation, paired group bootstrap comparisons and alternative
   split diagnostics help expose uncertainty without establishing external validity.

ACTUAL EDGE CASES TO INSPECT

There are 24 extra observations beyond unique target-antibiotic pairs and one
pair with opposing source labels. These are raw-data findings, not the number
of exact duplicates removed by the cleaner. The opposing pair is:
  PDT000555592.1, piperacillin-tazobactam, BioSample SAMN11179044.
  Microscan: susceptible, MIC <=16/4; Phoenix: resistant, MIC >64/4.
Both use reported CLSI standards. Both raw assays remain available. See
logs/raw_ast_edge_cases.json and the phenotype quarantine audit.

The raw collection has 81,756 absent laboratory platform entries, 81,568 absent
vendor entries and 153 disk-diffusion observations. MIC is absent for those disk
observations; filling them with a generic median MIC would confuse assay types.
All acquired isolates have an applied AMRFinderPlus analysis. The code still
handles unavailable analysis as a separate state for valid future inputs and
validation fixtures. Do not claim that this cohort contains observed missing
analysis merely because the pipeline supports it.

READING THE MODEL RESULTS

Ampicillin is the primary task; tetracycline and ciprofloxacin are secondary.
The fixed primary model is onehot_median_flags. The six ampicillin preprocessing
combinations compare ordinal/one-hot with median/median_flags/iterative_flags;
metadata-only and genes-only controls assess feature contributions. Logistic
regression is held fixed so preprocessing changes can be assessed directly.

Read results/model_study/metrics.csv together with summary.json. Compare matching
drug, split and threshold_policy. The main threshold is 0.5. A secondary threshold
is selected from validation data only; its test result is labeled separately.
Average precision, ROC AUC, balanced accuracy, precision, resistance recall, F1,
Brier score and confusion counts answer different questions. Accuracy alone can
hide failures on resistance. Inspect class counts and calibration before citing
a percentage. Do not select the best-looking test result after seeing the table.

The iterative procedure is a single deterministic multivariate imputation, not
pooled multiple-imputation inference. Sparse or absent numeric missingness may
produce little difference between methods. A small or uncertain difference is a
valid result. Bootstrap intervals condition on the fitted models and split; they
do not include all training or model-selection uncertainty. The source labels
come from different testing standards and unspecified editions; the pipeline does
not reinterpret them using guessed breakpoints.

SCALABILITY

The benchmark runs real ordered prefixes of 1,000, 10,000 and 50,000 observations
and the full source at several chunk sizes. It compares canonical row transforms
and hashes, not the complete preprocessing/model pipeline. It does not duplicate
rows or extrapolate to millions of isolates. Timing includes a single run per
configuration and is affected by process startup and operating-system caches.
This dataset fits local memory; Spark is a possible extension for larger sources,
not evidence that distributed processing was required or measured here.

PROVENANCE, REUSE AND AI ASSISTANCE

Official source documentation:
  AST format and downloadable results:
  https://www.ncbi.nlm.nih.gov/pathogens/docs/ast/
  Source-phenotype quality and sampling limitations:
  https://www.ncbi.nlm.nih.gov/pathogens/ast/
  Isolate, genotype, SNP-group and field definitions:
  https://www.ncbi.nlm.nih.gov/pathogens/pathogens_help/
  Genotype/phenotype distinction:
  https://www.ncbi.nlm.nih.gov/pathogens/faq/
  NCBI data-use and copyright policies:
  https://www.ncbi.nlm.nih.gov/home/about/policies/

NCBI makes molecular data available without imposing its own reuse restrictions;
its policy also notes that submitters may retain rights. Do not label this entire
collection CC0 or assign a third-party dataset license that was not supplied.
Retain accessions, acquisition provenance, NCBI attribution and source policies
when sharing. No separate software license has been selected for this project.

AI assistance was used to develop the code, documentation and analysis workflow.
The source data are public observations rather than fabricated examples. Shadman
should be able to rerun the work and explain the decisions before presenting it
as his research contribution. The study supports methodological learning and
computational evaluation; it does not validate patient-treatment decisions.

DELIVERED ARTIFACTS

reports/Research_Report_Prime.pdf is the generated research report.
notebooks/Research_Walkthrough.ipynb is executed, with11code cells and no errors.
DATA_DICTIONARY.csv documents all77cleaned export fields.
results/model_validation.json independently recomputes every saved validation/test
prediction and fixed-threshold metric across14pipelines.
Research_Project_Prime.zip contains the frozen data, code, outputs and guides.
SHA256SUMS.txt and results/archive_validation.json record archive verification.

To rebuild the PDF separately after a scientific run:
  /tmp/shadman_amr_prime_venv/bin/python src/build_report.py
To rebuild and verify the full project ZIP:
  /tmp/shadman_amr_prime_venv/bin/python src/package_project.py

Primary cohort finding:ampicillin average precision .974973 and resistance recall
.954451 on2070held-out BioSamples. Metadata-only average precision .555232.
Numerical imputation differences were very small with only40missing collection
years. Consult the report, confidence intervals and controls before interpretation.
