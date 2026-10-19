# Salmonella Genomic AMR Preprocessing

**Machine Learning-Based Analysis and Prediction of Antibiotic Resistance**

A reproducible study of preprocessing, cleaning, missing values, categorical encoding, and genomic phenotype prediction using a frozen NCBI Salmonella dataset. Prepared by **Shadman_Al_Muksit**, BSc Computer Science and Engineering / Data Science.

## Start here

- [Research report — 15 pages](reports/Research_Report_Prime.pdf)
- [Executed research walkthrough](notebooks/Research_Walkthrough.ipynb)
- [Detailed WSL instructions](README.txt)
- [Complete project ZIP](Research_Project_Prime.zip)
- [Data dictionary](DATA_DICTIONARY.csv)

## Frozen dataset

Acquired on **18 October 2026** from [NCBI Pathogen Detection](https://www.ncbi.nlm.nih.gov/pathogens/ast/). The raw experimental phenotypes and genomic annotations are separate source records.

| Unit | Count |
|---|---:|
| AST observations | 142,338 |
| Antibiotics | 72 |
| Assembled isolate records | 9,977 |
| Distinct raw BioSamples | 9,976 |
| Unambiguous retained BioSamples | 9,975 |
| Eligible binary BioSample–drug pairs across the panel | 138,911 |

AST rows are repeated antibiotic tests, not independent patients or isolates. Source queries, timestamps, page-level checks, accession identifiers, and SHA256 hashes are preserved in [the acquisition manifest](data/raw/acquisition_manifest.json).

## What the preprocessing does

- Preserves censoring in 75,874 assays and both concentrations in 19,368 combination MIC results; keeps disk diffusion and MIC in separate units.
- Audits repeated tests, quarantines a real inter-platform phenotype conflict, and excludes ambiguous BioSample-to-assembly joins.
- Distinguishes structural missingness, missing metadata, unavailable genomic analyses, and explicit completed zero-hit analyses.
- Preserves collection-date precision and genomic symbol/call-status semantics.
- Fits categorical vocabularies, numeric imputers, scaling, and genomic multi-hot encodings on training data only.
- Compares six ordinal/one-hot and median/indicator/iterative-imputation configurations under the same model and split.
- Provides explicit integer-label train/validation/test exports and their fitted maps.
- Separates known SNP/BioSample groups, runs grouped training cross-validation, and reports paired group-bootstrap intervals and source-split sensitivities.
- Benchmarks real source prefixes and verifies identical canonical outputs across chunk sizes.

## Main result

The prespecified primary task is **ampicillin** using one-hot encoding, median imputation with missingness indicators, and curated genomic multi-hot features.

| Held-out measure | Value |
|---|---:|
| Test BioSamples / connected groups | 2,070 / 242 |
| Average precision | 0.974973 |
| Average precision, group-bootstrap 95% interval | 0.935634–0.986415 |
| AUROC | 0.988799 |
| Resistance recall at threshold 0.5 | 0.954451 |
| Metadata-only average precision | 0.555232 |

Curated genomic annotations supply most of the predictive information. More elaborate numerical imputation did not establish an improvement on this cohort's nearly complete numeric data. Separately labeled MCAR/MAR masking experiments explore harder missingness conditions without presenting simulated missingness as naturally observed data.

Tetracycline and ciprofloxacin are secondary tasks. Full support, metrics, thresholds, comparisons, and limitations are in [the model results](results/model_study/metrics.csv) and [the report](reports/Research_Report_Prime.pdf).

## Reproduce in Ubuntu WSL

```bash
git clone https://github.com/Shadman1003/salmonella-genomic-amr-preprocessing.git
cd salmonella-genomic-amr-preprocessing
bash setup_wsl.sh
bash run_wsl.sh
```

The default isolated environment is `/tmp/shadman_amr_prime_venv`; override it with `VENV_DIR` if preferred. The tested scientific environment is Python 3.12.3 on Ubuntu WSL. Exact dependencies are in [requirements-lock-wsl.txt](requirements-lock-wsl.txt).

The full run uses the frozen raw snapshot and rebuilds audits, models, encoding exports, stress experiments, benchmarks, the executed notebook, and the PDF report. Acquisition is a separate opt-in command, `bash download_wsl.sh`, that preserves an existing snapshot.

```bash
# Verify frozen source and preprocessing outputs.
/tmp/shadman_amr_prime_venv/bin/python src/validate_preprocessing.py --root .

# Independently reproduce every saved validation/test prediction and metric.
/tmp/shadman_amr_prime_venv/bin/python src/validate_models.py

# Rebuild and verify the complete archive.
/tmp/shadman_amr_prime_venv/bin/python src/package_project.py
```

The completed workflow recorded 14 fitted pipelines, 18 grouped CV fits, 300 paired group-bootstrap replicates, and 50,299 independently recomputed validation/test predictions. Its 42 frozen metric rows reproduced to numerical precision. See [delivery QA](results/delivery_qa.json) and [model validation](results/model_validation.json).

## Scope, provenance, and reuse

This is retrospective analysis of a public surveillance/research convenience cohort. Genomic predictors are existing AMRFinderPlus annotations; this work does not discover new resistance mechanisms from raw genomes or establish prospective clinical performance. Known SNP grouping does not establish complete patient or phylogenetic independence. Later reference annotations, incomplete metadata, sampling bias, nonbinary-label exclusions, and small resistant support for some secondary outcomes limit interpretation.

The [primary-source register](docs/SOURCES.json), [study protocol](docs/STUDY_PROTOCOL.txt), and [project context](PROJECT_CONTEXT.txt) document the methods and decisions. Follow [NCBI source-use policies](https://www.ncbi.nlm.nih.gov/home/about/policies/); no unsupported blanket third-party dataset license is assigned here.

AI assistance was used for source discovery, implementation, analysis, figures, and writing. The researcher should review the work, understand the methods, and follow institutional disclosure requirements. No laboratory work, funding, new ethics approval, publication novelty, or clinical validation is claimed.
