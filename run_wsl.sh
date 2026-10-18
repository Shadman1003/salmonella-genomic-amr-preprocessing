#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
VENV_DIR="${VENV_DIR:-/tmp/shadman_amr_prime_venv}"
PY="$VENV_DIR/bin/python"
if [[ ! -x "$PY" ]]; then
  printf 'Run bash setup_wsl.sh first.\n' >&2
  exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
"$PY" src/preprocess.py --root . 2>&1 | tee logs/preprocessing.log
"$PY" src/validate_preprocessing.py --root . 2>&1 | tee logs/preprocessing_validation.log
"$PY" src/model_study.py --project-root . 2>&1 | tee logs/model_study.log
"$PY" src/export_label_encoded.py 2>&1 | tee logs/label_encoding_export.log
"$PY" src/imputation_stress.py 2>&1 | tee logs/imputation_stress.log
"$PY" src/benchmark_streaming.py --root . 2>&1 | tee logs/streaming_benchmark.log
"$PY" src/build_dictionary.py 2>&1 | tee logs/data_dictionary.log
"$PY" src/build_notebook.py 2>&1 | tee logs/notebook.log
"$PY" src/validate_models.py 2>&1 | tee logs/model_validation.log
"$PY" src/build_report.py 2>&1 | tee logs/report_build.log
printf 'Study complete. See results/, notebooks/, and reports/Research_Report_Prime.pdf.\n'
