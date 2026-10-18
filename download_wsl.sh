#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
VENV_DIR="${VENV_DIR:-/tmp/shadman_amr_prime_venv}"
if [[ -f data/raw/acquisition_manifest.json ]]; then
  printf 'Frozen snapshot already present; preserving it. Use a new project copy with an empty data/raw folder for a new acquisition.\n'
  exit 0
fi
"$VENV_DIR/bin/python" src/download_data.py --root .
