#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
VENV_DIR="${VENV_DIR:-/tmp/shadman_amr_prime_venv}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv --without-pip "$VENV_DIR"
fi
if ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  bootstrap_file="$(mktemp /tmp/amr_prime_get_pip_XXXXXX.py)"
  python3 -c 'import sys,urllib.request;urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py",sys.argv[1])' "$bootstrap_file"
  "$VENV_DIR/bin/python" "$bootstrap_file"
  rm -- "$bootstrap_file"
fi
"$VENV_DIR/bin/python" -m pip install -r requirements-lock-wsl.txt
"$VENV_DIR/bin/python" -c 'import numpy,pandas,sklearn,matplotlib,nbclient;print("Prime scientific environment ready")'
printf 'Ready. Run bash run_wsl.sh (VENV_DIR=%s).\n' "$VENV_DIR"
