#!/usr/bin/env bash
# Open Medical Jev — environment setup (Python side).
#
# Creates a virtualenv, installs the package (editable), and prints next steps.
# The core pipeline has **no third-party dependencies**; the optional HF
# tokenizer backend needs `transformers` (WITH_HF=1).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "python3 not found"; exit 1; }

"$PY" - <<'PY'
import sys
assert sys.version_info >= (3, 9), "Python >= 3.9 required, got " + sys.version.split()[0]
print("python:", sys.version.split()[0])
PY

if [ "${SKIP_VENV:-0}" != "1" ]; then
  if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
  .venv/bin/python -m pip install -q --upgrade pip
  if .venv/bin/python -m pip install -q -e . 2>/dev/null; then
    echo "installed: editable package (entry point: omj)"
    PYTHON=".venv/bin/python"
  else
    echo "!! editable install failed (offline?). You can still run with PYTHONPATH=src:"
    echo "   PYTHONPATH=src python3 -m open_medical_jev selftest"
    PYTHON=".venv/bin/python"
  fi
  if [ "${WITH_HF:-0}" = "1" ]; then
    .venv/bin/python -m pip install -q "transformers>=4.40"
    echo "installed: transformers (HF tokenizer backend)"
  fi
else
  PYTHON="$PY"
fi

echo
echo "self-check:"
PYTHONPATH="${PYTHONPATH:-}" PYTHONPATH="src" $PYTHON -m open_medical_jev selftest || true

echo
echo "next steps:"
echo "  1) install llama.cpp (see docs/deploy.md), then:"
echo "     scripts/download_models.sh q4"
echo "  2) serve the two models:"
echo "     scripts/serve_model.sh 27b     # 127.0.0.1:10361"
echo "     scripts/serve_model.sh 35b     # 127.0.0.1:10362"
echo "  3) quick demo:"
echo "     scripts/quickstart.sh"
