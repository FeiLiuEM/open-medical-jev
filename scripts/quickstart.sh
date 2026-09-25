#!/usr/bin/env bash
# Open Medical Jev — guided quickstart.
#
# Assumes both llama.cpp servers are running (scripts/serve_model.sh):
#   A: 127.0.0.1:10361 (27B)    B: 127.0.0.1:10362 (35B)
set -euo pipefail
cd "$(dirname "$0")/.."

A="${SERVER_A:-http://127.0.0.1:10361}"
B="${SERVER_B:-http://127.0.0.1:10362}"
PY="${PY:-python3}"
if [ -x .venv/bin/python ]; then PY=".venv/bin/python"; fi
run() { PYTHONPATH=src "$PY" -m open_medical_jev "$@"; }

echo "== 1) selftest (no models needed) =="
run selftest

echo
echo "== 2) server checks =="
run check-server --server "$A" || echo "  (server A not reachable: $A)"
run check-server --server "$B" || echo "  (server B not reachable: $B)"

echo
echo "== 3) single item through both readers + router =="
run read --item examples/item.json --servers "$A,$B" || {
  echo "  (skipped: servers not reachable — start them with scripts/serve_model.sh)"; }

echo
echo "== 4) batch evaluation on the toy fixtures =="
run evaluate --items tests/fixtures/sample_items.jsonl --servers "$A,$B" \
  --concurrency 4 --out /tmp/omj_demo || {
  echo "  (skipped: servers not reachable)"; }

echo
echo "done. For your own data see docs/protocol.md."
