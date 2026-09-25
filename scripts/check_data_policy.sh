#!/usr/bin/env bash
# Guard: this repository must never carry data, model weights, or secrets.
# See DATA_POLICY.md. Run locally and in CI.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
list_files() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git ls-files | grep -v '^\.recon/' || true
  else
    find . -type f -not -path './.git/*' -not -path './.recon/*' -not -path './.venv/*' -not -path './models/*' || true
  fi
}

echo "== 1) forbidden file types =="
while IFS= read -r f; do
  case "$f" in
    *.gguf|*.safetensors|*.bin|*.pt|*.ckpt|*.onnx)
      echo "FORBIDDEN model/weight file: $f"; fail=1;;
  esac
done < <(list_files)

echo "== 2) large files (>5MB) =="
while IFS= read -r f; do
  [ -f "$f" ] || continue
  sz=$(wc -c < "$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt 5242880 ]; then echo "LARGE file (>5MB): $f"; fail=1; fi
done < <(list_files)

echo "== 3) secret-like patterns =="
while IFS= read -r f; do
  [ -f "$f" ] || continue
  if grep -lE "ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY" "$f" >/dev/null 2>&1; then
    echo "SECRET-LIKE content: $f"; fail=1
  fi
done < <(list_files)

echo "== 4) corpus-ish extensions =="
while IFS= read -r f; do
  case "$f" in
    *.csv|*.tsv|*.parquet|*.feather|*.db|*.sqlite)
      echo "DATA-LIKE file (review!): $f"; fail=1;;
  esac
done < <(list_files)

if [ "$fail" = "0" ]; then
  echo "data-policy check: OK ✓"
else
  echo "data-policy check: FAILED ✗"; exit 1
fi
