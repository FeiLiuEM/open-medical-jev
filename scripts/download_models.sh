#!/usr/bin/env bash
# Open Medical Jev — download the GGUF models used by the recipe.
#
# Default (q4): Qwen3.5-27B-UD-Q4_K_XL + Qwen3.5-35B-A3B-UD-Q4_K_XL — the
# configuration measured in the shipped results (fits one 24GB GPU; the 35B
# MoE runs with experts partly on CPU via --n-cpu-moe).
#
# Usage:  scripts/download_models.sh [q4|q5]
# Env:    DEST=models (target dir), HF_ENDPOINT=https://hf-mirror.com (mirror)
#
# China mirror notes:
#   - HuggingFace mirror:  export HF_ENDPOINT=https://hf-mirror.com
#   - or ModelScope: search "Qwen3.5" in the ModelScope model hub and download
#     the same filenames, then verify sha256 (see docs/deploy.md)
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${DEST:-models}"
QUANT="${1:-q4}"
mkdir -p "$DEST"

fetch() { # <repo> <file>
  local repo="$1" file="$2"
  if [ -f "$DEST/$file" ]; then
    echo "already present: $DEST/$file"
    return 0
  fi
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" "$file" --local-dir "$DEST"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" "$file" --local-dir "$DEST"
  else
    local url="https://huggingface.co/$repo/resolve/main/$file"
    echo "downloading $url"
    curl -L --fail -C - -o "$DEST/$file.part" "$url"
    mv "$DEST/$file.part" "$DEST/$file"
  fi
}

case "$QUANT" in
  q4)
    fetch unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-UD-Q4_K_XL.gguf
    fetch unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
    ;;
  q5)
    fetch unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-UD-Q5_K_XL.gguf
    fetch unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
    ;;
  *)
    echo "usage: $0 [q4|q5]"; exit 1 ;;
esac

echo
echo "=== sha256 (compare with recipes/models.lock.yaml) ==="
python3 - "$DEST" <<'PY'
import glob, hashlib, os, sys
dest = sys.argv[1]
for p in sorted(glob.glob(os.path.join(dest, "**", "*.gguf"), recursive=True)):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    print(f"{h.hexdigest()}  {os.path.basename(p)}  ({os.path.getsize(p)/1e9:.1f} GB)")
PY
