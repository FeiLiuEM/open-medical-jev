#!/usr/bin/env bash
# Open Medical Jev — start a llama.cpp server for one of the two models.
#
# Usage:  scripts/serve_model.sh 27b [port]      # default port 10361
#         scripts/serve_model.sh 35b [port]      # default port 10362
#         scripts/serve_model.sh /path/to.gguf [port]
#
# The 35B-A3B MoE runs with attention on GPU and experts partly on CPU
# (--n-cpu-moe) so that it fits a single 24GB card. Tune NCPU_MOE if needed
# (more on CPU = less VRAM, slower).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-27b}"
PORT="${2:-}"
case "$MODEL" in
  27b)
    GGUF="${GGUF_27B:-models/Qwen3.5-27B-UD-Q4_K_XL.gguf}"
    PORT="${PORT:-10361}"
    EXTRA="-ngl ${NGL:-99}"
    ;;
  35b)
    GGUF="${GGUF_35B:-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf}"
    PORT="${PORT:-10362}"
    EXTRA="-ngl ${NGL:-99} --n-cpu-moe ${NCPU_MOE:-36}"
    ;;
  *)
    if [ -f "$MODEL" ]; then
      GGUF="$MODEL"; PORT="${PORT:-10361}"; EXTRA="-ngl ${NGL:-99}"
    else
      echo "usage: $0 27b|35b|/path/to.gguf [port]"; exit 1
    fi
    ;;
esac

[ -f "$GGUF" ] || { echo "GGUF not found: $GGUF"; echo "run scripts/download_models.sh first"; exit 1; }
command -v llama-server >/dev/null || { echo "llama-server not found on PATH (see docs/deploy.md)"; exit 1; }

echo "serving: $GGUF"
echo "port:    $PORT (127.0.0.1)"
exec llama-server \
  -m "$GGUF" $EXTRA \
  -c "${CTX:-8192}" -b "${UB:-1024}" -ub "${UB:-1024}" \
  --flash-attn on -ctk q8_0 -ctv q8_0 \
  -t "${THREADS:-16}" \
  --port "$PORT" --host 127.0.0.1 \
  -np "${NP:-8}" --no-warmup --jinja --alias "${ALIAS:-jev-reader}"
