# Deploy

Reference configuration: **one 24 GB GPU machine**, two `llama-server`
processes (GGUF Q4), everything local. The measured setup runs both readers on
a single card: the 27B fully on GPU, the 35B-A3B MoE with expert tensors partly
on CPU.

## 1. llama.cpp

Install a recent `llama.cpp` build whose `llama-server` supports
`--n-cpu-moe` and the `/tokenize` endpoint (2026-09-generation builds do):

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --config Release
# -> build/bin/llama-server
```

(Prebuilt release binaries work too — make sure the version is recent enough.)

## 2. Models

```bash
scripts/download_models.sh q4        # 27B-Q4 (~17 GB) + 35B-A3B-Q4 (~22.2 GB)
# sha256 values: recipes/models.lock.yaml
```

Mirrors (China-friendly):

* HuggingFace mirror: `export HF_ENDPOINT=https://hf-mirror.com` before
  fetching (the script uses `hf download`/`curl`, both honor it).
* ModelScope also carries Qwen3.5 GGUF mirrors — search "Qwen3.5" in the
  ModelScope model hub and download the same filenames, then verify sha256.

## 3. Serve

```bash
scripts/serve_model.sh 27b     # 127.0.0.1:10361  (full GPU: -ngl 99)
scripts/serve_model.sh 35b     # 127.0.0.1:10362  (-ngl 99 --n-cpu-moe 36)
```

Both bind to `127.0.0.1` only. Serving flags (mirroring the measured setup):

| flag | value | why |
|---|---|---|
| `-ngl` | 99 | all attention layers on GPU |
| `--n-cpu-moe` | 36 (35B only) | keep MoE expert tensors partly on CPU so the 35B fits 24 GB — raise this number if you run out of VRAM |
| `-c` | 8192 | enough for long MCQ + context |
| `-b/-ub` | 1024 | batch sizes |
| `--flash-attn on -ctk q8_0 -ctv q8_0` | | KV cache quantization |
| `-np 8` | | parallel slots for batch evaluation |
| `--no-warmup --jinja` | | serve quickly; harmless for raw prompts |

Verify before use:

```bash
python -m open_medical_jev check-server --server http://127.0.0.1:10361
python -m open_medical_jev verify-tokenizer --server http://127.0.0.1:10361 \
    --hf-repo Qwen/Qwen3.5-27B        # requires: pip install "transformers>=4.40"
```

## 4. Python side

```bash
scripts/setup_env.sh          # venv + editable install + selftest (no heavy deps)
# optional HF tokenizer backend:
WITH_HF=1 scripts/setup_env.sh
```

The core pipeline is stdlib-only; `transformers` is only needed for
`--tokenizer-backend hf`.

## Troubleshooting

| symptom | fix |
|---|---|
| server fails to start / CUDA OOM | raise `--n-cpu-moe` (35B) or lower `-c`/`-np` |
| `/tokenize` returns 404 | llama.cpp too old — update it, or run with `--tokenizer-backend hf` |
| all readings "uncertain" / letter not found | run `verify-tokenizer`; confirm the GGUF matches `recipes/models.lock.yaml` |
| run too slow | check `-t` (threads), GPU utilization; more `--n-cpu-moe` = slower but fits |
| `labels not in top-200` errors | raise `--n-probs`; check the prompt wasn't truncated |

## Notes

* Multi-user serving: put your own gateway/auth in front; these servers have
  no authentication by design (local research use).
* The measured per-question latency (one 24 GB GPU, batch concurrency 8):
  ≈0.076 s/question throughput, ≈0.28 s single-flight — readout-dependent;
  re-measure on your hardware.
