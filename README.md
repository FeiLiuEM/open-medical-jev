# Open Medical Jev

**Jev-class judgment from frozen open models — computation, not training.**

![Open Medical Jev architecture — two frozen readers → per-option probabilities → mean fusion → router (confidence, auto-release gate, conformal set) → decision](assets/architecture.svg)

<sub>Vector source: [`assets/architecture.svg`](assets/architecture.svg). Two frozen readers answer a yes/no judgment per option; the fit-free router turns their agreement into a confidence, an auto-release gate and a guaranteed candidate set. Constants: [`recipes/routing.yaml`](recipes/routing.yaml).</sub>

> **Jev- and OpenJev-level results on national medical exams — with no training of any kind.**
> On 600-item licensing-exam papers, the full 4-reading configuration lands within 2 points of Jev
> on all three (0.8883 / 0.8634 / 0.8100 vs 0.8967 / 0.8833 / 0.8133) — level with the OpenJev
> open-weights run (ahead on China and India; within 2.2 pp on the US paper) — running locally on
> one 24 GB GPU. No fine-tuning, no distillation, no corpus.

[![ci](https://github.com/FeiLiuEM/open-medical-jev/actions/workflows/ci.yml/badge.svg)](https://github.com/FeiLiuEM/open-medical-jev/actions/workflows/ci.yml)

Open Medical Jev pairs two untouched, off-the-shelf open models
([Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) and
[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), via GGUF
quantizations) as readers of a yes/no judgment task, and adds a small routing
layer on top:

* **combined confidence** — agreement between the two readers, with a measured
  discount when they disagree,
* **auto-release gate** — Chow's rule on the combined confidence (high
  confidence releases automatically; the rest is routed to human review),
* **guaranteed candidate set** — split-conformal prediction set at a chosen
  error rate (with a documented recalibration procedure).

No fine-tuning. No distillation. No corpus. Code + recipe only.

**Model-agnostic by design.** Because nothing is trained, a newer open model is supported by swapping the reader and re-fitting the two small routing constants (with a verification pass on your own data) — the protocol, router and guarantees carry over unchanged, and there is no training pipeline to rebuild.

## Key results

Measured on the frozen models (nothing trained), against TypeSafe Jev 1.13.0
on the same item sets. `coverage @ accuracy` for gated rows; full-set accuracy
otherwise. See [reports/results_summary.md](reports/results_summary.md) for
protocols, caveats and every table.

| Item set | 27B | 35B | 2-model mean fusion | router: unanimous coverage @ acc | Jev (hosted) |
|---|---|---|---|---|---|
| dev-300 (MedMCQA subset) | 0.8000 | 0.7933 | **0.8167** | 78.7% @ 0.9025 | 0.8300 |
| JevBench (public, 70/72) | 0.8143 | 0.7857 | 0.8143 | 82.9% @ 0.9310 | 0.9860 |
| jev-decision-bench (916/39 tasks) | 0.8395 | 0.8046 | 0.8373 | 87.1% @ 0.8697 | ~0.86 |

### National licensing exams (600 items each; frozen models, nothing trained)

| system | China (NMLE 2021) | US (USMLE equiv.) | India (NEET-PG equiv.) |
|---|---|---|---|
| Jev 1.13.0 (hosted API) | 0.8967 | 0.8833 | 0.8133 |
| OpenJev (open weights, GGUF Q4, local run) | 0.8800 | 0.8850 | 0.7750 |
| **Open Medical Jev — 4 readings** (both readers × both readout structures) | **0.8883** | **0.8634** | **0.8100** |
| Open Medical Jev — 27B (single reader) | 0.8767 | 0.7437 | 0.7900 |
| Open Medical Jev — 35B-A3B (single reader) | 0.8667 | 0.7352 | 0.7533 |

China is the real 2021 paper; US/India are fixed-seed equivalent draws. 593/600 readable for
us on the US paper (7 read failures, excluded from the denominator). Timings are per-item
model compute only (ours 0.27–0.31 s/item locally; Jev's API ≈1.02 s/item). All three
systems clear each paper's written pass line (60% / 60% / 50%). The 4-reading row runs every
reading the system offers — both readers × both readout structures — so its per-item compute is
higher than the single-reader rows.

![Coverage-accuracy of Open Medical Jev (4 readings) vs Jev and OpenJev on the three exam papers](assets/coverage_accuracy.svg)

<sub>**Coverage–accuracy** on the three 600-item papers ((a) China — the real 2021 paper; (b, c) fixed-seed equivalent draws). Each curve sorts its own system's answers by per-item confidence: x = fraction auto-answered, y = accuracy within that fraction. "4 readings" = both frozen readers × both readout structures, combined with fixed weights from each reading's measured accuracy — nothing trained. OpenJev exposes no per-item confidence, so it appears as a single square at full coverage. Jev's hosted confidence curve still leads at the highest precision tiers; toward full coverage the systems converge. Source: [reports/results_summary.md](reports/results_summary.md).</sub>

**Calibration**: after a 1-parameter tier-conditioned temperature fit (on
dev-300 only), the fused probability of `jev-decision-bench` reaches **ECE
0.0096** vs Jev's public 0.027 on the same benchmark (~3x better); dev-300:
0.0322. The conformal candidate set is valid on the examined sets when the
calibration distribution matches (see `docs/evaluation.md` for the
distribution-shift caveat).

## How it works

* **pair readout** (primary): "Is the candidate answer correct? Reply yes or
  no." — read yes/no token probabilities; the per-option signal is
  `Δ = max logprob(yes) − max logprob(no)`.
* **choice readout** (secondary): all options in one prompt; read letter
  probability mass.

Both prompts, the decoding parameters and the tokenizer notes are pinned in
[docs/protocol.md](docs/protocol.md) — that file is the contract; keep prompts
in `src/open_medical_jev/protocol.py` in sync with it.

## Quickstart

```bash
# 1) Python side (no third-party deps for the core pipeline)
scripts/setup_env.sh              # venv + editable install + selftest
PYTHONPATH=src python3 -m open_medical_jev selftest

# 2) Models (GGUF, ~17 GB + ~22 GB; China mirrors: see docs/deploy.md)
scripts/download_models.sh q4

# 3) Serve the two readers (llama.cpp; single 24 GB GPU)
scripts/serve_model.sh 27b        # 127.0.0.1:10361
scripts/serve_model.sh 35b        # 127.0.0.1:10362  (MoE: experts partly on CPU)

# 4) Demo end-to-end
scripts/quickstart.sh

# 5) Your own data (JSONL schema in docs/protocol.md)
python -m open_medical_jev evaluate --items mydata.jsonl \
    --servers http://127.0.0.1:10361,http://127.0.0.1:10362 --out results/run1
```

## CLI

| command | what it does |
|---|---|
| `selftest` | dependency-free sanity checks (16 cases) |
| `demo` | router demo on synthetic inputs |
| `check-server` / `verify-tokenizer` | health, tokenizer equivalence check |
| `read` | one item through both readers + router → decision JSON |
| `evaluate` | batch run on your labelled JSONL → rows + summary dict |
| `calibrate` | recalibrate the conformal table from your own results |
| `flip` | option-order flip rate (choice readout) |

## What's in the box

```
src/open_medical_jev/   protocol · reader · fusion · router · calibration · metrics · evaluate · cli
recipes/                models.lock.yaml (sources + sha256) · routing.yaml (constants)
scripts/                setup_env · download_models · serve_model · quickstart · check_data_policy
docs/                   method · protocol · evaluation · deploy · comparison
tests/                  stdlib-only tests + hand-written toy fixtures
reports/                results_summary.md
assets/                 architecture.svg · architecture.png · coverage_accuracy.svg · coverage_accuracy.png
```

## Status and scope (v0.1)

* This is a **research release**: code + recipe, measured on the sets listed
  above. It is not a medical device and must not be used for diagnosis or
  treatment (`NOTICE` → Scope and safety).
* **Not included on purpose**: corpora, exam papers, per-item data for
  non-redistributable sets, and any trained weights (there are none).
  See [DATA_POLICY.md](DATA_POLICY.md).
* **Development note**: this project was built with extensive AI assistance
  (LLM coding agents); every number reported here is reproducible from the
  shipped scripts and pinned recipes.
* Planned next: weighted multi-signal fusion (the router is fit-free today),
  more languages, and serving recipes for more hardware tiers.

## Independence

Open Medical Jev is an independent project. Not affiliated with TypeSafe;
"Jev" is their product. Not affiliated with Medical-OpenJev, MedJev,
ClinicalJev or the OpenJev project — see [docs/comparison.md](docs/comparison.md)
for how they relate.

## License

MIT ([LICENSE](LICENSE), [NOTICE](NOTICE)). The Qwen3.5 base models
are Apache-2.0 by Alibaba / Qwen team; the GGUF quantizations come from the
Unsloth HF repositories; llama.cpp (MIT) is the serving runtime. None of these
are bundled with this repository.
