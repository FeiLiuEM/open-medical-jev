# Results summary

All numbers below were produced with the frozen, unmodified models (no
fine-tuning, no distillation) using the protocols pinned in
[docs/protocol.md](../docs/protocol.md). Results were measured in September
2026; code in this repository is the cleaned extraction of the same pipeline.

**Aggregate-only**: some item sets used in measurement cannot be redistributed
(see [DATA_POLICY.md](../DATA_POLICY.md)). What is published here are the
aggregate tables; the harness is shipped so you can re-run on your own data.

## Item sets

| set | what it is | size | notes |
|---|---|---|---|
| dev-300 | a 300-item subset of MedMCQA (medical MCQ, pick the correct option) | 300 | subset file not redistributed |
| JevBench original-public | the public decision-primitive benchmark ("noul/choice" style) | 72 (we completed 70) | official Jev: 0.9860 |
| jev-decision-bench (jdb) | public multi-task decision set (BoolQ/ARC/XNLI/GSM8K …) | 916 completed (39 tasks) | Jev per-task mean ≈0.86 |

## Accuracy (no gating — plain argmax)

| set | 27B | 35B-A3B | 2-model mean fusion | Jev | gap (fusion vs Jev) |
|---|---|---|---|---|---|
| dev-300 | 0.8000 | 0.7933 | **0.8167** | 0.8300 | −1.3 pp |
| JevBench | 0.8143 | 0.7857 | 0.8143 | 0.9860 | −17.2 pp |
| jdb | 0.8395 | 0.8046 | 0.8373 | ~0.86 | −2 to −3 pp |

## Router gating (coverage @ accuracy)

Format: **coverage @ accuracy** — the router releases automatically only for
the covered stratum; the rest is routed to human review. Never compare a gated
number against a 100%-coverage number without the coverage.

| set | strict (agree & top-half margin) | unanimous (models agree) | split (→ human review) | full-set acc |
|---|---|---|---|---|
| dev-300 | 39.3% @ 0.9915 | **78.7% @ 0.9025** | 21.3% @ 0.4219 | 0.8000 |
| JevBench | 41.4% @ 0.9310 | **82.9% @ 0.9310** | 17.1% @ 0.5833 | 0.8714 |
| jdb | 43.6% @ 0.9574 | **87.1% @ 0.8697** | 12.9% @ 0.6356 | 0.8395 |

A 4-readings variant (both readers × both readout structures; dev-300 only):
strict 22.0% @ 0.9848, unanimous 66.7% @ 0.9457. The 4-reading and 2-reading
configurations are not interchangeable.

## Calibration (probability quality)

Expected calibration error (10 equal-width bins on decision confidence):

| config | dev-300 | JevBench | jdb |
|---|---|---|---|
| 27B | 0.0627 | 0.0956 | 0.0620 |
| 35B-A3B | 0.0460 | 0.0668 | 0.0276 |
| 2-model fusion | 0.0322 | 0.0844 | 0.0258 |
| fusion + tier-conditioned temperature (agree T=0.95 / split T=1.25, fitted on dev-300) | 0.0360 | 0.0896 | **0.0096** |
| Jev (published reference) | 0.027 | 0.027 | — |

Notes: isotonic regression scored 0.0000 in-fit but degraded out-of-distribution
(JevBench 0.0928, jdb 0.0320) and was **not adopted**; the 1-parameter
temperature version was (see `recipes/routing.yaml` for constants, and
`recalibrate` in docs/protocol.md to fit your own).

## Guaranteed candidate sets (exam-paper example)

On a Chinese medical licensing-exam set (600 items), the split-conformal
candidate set at each target error rate ε (calibration split-half in-set):

| ε | measured coverage (target ≥ 1−ε) | avg set size |
|---|---|---|
| 0.01 | 99.33% | 2.95 |
| 0.05 | 97.33% | 1.57 |
| 0.10 | 93.33% | 1.18 |
| 0.20 | 88.00% | 1.00 (singleton) |

Auto-release on the same set: Chow gate (cost ratio 0.10) released 53.3% of
items at 98.12% accuracy; precision-optimised threshold reached 59.0% coverage
at ≥98% precision.

**Distribution-shift caveat (important)**: these conformal quantiles were
calibrated on the Chinese set. Applied to other distributions they can
*undercover* — e.g. an Indian-set check at ε=0.05 measured 92.83% coverage
(< 95% nominal). Recalibrate per distribution (docs/protocol.md).

## Throughput (as measured)

Single 24 GB GPU, llama.cpp, GGUF Q4:

* ≈0.28 s per question single-flight; ≈0.076 s per question at batch
  concurrency 8 (readout-dependent — re-measure on your hardware).
* pair readout on the 35B-A3B with MoE offload (`--n-cpu-moe 36`):
  ≈0.51 s per judgment at concurrency 8; the 27B fully on GPU is faster.

## Discipline checklist (how to read these tables)

1. The three item sets have very different distributions — do not mix numbers
   across them.
2. 2-reading, 4-reading and single-model results are not interchangeable.
3. Gated numbers must always carry their coverage.
4. All "ours" numbers come from unmodified official models; no trained copy
   contributes to any number in this file.
