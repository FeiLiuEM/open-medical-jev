# Evaluation

How the numbers in [reports/results_summary.md](../reports/results_summary.md)
are produced, and how to produce compatible numbers on your own data.

## Decisions, not strings

The unit of evaluation is a **judgment**: each candidate option receives a
probability (`p_yes` for the pair readout, normalized letter mass for the
choice readout). An item's decision is the argmax over its options; everything
downstream (accuracy, gates, sets) is arithmetic over those probabilities.

## Metric definitions

| metric | definition |
|---|---|
| full-set accuracy | share of items whose decision equals the gold option |
| coverage @ accuracy | keep items with `confidence ≥ τ` (or a rule); report both the kept share (coverage) and accuracy within the kept set |
| router tiers | `strict` = readers agree and margin in the top half of agreed items; `unanimous` = readers agree (strict ⊂ this); `split` = readers disagree (recommend human review) |
| auto-release gate | Chow's rule: auto when `confidence ≥ 1 − cost_ratio`; report coverage and accuracy of the auto stratum |
| conformal coverage | share of items whose gold option is inside the candidate set; report the average set size too |
| ECE | expected calibration error, 10 equal-width bins on decision confidence (binning convention matters — ours is 10 equal-width) |
| flip rate | share of items whose answer changes under option reordering (choice readout, n permutations) |

## Baselines used in the shipped tables

* **Jev (TypeSafe 1.13.0)**: compared on identical item sets where its outputs
  could be obtained under our protocol; its numbers are 100%-coverage by
  construction, so gated rows always show coverage.
* **open-weight peers** (e.g. OpenJev): quoted from their own published
  numbers where marked; not re-measured here unless stated.

## Reproducing / extending

```bash
# point at your two servers; items in the schema of docs/protocol.md
python -m open_medical_jev evaluate --items mydata.jsonl \
    --servers http://127.0.0.1:10361,http://127.0.0.1:10362 \
    --readout pair --concurrency 8 --out results/mine

# outputs: results/mine.jsonl (per-item rows) + results/mine.summary.json
```

The summary contains: full accuracy (fused and per reader), router tiers,
auto-gate coverage/accuracy, conformal coverage/set size, ECE. Recalibrate the
conformal table for your distribution with `calibrate` (see protocol.md).

## Reading discipline

1. Never mix distributions (dev-300 ≠ JevBench ≠ jdb).
2. Never mix configurations (single reader ≠ 2-reader ≠ 4-reading).
3. Gated numbers require their coverage.
4. Fit on calibration splits only; report out-of-sample.
5. `selftest` and `verify-tokenizer` before trusting a run.
