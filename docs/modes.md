# Modes

Three compute presets over the same frozen readers. They differ only in **how
much of the pipeline runs per item** — never in the models, prompts, or
router. Everything below was measured on the three 600-item exam sets with the
frozen models (nothing trained), 2026-09.

| mode | what runs per item | speed | measured (China / US / India) |
|---|---|---|---|
| `fast` | one choice readout (35B-A3B); every item answered from that single reading | **≈0.076 s/item** at 8-way concurrency (single-flight ≈0.28 s)\* | accuracy 0.8667 / 0.7352 / 0.7533 |
| `general` | quick pass + calibrated release gate; released items decided immediately, the rest run the **full four-reading path** | ≈1.5 / 2.7 / 2.3 of 4 reading waves per item | releases **84.2 / 44.2 / 55.8 %** of items at **93.07 / 93.13 / 93.13 %** precision; saves **≈63 / 33 / 42 %** of compute |
| `high` | same cascade, stricter release gate | ≈2.0 / 2.9 / 3.1 of 4 reading waves per item | releases **65.5 / 35.4 / 29.7 %** at **97.20 / 97.14 / 97.19 %** precision; saves **≈49 / 27 / 22 %** of compute |

\* Speed measured on one 24 GB GPU machine running the 35B-A3B GGUF
(`-np 8`, the shipped `scripts/serve_model.sh` configuration); re-measure on
your hardware (see `docs/deploy.md`).

## What the cascade does

1. **Quick pass** — one `choice` readout from reader **B** (the faster
   35B-A3B, by convention) for every item.
2. **Release gate** — if the top-1 probability of that reading clears the
   mode's threshold τ, the item is decided right there
   (`stage: "quick"`, `auto: true`). No pair readouts, no second model.
3. **Full path** — everything else gets the complete four-reading treatment:
   A choice, A pair, B pair (B choice already read), each reader's two
   structures averaged into one per-option distribution, then the two-reader
   router produces the answer, combined confidence, auto-release flag and
   conformal set (`stage: "full"`).

The shipped thresholds: τ = **0.636** (`general`, target release precision
93 %) and **0.838** (`high`, target 97 %), on the quick-pass top-1
probability.

## Measured tables

Release/precision/saving per paper (2026-09 exam sets, 600 items each):

| paper | mode | released share | released precision | saving vs full path (waves / raw passes) |
|---|---|---|---|---|
| China | general | 84.2 % | 93.07 % | ≈63 % / ≈77 % |
| China | high | 65.5 % | 97.20 % | ≈49 % / ≈60 % |
| US | general | 44.2 % | 93.13 % | ≈33 % / ≈40 % |
| US | high | 35.4 % | 97.14 % | ≈27 % / ≈32 % |
| India | general | 55.8 % | 93.13 % | ≈42 % / ≈50 % |
| India | high | 29.7 % | 97.19 % | ≈22 % / ≈27 % |

The full path's own accuracy on the same items: 0.8917 / 0.8617 / 0.8083
(China / US / India) — within 0.3 pp of the shipped four-reading numbers in
`reports/results_summary.md` (the small difference is the fit-free
per-reader averaging used by the staged pipeline).

## Compute accounting

"Reading waves" count how many batched forward passes a pipeline schedules per
item: one `choice` readout ≈ one `pair` readout ≈ **1 wave** (the per-option
calls of a pair read are independent and are scheduled together on the
server's parallel slots); the full path = **4 waves**. Saving = 1 −
(waves per item) / 4. "Raw passes" is the same saving counted without
batching, with a pair read counted as K single-option passes (K = number of
options: 5 on the Chinese paper, 4 on the others).

These are estimates, not a wall-clock promise: mixed workloads, different
prompt lengths and different hardware shift the numbers.

## Using it

```bash
# fast: one server (the fast reader)
python -m open_medical_jev evaluate --items mydata.jsonl --mode fast \
    --servers http://127.0.0.1:10362

# general / high: both readers (A = 27B, B = 35B-A3B); the quick pass runs on B
python -m open_medical_jev evaluate --items mydata.jsonl --mode general \
    --servers http://127.0.0.1:10361,http://127.0.0.1:10362

# override the release threshold (e.g. after your own calibration)
python -m open_medical_jev evaluate --items mydata.jsonl --mode general \
    --gate 0.71 --servers http://127.0.0.1:10361,http://127.0.0.1:10362
```

The run summary reports `stage_share`, `accuracy_quick` / `accuracy_full`,
`est_reading_waves_per_item` and `est_saving_vs_full_path` alongside the usual
metrics. Per-item rows carry `stage` (`quick` / `full`) and the quick-pass
probabilities (`p_quick`).

## Calibration and scope

* τ is fitted on **released precision** (accuracy of the items that skip the
  full path). It is distribution-specific. The shipped defaults were
  calibrated on the Chinese exam set; the same procedure on the other two
  sets gives τ ≈ 0.78 (US `general`) / 0.79 (India `general`) and
  τ ≈ 0.84 / 0.93 (`high`) — the defaults deliberately stay conservative
  across sets.
* **Recalibrating on your own data**: run the full pipeline once on labelled
  items (`--readout choice` from the fast reader is enough for the quick-pass
  scores), then choose the threshold that hits your target release precision
  (sort the quick-pass top-1 probabilities, scan the threshold, keep the
  largest coverage whose running precision ≥ target). This is the same
  split-half discipline used for the conformal table.
* The full path keeps the system honest on hard items: overall accuracy in
  `general`/`high` stays near the full-pipeline numbers — what changes is
  *where the compute goes*, not the model stack.
* Modes never change the models, prompts, or router constants; if you replace
  a base model, re-fit τ together with the two routing constants.

Keep `recipes/routing.yaml` in sync when you change any threshold.
