# Method

Open Medical Jev turns two frozen open models into one decision, with
guarantees attached to the parts that can carry them.

```
 item (question + candidate options, optional context)
        │
        ├── reader A: Qwen3.5-27B    (GGUF, llama.cpp)  ── pair/choice readout
        └── reader B: Qwen3.5-35B-A3B (GGUF, llama.cpp) ── pair/choice readout
        │
        ▼
 per-option probabilities  ──►  fusion (mean)  ──►  router
                                                     ├─ combined confidence
                                                     ├─ auto-release gate
                                                     └─ conformal set
        │
        ▼
 decision: {answer, confidence, auto, conformal_set, …}
```

## Why this shape

* **The capability is already in the generalist model.** The models are used
  exactly as released (official Apache-2.0 weights, GGUF quantization for
  local serving). Nothing about the judgment task is trained: no fine-tuning,
  no distillation, no task-specific head.
* **Two readers, one signal each, then arithmetic.** Each reader answers a
  deliberately narrow question per candidate option ("is this correct?"); the
  system's behavior comes from combining readings deterministically. This
  keeps the pipeline auditable: every step is a probability, a mean, or a
  threshold.
* **Where guarantees are possible, they are made.** The candidate set uses
  split-conformal prediction, which carries an error-rate guarantee *under
  exchangeability*. Where the guarantee cannot hold (distribution shift,
  swapped readers), the documentation says so and the recalibration path is
  shipped.

## Components

| module | role | notes |
|---|---|---|
| `protocol.py` | exact prompt construction for the two readout structures | the contract; mirrored in docs/protocol.md |
| `reader.py` | speaks to `llama-server` (`/completion`, `/tokenize`); parses top-logprobs | no third-party deps by default |
| `fusion.py` | mean fusion + consensus tiers | deterministic, fit-free |
| `router.py` | combined confidence, Chow gate, conformal set | stdlib only |
| `calibration.py` | split-conformal quantiles for your own data | see "Recalibration" |
| `metrics.py` / `evaluate.py` | coverage@accuracy, conformal checks, ECE, batch runs | |
| `selftest.py` / `tests/` | dependency-free checks of every piece above | run before reporting |

## What the method does not do

* No free-form generation: outputs are per-option probabilities, a decision,
  and a candidate set. (This is deliberate — see the output contract.)
* No open-ended questions, no long reasoning chains; the readouts are single
  yes/no (or option-letter) judgments.
* No learned router weights in v0.1: the router is fit-free. If you fit
  fusion weights, keep them in `recipes/` so results stay reproducible.
* Calibration constants are only as good as their calibration distribution —
  see the caveat in `reports/results_summary.md`.

## Where to look next

* prompts & reading rules → [protocol.md](protocol.md)
* numbers & caveats → [results_summary.md](../reports/results_summary.md)
* serving on one 24 GB GPU → [deploy.md](deploy.md)
* relationship to other "Jev" projects → [comparison.md](comparison.md)
