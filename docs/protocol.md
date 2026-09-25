# Protocol

This file pins the **exact** measurement protocol. Prompts are built in
`src/open_medical_jev/protocol.py`; keep the two in sync — prompt drift
changes results.

## Item schema (JSONL)

```json
{"id": "q-0001",
 "input": "question text",
 "context": "optional context block (public sets: used when present)",
 "options": [{"key": "A", "text": "..."}, {"key": "B", "text": "..."},
             {"key": "C", "text": "..."}, {"key": "D", "text": "..."}],
 "answer": "B"}
```

`context` and `answer` are optional. `answer` is required for accuracy
metrics; it is ignored by `read`.

## Readout 1 — pair (primary contract)

For **each candidate option**, one forward pass. The rendered prompt (golden
example, verified token-by-token against the reference implementation):

```
<|im_start|>system
You are a careful medical expert. You will be given a question and one candidate answer. Judge whether that candidate answer correctly answers the question.<|im_end|>
<|im_start|>user
Question: What is 2+2?

Candidate answer: 4

Is the candidate answer correct? Reply yes or no.<|im_end|>
<|im_start|>assistant
<think>

</think>

Answer: 
```

Notes:

* The system/user text is built from `PAIR_SYS` + `head_text(item)` +
  `Candidate answer: <option>` + `PAIR_TAIL` (see `protocol.py`).
* The assistant turn contains the **empty thinking block**
  (`<think>\n\n</think>\n\n`) produced by `enable_thinking=False` in the
  Qwen3.5 chat template, followed by the literal prefix `Answer: `.
* Context blocks (when present) are capped at 6000 characters.

**Reading rule.** At the first generated position, take the top-`n_probs`
logprobs and locate the single-token variants of `yes` and `no`
(`"yes"`, `" yes"` / `"no"`, `" no"`):

```
delta  = max_logprob(yes variants) − max_logprob(no variants)
p_yes  = sigmoid(delta)          # clipped to ±30 for safety
```

An option whose yes/no tokens are both absent from the top-`n_probs` is
recorded as an *uncertain reading* and reported, never silently zeroed.

**Decoding parameters** (all readouts): `n_predict=1`, `temperature=0.0`,
`cache_prompt=false`, `n_probs=200`.

## Readout 2 — choice (secondary)

One forward pass with all options listed; read the probability mass on each
option letter:

```
<|im_start|>system
You are a licensed medical professional taking a national licensing exam. Answer each question with the single best option.
<|im_end|>
<|im_start|>user
<input>

A. ...
B. ...
C. ...
D. ...

Answer with exactly one option letter: A, B, C, D. Output only the letter.<|im_end|>
<|im_start|>assistant
```

**Reading rule.** For each letter, combine the logprobs of its surface
variants (`"A"`, `" A"`, `"A."`, `" A."`, lowercase, …) with log-sum-exp, then
softmax over letters. Prompts are truncated to 3600 tokens from the front
when longer.

## Tokenizer

* The Qwen3.5 family shares its vocabulary; prompts can be tokenized with any
  family member's tokenizer.
* Default backend: the server's own `/tokenize` endpoint
  (`add_special=false`, `parse_special=true`) — no extra dependencies.
* Optional backend: a HuggingFace tokenizer (`--tokenizer-backend hf`,
  requires `transformers`).
* Verify equivalence on your setup **before trusting a reproduction**:

```bash
python -m open_medical_jev verify-tokenizer --server http://127.0.0.1:10361 \
    --hf-repo Qwen/Qwen3.5-27B
```

## Scoring and routing

1. Per-option probabilities are normalised (sum to 1).
2. Fusion (default): elementwise **mean** across readers.
3. Router (`router.py`): combined confidence = mean of the two top-1
   probabilities when the readers agree, `min(top1) × 0.25` when they
   disagree; auto-release when `confidence ≥ 1 − cost_ratio`; conformal set
   `{option : p ≥ 1 − q(ε)}`.

## Evaluation discipline

* **Fit on one split, report on another.** Thresholds / calibration constants
  are fitted on a calibration split and reported out-of-sample (the shipped
  numbers use dev-A → dev-B or split-half).
* **Coverage and accuracy travel together.** Any gated number without its
  coverage is meaningless (the reference Jev numbers are 100% coverage).
* **Conformal validity needs exchangeability.** Same distribution required;
  under shift, expect undercoverage and recalibrate.

## Recalibration (do this for your distribution)

```bash
# 1. Run the batch evaluator on your labelled data (writes rows + summary)
python -m open_medical_jev evaluate --items mydata.jsonl \
    --servers http://127.0.0.1:10361,http://127.0.0.1:10362 --out results/mine

# 2. Fit the conformal quantile table from your rows
python -m open_medical_jev calibrate --results results/mine.jsonl \
    --write recipes/routing_calibrated.json

# 3. Use the table (pass it as `q` to router.conformal_set, or update routing.yaml)
```

The same procedure applies to the confidence discount and any fitted
temperature: fit on your calibration split, report out-of-sample.

## Reproducibility checklist

1. `python -m open_medical_jev selftest` → all cases pass.
2. GGUF files match `recipes/models.lock.yaml` sha256.
3. `verify-tokenizer` reports `equal: true` (or stick to `--tokenizer-backend hf`).
4. Report coverage for anything gated; keep calibration and test splits separate.
