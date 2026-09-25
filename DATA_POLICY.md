# Data Policy

Open Medical Jev is published as **code + recipe** only. This is a hard
boundary, not a preference.

## What this repository contains
- Source code (Apache-2.0)
- Recipes: prompt templates, reader / calibration / routing configuration
- Documentation, small **hand-written** test fixtures, and **aggregate**
  result tables

## What this repository does not contain, by policy
- Training or evaluation corpora of any kind
- Synthetic question/answer text produced from any corpus
- Exam papers or question-set files (e.g., licensing-exam materials)
- Internal research ledgers and raw evaluation logs

## Running evaluations
The evaluation harness runs on **user-supplied data** in a documented JSONL
schema (see `docs/protocol.md`). You are responsible for complying with the
license and terms of any dataset you use.

## Contributing
Do not commit data files. Pull requests containing data, corpus excerpts, or
synthetic text will be rejected. Small hand-written examples used by the test
suite are the only exception; label them clearly as such.
