"""Evaluate — batch pipeline: items -> two readers -> fusion -> router -> summary.

Part of Open Medical Jev (MIT).

Input items are JSON objects (JSONL file) in the schema documented in
``docs/protocol.md``::

    {"id": "q-0001",
     "input": "question text",
     "context": "optional context block",
     "options": [{"key": "A", "text": "..."}, ...],
     "answer": "B"}          # optional gold key; required for accuracy metrics

Output: per-item rows (JSONL) + a summary dict, both written by the CLI.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from . import fusion as fusion_mod
from . import metrics as metrics_mod
from .reader import Reader, make_tokenizer
from .router import Router

__all__ = ["read_all", "evaluate"]


def read_all(items, server, readout="pair", tokenizer_backend="server",
             hf_repo="Qwen/Qwen3.5-27B", concurrency=8, n_probs=200,
             max_tokens=0, log=None):
    """Read signals for all items from one server. Returns per-item dicts."""
    tok = make_tokenizer(tokenizer_backend, server, hf_repo)
    reader = Reader(server, tok, n_probs=n_probs, max_tokens=max_tokens)
    out = [None] * len(items)

    def job(i):
        if readout == "choice":
            return i, reader.read_choice(items[i])
        return i, reader.read_pair(items[i])

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        for i, res in ex.map(job, range(len(items))):
            out[i] = res
            done += 1
            if log and done % 25 == 0:
                log(f"  read {done}/{len(items)}")
    return out


def _probs_of(res, readout):
    return res["probs"] if readout == "choice" else res["p_yes"]


def evaluate(items, servers, readout="pair", tokenizer_backend="server",
             hf_repo="Qwen/Qwen3.5-27B", concurrency=8, n_probs=200,
             max_tokens=0, cost_ratio=0.10, epsilon=0.05, log=print):
    """Run the full pipeline on labelled items; returns (rows, summary)."""
    assert len(servers) == 2, "exactly two model servers (A, B)"
    t0 = time.time()
    log(f"[evaluate] {len(items)} items | readout={readout} | A={servers[0]} | B={servers[1]}")
    res_a = read_all(items, servers[0], readout, tokenizer_backend, hf_repo,
                     concurrency, n_probs, max_tokens, log=log)
    res_b = read_all(items, servers[1], readout, tokenizer_backend, hf_repo,
                     concurrency, n_probs, max_tokens, log=log)

    router = Router()
    rows = []
    for it, ra, rb in zip(items, res_a, res_b):
        pa, pb = _probs_of(ra, readout), _probs_of(rb, readout)
        ok = (all(v is not None for v in pa.values())
              and all(v is not None for v in pb.values()))
        row = {"id": it.get("id"), "gold": it.get("answer"),
               "p_a": pa, "p_b": pb, "errors_a": ra.get("errors", []),
               "errors_b": rb.get("errors", [])}
        if ok:
            fused = fusion_mod.mean_probs([pa, pb])
            dec = router.route(pa, pb, cost_ratio=cost_ratio, epsilon=epsilon)
            pred = dec["answer"]
            row.update({"fused": fused, "pred": pred,
                        "confidence": dec["confidence"], "auto": dec["auto"],
                        "agree": dec["agree"], "conformal_set": dec["conformal_set"]})
            if it.get("answer"):
                row["correct"] = int(pred == it["answer"])
                row["in_set"] = int(it["answer"] in dec["conformal_set"])
        else:
            row.update({"pred": None, "confidence": None, "auto": None,
                        "agree": None, "conformal_set": None,
                        "correct": (int(False) if it.get("answer") else None)})
        rows.append(row)

    complete = [r for r in rows if r["pred"] is not None]
    labelled = [r for r in complete if r.get("correct") is not None]
    summary = {
        "n_items": len(rows),
        "n_complete": len(complete),
        "n_labelled": len(labelled),
        "readout": readout,
        "seconds": round(time.time() - t0, 1),
        "ms_per_item": round(1000 * (time.time() - t0) / max(1, len(rows)), 1),
    }
    if labelled:
        corr = [r["correct"] for r in labelled]
        summary["accuracy_fused"] = round(sum(corr) / len(corr), 4)
        summary["accuracy_model_a"] = round(
            sum(int(max(r["p_a"], key=lambda k: r["p_a"][k]) == r["gold"]) for r in labelled) / len(labelled), 4)
        summary["accuracy_model_b"] = round(
            sum(int(max(r["p_b"], key=lambda k: r["p_b"][k]) == r["gold"]) for r in labelled) / len(labelled), 4)
        conf = [r["confidence"] for r in labelled]
        margins = [sorted(r["fused"].values(), reverse=True)[0]
                   - sorted(r["fused"].values(), reverse=True)[1] for r in labelled]
        summary["r2_tiers"] = metrics_mod.tier_summary(
            [r["agree"] for r in labelled], margins, corr)
        auto = [r for r in labelled if r["auto"]]
        summary["auto_gate"] = {
            "coverage": round(len(auto) / len(labelled), 4),
            "acc": round(sum(r["correct"] for r in auto) / max(1, len(auto)), 4),
        }
        summary["conformal"] = {
            "epsilon": epsilon,
            "coverage": round(sum(r["in_set"] for r in labelled) / len(labelled), 4),
            "avg_set_size": round(sum(len(r["conformal_set"]) for r in labelled) / len(labelled), 2),
            "singleton_share": round(
                sum(1 for r in labelled if len(r["conformal_set"]) == 1) / len(labelled), 4),
        }
        summary["ece"] = round(metrics_mod.ece(conf, corr, bins=10), 4)
    return rows, summary
