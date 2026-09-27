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


# ---------------------------------------------------------------------------
# staged modes (fast / general / high)
# ---------------------------------------------------------------------------

def evaluate_mode(items, servers, mode="general", tokenizer_backend="server",
                  hf_repo="Qwen/Qwen3.5-27B", concurrency=8, n_probs=200,
                  max_tokens=0, cost_ratio=0.10, epsilon=0.05, tau=None,
                  log=print):
    """Staged evaluation for a compute preset (``fast`` / ``general`` / ``high``).

    ``servers``: one URL for ``fast`` (the fast reader); two ``(A, B)``
    otherwise. The quick pass runs on **B** — by convention the faster
    35B-A3B.

    Stage 1 reads one choice readout from B for every item. Items whose top-1
    probability clears the mode's release threshold are decided immediately
    (``stage="quick"``); the rest run the full four-reading path — A choice,
    A pair, B pair (B choice already read) — and are decided by the two-reader
    router (``stage="full"``).

    Returns ``(rows, summary)``; ``summary["est_saving_vs_full_path"]``
    estimates the compute saved against running the full path on every item
    (batched reading-wave accounting, see ``modes.READING_WAVES``).
    """
    from . import modes as modes_mod

    assert mode in modes_mod.MODES, f"unknown mode {mode!r}"
    thr = modes_mod.release_threshold(mode, tau)
    t0 = time.time()

    if mode == "fast":
        assert len(servers) == 1, "fast mode takes exactly one server (the fast reader)"
        res_b = read_all(items, servers[0], "choice", tokenizer_backend, hf_repo,
                         concurrency, n_probs, max_tokens, log=log)
        rows = []
        for it, rb in zip(items, res_b):
            row = {"id": it.get("id"), "gold": it.get("answer"),
                   "stage": "quick", "p_quick": rb["probs"],
                   "errors": rb.get("errors", []),
                   "auto": True, "agree": None, "conformal_set": None}
            try:
                pred, conf = modes_mod.top1(rb["probs"])
                row.update({"pred": pred, "confidence": round(conf, 4)})
                if it.get("answer"):
                    row["correct"] = int(pred == it["answer"])
            except Exception:
                row.update({"pred": None, "confidence": None,
                            "correct": (0 if it.get("answer") else None)})
            rows.append(row)
        return rows, _mode_summary(rows, mode, thr, t0)

    # -- cascade: quick pass + gate, then the full path for the rest --------
    assert len(servers) == 2, "general/high modes take two servers: A,B"
    res_b = read_all(items, servers[1], "choice", tokenizer_backend, hf_repo,
                     concurrency, n_probs, max_tokens, log=log)
    released, rest_idx = {}, []
    for i, rb in enumerate(res_b):
        try:
            pred, conf = modes_mod.top1(rb["probs"])
        except Exception:
            rest_idx.append(i)
            continue
        (released.__setitem__(i, (pred, conf)) if conf >= thr
         else rest_idx.append(i))

    rows = [None] * len(items)
    for i, (pred, conf) in released.items():
        it = items[i]
        row = {"id": it.get("id"), "gold": it.get("answer"), "stage": "quick",
               "p_quick": res_b[i]["probs"], "errors": res_b[i].get("errors", []),
               "pred": pred, "confidence": round(conf, 4), "auto": True,
               "agree": None, "conformal_set": None}
        if it.get("answer"):
            row["correct"] = int(pred == it["answer"])
        rows[i] = row

    if rest_idx:
        rest = [items[i] for i in rest_idx]
        log(f"[mode={mode}] full path for {len(rest)}/{len(items)} items")
        ra_c = read_all(rest, servers[0], "choice", tokenizer_backend, hf_repo,
                        concurrency, n_probs, max_tokens, log=log)
        ra_p = read_all(rest, servers[0], "pair", tokenizer_backend, hf_repo,
                        concurrency, n_probs, max_tokens, log=log)
        rb_p = read_all(rest, servers[1], "pair", tokenizer_backend, hf_repo,
                        concurrency, n_probs, max_tokens, log=log)
        router = Router()
        for j, i in enumerate(rest_idx):
            it = items[i]
            row = {"id": it.get("id"), "gold": it.get("answer"), "stage": "full",
                   "p_quick": res_b[i]["probs"],
                   "errors": (ra_c[j].get("errors", []) + ra_p[j].get("errors", [])
                              + rb_p[j].get("errors", [])),
                   "p_a": None, "p_b": None, "confidence": None, "auto": None,
                   "agree": None, "conformal_set": None}
            try:
                pa = modes_mod.combine_dual(ra_c[j]["probs"], ra_p[j]["p_yes"])
                pb = modes_mod.combine_dual(res_b[i]["probs"], rb_p[j]["p_yes"])
            except Exception:
                pa = pb = None
            if pa and pb and any(v for v in pa.values()) and any(v for v in pb.values()):
                dec = router.route(pa, pb, cost_ratio=cost_ratio, epsilon=epsilon)
                row.update({"p_a": pa, "p_b": pb, "pred": dec["answer"],
                            "confidence": dec["confidence"], "auto": dec["auto"],
                            "agree": dec["agree"],
                            "conformal_set": dec["conformal_set"]})
                if it.get("answer"):
                    row["correct"] = int(dec["answer"] == it["answer"])
                    row["in_set"] = int(it["answer"] in dec["conformal_set"])
            else:
                row.update({"p_a": pa, "p_b": pb, "pred": None,
                            "correct": (0 if it.get("answer") else None)})
            rows[i] = row

    return rows, _mode_summary(rows, mode, thr, t0)


def _mode_summary(rows, mode, thr, t0):
    """Summary dict for a staged-mode run (shares shape with `evaluate`)."""
    from . import modes as modes_mod

    labelled = [r for r in rows if r.get("correct") is not None]
    quick = [r for r in labelled if r["stage"] == "quick"]
    full = [r for r in labelled if r["stage"] == "full"]
    n = len(rows)
    waves = (len(quick) * modes_mod.READING_WAVES["quick"]
             + len(full) * modes_mod.READING_WAVES["full"])
    denom = max(1, len(labelled))
    summary = {
        "mode": mode,
        "gate_tau": thr,
        "n_items": n,
        "n_labelled": len(labelled),
        "seconds": round(time.time() - t0, 1),
        "ms_per_item": round(1000 * (time.time() - t0) / max(1, n), 1),
        "stage_share": {"quick": round(len(quick) / denom, 4),
                        "full": round(len(full) / denom, 4)},
        "est_reading_waves_per_item": round(waves / denom, 3),
        "est_saving_vs_full_path": round(1.0 - waves / denom / modes_mod.READING_WAVES["full"], 4),
    }
    if labelled:
        corr = [r["correct"] for r in labelled]
        summary["accuracy"] = round(sum(corr) / len(corr), 4)
        for name, grp in (("quick", quick), ("full", full)):
            if grp:
                summary[f"accuracy_{name}"] = round(
                    sum(r["correct"] for r in grp) / len(grp), 4)
    return summary
