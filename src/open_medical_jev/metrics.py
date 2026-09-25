"""Metrics — coverage / accuracy views used by the evaluation reports.

Part of Open Medical Jev (Apache-2.0).

All functions here operate on plain Python lists (no numpy required):

* ``conf``    — per-item decision confidence in [0, 1]
* ``correct`` — per-item correctness (0/1, or booleans)
* ``agree``   — per-item two-model agreement (bools)
* ``margins`` — per-item margin between the top-1 and top-2 fused probabilities
"""

from __future__ import annotations

import math

__all__ = [
    "coverage_at_precision", "sweep_taus", "fit_tau_for_target",
    "conformal_tau", "tier_summary", "ece",
]


def coverage_at_precision(conf, correct, tau):
    """Coverage (share kept) and accuracy among items with ``conf >= tau``."""
    keep = [i for i in range(len(conf)) if conf[i] >= tau]
    if not keep:
        return 0.0, None
    return len(keep) / len(conf), sum(correct[i] for i in keep) / len(keep)


def sweep_taus(conf, correct, taus=None):
    """Rows of ``(tau, coverage, accuracy)`` over a grid of thresholds."""
    taus = list(taus) if taus is not None else [i / 100 for i in range(5, 100, 5)]
    return [(t, *coverage_at_precision(conf, correct, t)) for t in taus]


def fit_tau_for_target(conf, correct, target):
    """Largest-coverage tau at which kept-set accuracy >= ``target``.

    Fit on this set (use a calibration split for honest out-of-sample numbers
    — see docs/protocol.md).
    """
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    best = (1.0, 0.0, None)
    for n in range(1, len(order) + 1):
        acc = sum(correct[i] for i in order[:n]) / n
        if acc >= target and n / len(order) > best[1]:
            best = (conf[order[n - 1]], n / len(order), acc)
    return best


def conformal_tau(conf, correct, eps):
    """Split-conformal threshold on decision confidence: (1-eps)-quantile of
    the confidences of correct decisions."""
    ok = sorted([conf[i] for i in range(len(conf)) if correct[i]])
    if not ok:
        return 1.0
    q = max(0, math.ceil((1 - eps) * (len(ok) + 1)) - 1)
    return ok[min(q, len(ok) - 1)]


def tier_summary(agree, margins, correct):
    """R2 tier table (strata partition the item set):

    * ``strict``    — two models agree AND margin in the top half of agreed items
    * ``unanimous`` — two models agree (strict is a subset of this)
    * ``split``     — the models disagree (candidates for human review)
    """
    idx_ag = [i for i in range(len(agree)) if agree[i]]
    idx_sp = [i for i in range(len(agree)) if not agree[i]]
    med = None
    if idx_ag:
        ms = sorted(margins[i] for i in idx_ag)
        med = ms[len(ms) // 2]
    idx_st = [i for i in idx_ag if med is not None and margins[i] >= med]

    def stat(idx):
        if not idx:
            return {"n": 0, "coverage": 0.0, "acc": None}
        return {"n": len(idx), "coverage": len(idx) / len(agree),
                "acc": sum(correct[i] for i in idx) / len(idx)}

    return {"strict": stat(idx_st), "unanimous": stat(idx_ag), "split": stat(idx_sp)}


def ece(conf, correct, bins=10):
    """Expected calibration error, equal-width bins on the confidence.

    Note: reported ECE values depend on the binning convention. The shipped
    numbers were produced with 10 equal-width bins on decision confidence.
    """
    tot = len(conf)
    if not tot:
        return None
    acc = [[0, 0.0, 0.0] for _ in range(bins)]
    for c, ok in zip(conf, correct):
        b = min(bins - 1, max(0, int(c * bins)))
        acc[b][0] += 1
        acc[b][1] += c
        acc[b][2] += ok
    e = 0.0
    for n, csum, osum in acc:
        if not n:
            continue
        e += (n / tot) * abs(osum / n - csum / n)
    return e
