"""Fusion — combine per-option probabilities from several readers/signals.

Part of Open Medical Jev (MIT).

Two small, deterministic combiners are provided:

* :func:`mean_probs` — average the per-option probabilities of the signals
  (this is the "2-model mean fusion" used in the shipped results tables).
* :func:`consensus`  — the consensus-tier policy: each signal votes for an
  option; unanimity (± margin) releases automatically, weaker agreement is
  routed to tiers. (Measured on dev-300: L1 50% @ ~98%, L2 66.7% @ 94.5%,
  L3 85% @ 88.2% coverage @ accuracy.)

Fitting weighted fusion on labelled data is out of scope for v0.1 — the
shipped configuration is deliberately fit-free. If you calibrate your own
weights, keep the recipe in ``recipes/`` so results stay reproducible.
"""

from __future__ import annotations

import math
from collections import Counter

__all__ = ["mean_probs", "consensus"]

MARGIN_THRESHOLD = 0.25  # from the measured L1 policy


def _norm(p: dict) -> dict:
    total = sum(v for v in p.values() if v is not None)
    if not total:
        return {k: 0.0 for k in p}
    return {k: (v / total if v is not None else 0.0) for k, v in p.items()}


def mean_probs(dicts) -> dict:
    """Elementwise mean of probability dicts (same option keys)."""
    dicts = [_norm(d) for d in dicts]
    keys = list(dicts[0].keys())
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def _margin(p: dict) -> float:
    s = sorted(p.values(), reverse=True)
    return s[0] - s[1] if len(s) > 1 else s[0]


def consensus(score_dicts: list) -> dict:
    """Consensus tiers over S signals.

    ``score_dicts``: one per-option probability dict per signal.

    Returns ``{"answer", "tier", "route", "votes", "n_signals", "margin",
    "confidence"}`` where tier/route follow:

    ======  ==================================  ============
    tier    rule                                 route
    ======  ==================================  ============
    L1      unanimous and margin >= 0.25         auto
    L2      unanimous                            auto
    L3      votes >= ceil(0.75 * S)              auto
    --      otherwise                            human review
    ======  ==================================  ============
    """
    n_sig = len(score_dicts)
    keys = list(score_dicts[0].keys())
    preds = [max(keys, key=lambda k: sd[k]) for sd in score_dicts]
    answer, votes = Counter(preds).most_common(1)[0]
    fused = mean_probs(score_dicts)
    margin = _margin(fused)
    unanimous = votes == n_sig
    need = math.ceil(0.75 * n_sig)
    if unanimous and margin >= MARGIN_THRESHOLD:
        tier, route = "L1", "auto"
    elif unanimous:
        tier, route = "L2", "auto"
    elif votes >= need:
        tier, route = "L3", "auto"
    else:
        tier, route = "-", "human"
    return {"answer": answer, "tier": tier, "route": route, "votes": votes,
            "n_signals": n_sig, "margin": round(margin, 4),
            "confidence": round(max(fused.values()), 4)}
