"""Router — turn two models' per-option probabilities into one decision.

Part of Open Medical Jev (Apache-2.0).

The router consumes per-option probabilities produced by the reader (one
probability per candidate option, once for each of the two models) and returns
a single decision object::

    {"answer", "confidence", "auto", "conformal_set", "single_option",
     "agree", "note"}

Three built-in behaviours
-------------------------
1. **Combined confidence** — if the two models pick the same option, average
   their top-1 probabilities; otherwise apply a strong discount (x0.25).
   (The discount was measured: on the mixed-agreement stratum the raw accuracy
   is far below the same-model-agreement stratum; see docs/evaluation.md.)
2. **Auto-release gate** — Chow's rule: release automatically when
   ``confidence >= 1 - cost_ratio`` (``cost_ratio`` = human cost / error cost);
   otherwise recommend human review.
3. **Guaranteed candidate set** — split-conformal prediction set at target
   error rate ``epsilon`` using calibration quantiles (``CONFORMAL_Q``).

Calibration provenance
----------------------
The default ``CONFORMAL_Q`` constants were calibrated on an in-house Chinese
medical licensing-exam set (600 items, split-half calibration; conservative
values kept so coverage does not fall below nominal on the measured sets).
**For a different task distribution, recalibrate with your own labelled data**
— see :func:`open_medical_jev.calibration.conformal_quantile` and
``docs/protocol.md``.

The constants mirror the measured configuration and should stay consistent
with ``recipes/routing.yaml``.
"""

from __future__ import annotations

import math

__all__ = ["Router", "synth_confidence", "conformal_set", "CONFORMAL_Q"]

#: Calibration quantiles q(epsilon) for the conformal set. The threshold on
#: confidence is ``1 - q``; a set is ``{option : p_option >= 1 - q}`` (falling
#: back to the top-1 option if empty).
CONFORMAL_Q = {0.01: 0.9965, 0.05: 0.9350, 0.10: 0.8200, 0.20: 0.6000}

#: Common human-cost / error-cost ratios for the auto-release gate.
COST_RATIOS = (0.05, 0.10, 0.20, 0.40)

#: Discount applied to the combined confidence when the two models disagree.
AGREEMENT_DISCOUNT = 0.25


def _norm(p: dict) -> dict:
    """Normalise a probability dict (sums to 1; degenerate input -> uniform)."""
    s = sum(p.values())
    if not s or s <= 0:
        n = max(1, len(p))
        return {k: 1.0 / n for k in p}
    return {k: v / s for k, v in p.items()}


def synth_confidence(p_a: dict, p_b: dict):
    """Combined confidence of two models.

    Returns ``(confidence, answer_a, answer_b, agree)``.

    * agree    -> mean of the two top-1 probabilities
    * disagree -> ``min(top1_a, top1_b) * AGREEMENT_DISCOUNT``
    """
    p_a, p_b = _norm(p_a), _norm(p_b)
    ca, cb = max(p_a.values()), max(p_b.values())
    aa = max(p_a, key=lambda k: p_a[k])
    ab = max(p_b, key=lambda k: p_b[k])
    agree = aa == ab
    conf = (ca + cb) / 2.0 if agree else min(ca, cb) * AGREEMENT_DISCOUNT
    return conf, aa, ab, agree


def conformal_set(p: dict, eps: float, q: dict | None = None) -> list:
    """Split-conformal candidate set at target error rate ``eps``.

    ``{option : p_option >= 1 - q(eps)}``; if the set is empty it falls back to
    the top-1 option (a size-1 set, not a guarantee).
    """
    q = CONFORMAL_Q if q is None else q
    if eps not in q:
        raise ValueError(f"unsupported epsilon={eps}; available: {sorted(q)}")
    p = _norm(p)
    thr = 1.0 - q[eps]
    s = [k for k, v in p.items() if v >= thr]
    return s or [max(p, key=lambda k: p[k])]


class Router:
    """Two-model router: probabilities in, decision out."""

    def __init__(self, conf_scale: float = 1.0):
        """``conf_scale``: global confidence scaling (default 1.0 = no scaling)."""
        self.conf_scale = float(conf_scale)

    def route(self, p_a: dict, p_b: dict, cost_ratio: float = 0.10,
              epsilon: float = 0.05) -> dict:
        """Route one item.

        Parameters
        ----------
        p_a, p_b : dict
            Per-option probabilities from model A / model B (need not be
            normalised; keys must match).
        cost_ratio : float
            Human cost / error cost; auto-release when confidence >= 1 - it.
        epsilon : float
            Target error rate for the conformal set (0.01 / 0.05 / 0.10 / 0.20).

        Returns
        -------
        dict with keys: answer, confidence, auto, conformal_set,
        single_option, agree, note.
        """
        conf, a_a, a_b, agree = synth_confidence(p_a, p_b)
        conf = min(1.0, conf * self.conf_scale)
        auto = conf >= (1.0 - cost_ratio)
        na, nb = _norm(p_a), _norm(p_b)
        if agree:
            p_use = na
        else:
            keys = set(na) | set(nb)
            p_use = {k: (na.get(k, 0.0) + nb.get(k, 0.0)) / 2.0 for k in keys}
        cs = conformal_set(p_use, epsilon)
        return {
            "answer": max(p_use, key=lambda k: p_use[k]),
            "confidence": round(conf, 4),
            "auto": bool(auto),
            "conformal_set": cs,
            "single_option": len(cs) == 1,
            "agree": agree,
            "note": ("auto-release" if auto else "recommend human review")
                    + ("; singleton set" if len(cs) == 1 else f"; {len(cs)}-option set"),
        }
