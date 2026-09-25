"""Calibration — split-conformal quantiles for the guaranteed candidate set.

Part of Open Medical Jev (MIT).

The router's default ``CONFORMAL_Q`` constants were calibrated on an in-house
Chinese medical licensing-exam set. **For a different task distribution,
recalibrate on your own labelled data**:

1. Run the reader over a labelled calibration set (same readout structure).
2. Compute per-item scores ``s = 1 - p(gold)`` where ``p`` is the (fused)
   per-option probability vector and ``gold`` is the correct option key.
3. Take the split-conformal order statistic for each target error rate
   ``epsilon`` with :func:`conformal_quantile`.
4. Use the resulting dict as the ``q`` argument of
   :func:`open_medical_jev.router.conformal_set` (or update
   ``recipes/routing.yaml``).

Split-conformal validity relies on exchangeability between calibration and
test items (same distribution). Under distribution shift, coverage can fall
below the nominal level — recalibrate per distribution.
"""

from __future__ import annotations

import math

__all__ = ["conformal_quantile", "calibration_scores", "fit_quantile_table"]


def calibration_scores(items) -> list:
    """Per-item conformal scores from labelled results.

    ``items``: iterable of ``{"probs": {key: p, ...}, "gold": key}``.
    Returns ``[1 - p[gold], ...]``.
    """
    out = []
    for it in items:
        p = it["probs"]
        gold = it["gold"]
        total = sum(v for v in p.values() if v is not None)
        if not gold or gold not in p or p[gold] is None or total <= 0:
            continue
        out.append(1.0 - (p[gold] / total))
    return sorted(out)


def conformal_quantile(scores, eps: float) -> float:
    """Split-conformal order statistic: the q such that threshold = 1 - q.

    Standard formula: ``q = s_(ceil((n+1)(1-eps)) - 1)`` (0-indexed, clamped).
    """
    s = sorted(scores)
    n = len(s)
    if n == 0:
        raise ValueError("empty calibration set")
    idx = min(n - 1, max(0, math.ceil((n + 1) * (1 - eps)) - 1))
    return s[idx]


def fit_quantile_table(items, epsilon_values=(0.01, 0.05, 0.10, 0.20)) -> dict:
    """Recalibrate the full quantile table from labelled items."""
    scores = calibration_scores(items)
    return {eps: conformal_quantile(scores, eps) for eps in epsilon_values}
