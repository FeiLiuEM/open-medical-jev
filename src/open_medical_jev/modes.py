"""Modes — speed / accuracy presets over the same frozen readers.

Part of Open Medical Jev (MIT).

The project ships three modes. They differ only in **how much of the
pipeline runs per item** — never in the models, prompts or router:

===========  ==============================================================
``fast``     one choice readout from the faster model (35B-A3B), no gate:
             every item is answered from that single reading.
             (Measured ≈0.076 s/item at 8-way concurrency.)
``general``  cascade — the fast readout first; items whose top-1 probability
             clears a calibrated release threshold are decided immediately;
             everything else runs the full four-reading path (both models ×
             both readout structures) through the two-reader router.
``high``     the same cascade with a stricter release threshold: fewer
             items skip the full path, higher precision on the released
             stratum.
===========  ==============================================================

Measured on the three 600-item exam sets (frozen models, nothing trained;
full tables in ``docs/modes.md``) — at the shipped defaults:

* ``general`` (τ = 0.636): releases 84.2 / 44.2 / 55.8 % of items at
  93.07 / 93.13 / 93.13 % precision (China / US / India), saving
  ≈63 / 33 / 42 % of the compute of running the full path everywhere.
* ``high`` (τ = 0.838): releases 65.5 / 35.4 / 29.7 % at
  97.20 / 97.14 / 97.19 %, saving ≈49 / 27 / 22 %.

The thresholds were calibrated on an in-house Chinese medical
licensing-exam set (600 items, split-half discipline, 2026-09) and are
therefore safest for similar distributions. **Recalibrate on your own
labelled data** — see ``docs/modes.md`` for the procedure and commands.
"""

from __future__ import annotations

from .fusion import mean_probs

__all__ = ["MODES", "GATE_TAU_DEFAULT", "READING_WAVES",
           "release_threshold", "top1", "combine_dual"]

MODES = ("fast", "general", "high")

#: Release thresholds on the quick-pass top-1 probability (35B-A3B choice
#: readout). Calibrated at target release precision 93 % (``general``) /
#: 97 % (``high``). Keep in sync with ``recipes/routing.yaml``.
GATE_TAU_DEFAULT = {"general": 0.636, "high": 0.838}

#: Compute accounting used for the saving estimates, in *batched reading
#: waves*: one choice readout ≈ one pair readout ≈ 1 wave when the per-option
#: calls are scheduled together on the server; the full path = 4 waves.
READING_WAVES = {"quick": 1.0, "full": 4.0}


def release_threshold(mode: str, tau: float | None = None) -> float | None:
    """Release threshold for a mode.

    ``None`` means no gate (``fast`` releases everything). ``tau`` overrides
    the built-in default.
    """
    if mode == "fast":
        return None
    if mode not in GATE_TAU_DEFAULT:
        raise ValueError(f"unknown mode: {mode!r}; expected one of {MODES}")
    return GATE_TAU_DEFAULT[mode] if tau is None else float(tau)


def top1(probs: dict) -> tuple:
    """Normalise a per-option dict and return ``(top option, probability)``.

    Raises ``ValueError`` when nothing readable is present.
    """
    p = {k: v for k, v in probs.items() if v is not None}
    total = sum(p.values())
    if not p or total <= 0:
        raise ValueError("no readable probabilities")
    p = {k: v / total for k, v in p.items()}
    a = max(p, key=lambda k: p[k])
    return a, p[a]


def combine_dual(choice_probs: dict, pair_probs: dict) -> dict:
    """One reader's two readout structures → one per-option distribution.

    Fit-free elementwise mean — the same spirit as the two-model fusion the
    router consumes.
    """
    return mean_probs([choice_probs, pair_probs])
