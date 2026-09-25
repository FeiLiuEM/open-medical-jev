"""Selftest — dependency-free sanity checks for the whole package.

Run with ``python -m open_medical_jev selftest`` (no models or network needed).
"""

from __future__ import annotations

from . import calibration, fusion, metrics, protocol
from .router import Router, conformal_set, synth_confidence

__all__ = ["run"]


def _case_router():
    r = Router()
    o1 = r.route({"A": 0.02, "B": 0.95, "C": 0.02, "D": 0.01},
                 {"A": 0.03, "B": 0.93, "C": 0.03, "D": 0.01},
                 cost_ratio=0.10, epsilon=0.05)
    assert o1["auto"] and o1["answer"] == "B" and o1["single_option"], o1


def _case_disagree():
    r = Router()
    o2 = r.route({"A": 0.02, "B": 0.95, "C": 0.02, "D": 0.01},
                 {"A": 0.90, "B": 0.05, "C": 0.03, "D": 0.02},
                 cost_ratio=0.10, epsilon=0.05)
    assert (not o2["auto"]) and (not o2["agree"]), o2


def _case_conformal_monotone():
    p = {"A": 0.60, "B": 0.25, "C": 0.10, "D": 0.05}
    assert len(conformal_set(p, 0.20)) <= len(conformal_set(p, 0.05)), p


def _case_norm_tolerance():
    conf, _, _, _ = synth_confidence({"A": 1, "B": 9}, {"A": 1, "B": 8})
    assert 0.0 <= conf <= 1.0, conf


def _case_protocol_golden():
    s = protocol.pair_prompt_string({"input": "What is 2+2?"}, "4")
    expected = (
        '<|im_start|>system\nYou are a careful medical expert. You will be given a '
        'question and one candidate answer. Judge whether that candidate answer '
        'correctly answers the question.<|im_end|>\n'
        '<|im_start|>user\nQuestion: What is 2+2?\n\nCandidate answer: 4\n\n'
        'Is the candidate answer correct? Reply yes or no.<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ')
    assert s == expected, repr(s)


def _case_protocol_choice():
    s = protocol.choice_prompt_string({"input": "Q?", "options": [
        {"key": "A", "text": "x"}, {"key": "B", "text": "y"}]})
    assert "Answer with exactly one option letter: A, B." in s
    assert s.endswith("<|im_start|>assistant\n"), repr(s[-40:])


def _case_consensus_l1():
    out = fusion.consensus([{"A": .02, "B": .96, "C": .01, "D": .01},
                            {"A": .05, "B": .90, "C": .03, "D": .02},
                            {"A": .02, "B": .94, "C": .02, "D": .02}])
    assert out["tier"] == "L1" and out["route"] == "auto" and out["answer"] == "B", out


def _case_consensus_l3():
    out = fusion.consensus([{"A": .02, "B": .96, "C": .01, "D": .01},
                            {"A": .02, "B": .96, "C": .01, "D": .01},
                            {"A": .02, "B": .96, "C": .01, "D": .01},
                            {"A": .90, "B": .02, "C": .05, "D": .03}])
    assert out["tier"] == "L3" and out["route"] == "auto", out


def _case_consensus_human():
    out = fusion.consensus([{"A": .02, "B": .96, "C": .01, "D": .01},
                            {"A": .02, "B": .96, "C": .01, "D": .01},
                            {"A": .90, "B": .02, "C": .05, "D": .03},
                            {"A": .04, "B": .02, "C": .90, "D": .04}])
    assert out["tier"] == "-" and out["route"] == "human", out


def _case_coverage_at_precision():
    cov, acc = metrics.coverage_at_precision([0.9, 0.8, 0.3], [1, 1, 0], 0.5)
    assert abs(cov - 2 / 3) < 1e-9 and abs(acc - 1.0) < 1e-9, (cov, acc)


def _case_fit_tau():
    tau, cov, acc = metrics.fit_tau_for_target([0.9, 0.7, 0.5, 0.3],
                                               [1, 1, 0, 0], 0.9)
    assert acc is not None and acc >= 0.9 and 0 < cov <= 1.0, (tau, cov, acc)


def _case_conformal_tau():
    tau = metrics.conformal_tau([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0], 0.5)
    assert 0.6 <= tau <= 0.9, tau


def _case_quantile():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    q = calibration.conformal_quantile(scores, 0.10)
    assert q == 1.0, q  # (n+1)(1-eps)=9.9 -> ceil 10 -> clamped to last


def _case_ece_zero():
    assert metrics.ece([1.0] * 5, [1] * 5, bins=10) == 0.0


def _case_tier_summary():
    t = metrics.tier_summary([True, True, False], [0.8, 0.2, 0.5], [1, 0, 0])
    assert t["unanimous"]["n"] == 2 and t["split"]["n"] == 1, t


def _case_lse_sigmoid():
    from .reader import lse, sigmoid
    assert abs(sigmoid(0.0) - 0.5) < 1e-12
    assert abs(lse([0.0, 0.0]) - 0.6931471805599453) < 1e-9


CASES = [
    ("router: agree -> auto + singleton", _case_router),
    ("router: disagree -> not auto", _case_disagree),
    ("router: conformal set monotone in epsilon", _case_conformal_monotone),
    ("router: normalisation tolerance", _case_norm_tolerance),
    ("protocol: pair prompt golden string", _case_protocol_golden),
    ("protocol: choice prompt structure", _case_protocol_choice),
    ("fusion: consensus L1", _case_consensus_l1),
    ("fusion: consensus L3", _case_consensus_l3),
    ("fusion: consensus -> human", _case_consensus_human),
    ("metrics: coverage@precision", _case_coverage_at_precision),
    ("metrics: tau fitting", _case_fit_tau),
    ("metrics: conformal tau", _case_conformal_tau),
    ("calibration: conformal quantile", _case_quantile),
    ("metrics: ECE zero on calibrated", _case_ece_zero),
    ("metrics: tier summary", _case_tier_summary),
    ("reader: lse / sigmoid", _case_lse_sigmoid),
]


def run() -> int:
    failures = []
    for name, fn in CASES:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures.append((name, repr(e)))
    if failures:
        print(f"SELFTEST FAILED ({len(CASES) - len(failures)}/{len(CASES)})")
        for name, err in failures:
            print(f"  ✗ {name}: {err}")
        return 1
    print(f"SELFTEST OK ✓ ({len(CASES)}/{len(CASES)})")
    return 0
