"""Test suite for Open Medical Jev.

Stdlib-only. Run either way::

    python3 tests/test_package.py     # no dependencies
    pytest tests/                     # if you have pytest

Part of Open Medical Jev (Apache-2.0).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from open_medical_jev import metrics  # noqa: E402
from open_medical_jev.reader import _lut_by_id, _lut_by_token  # noqa: E402
from open_medical_jev.selftest import CASES  # noqa: E402


def test_selftest_cases():
    """Every embedded selftest case passes."""
    for name, fn in CASES:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"selftest case failed: {name}: {e!r}")


def test_ece_perfectly_calibrated():
    assert metrics.ece([1.0] * 5, [1] * 5) == 0.0


def test_coverage_sweep_monotone():
    conf = [0.9, 0.8, 0.7, 0.6, 0.5]
    corr = [1, 1, 0, 1, 0]
    rows = metrics.sweep_taus(conf, corr, taus=[0.5, 0.7, 0.9])
    covs = [r[1] for r in rows]
    assert covs[0] >= covs[1] >= covs[2], rows


def test_reader_parsing():
    out = {"completion_probabilities": [{"top_logprobs": [
        {"id": 1, "token": "yes", "logprob": -0.1},
        {"id": 2, "token": "no", "logprob": -2.0},
    ]}]}
    assert _lut_by_id(out) == {1: -0.1, 2: -2.0}
    assert abs(_lut_by_token(out)["yes"] + 0.1) < 1e-12
    assert _lut_by_id({}) == {}


def test_schema_of_fixture():
    import json
    path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_items.jsonl")
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            assert set(item) >= {"id", "input", "options"}
            assert all(set(o) >= {"key", "text"} for o in item["options"])
            n += 1
    assert n >= 3


if __name__ == "__main__":
    test_selftest_cases()
    test_ece_perfectly_calibrated()
    test_coverage_sweep_monotone()
    test_reader_parsing()
    test_schema_of_fixture()
    from open_medical_jev.selftest import run
    raise SystemExit(run())
