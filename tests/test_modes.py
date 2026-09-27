"""Mode tests — cascade gating logic + staged pipeline against mock servers.

Run:  python3 tests/test_modes.py
Part of Open Medical Jev (MIT).

The mock llama-server returns canned distributions; both servers point at the
same mock. Completion calls are counted to verify the cascade really does
what the docs claim item by item:

* a released item costs exactly 1 completion (the quick choice readout);
* an escalated item costs 1 + 1 + K + K completions
  (B choice + A choice + A pair (K options) + B pair (K options)),
  K = number of options.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

from open_medical_jev import modes as modes_mod  # noqa: E402
from open_medical_jev.evaluate import evaluate_mode  # noqa: E402

YES_ID, NO_ID = 4242, 4243
LETTER_IDS = {"A": 51, "B": 52, "C": 53, "D": 54}


def _lp(tid, token, logprob):
    return {"id": tid, "token": token, "logprob": logprob}


class MockHandler(BaseHTTPRequestHandler):
    """Canned llama-server responses.

    modes:
      uniform — flat letters (top-1 ≈ 0.33): never released by any gate.
      sharp   — A strongly dominant (top-1 ≈ 0.95): released by both gates.
      mild    — A dominant but not extreme (top-1 ≈ 0.664): released by
                ``general`` (τ=0.636) but NOT by ``high`` (τ=0.838).
    """

    mode = "uniform"
    calls = 0

    def log_message(self, *a):
        pass

    def _send(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        else:
            self._send({}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/tokenize":
            content = body.get("content", "")
            if content in ("yes", "no", " yes", " no"):
                self._send({"tokens": [YES_ID if content.strip() == "yes" else NO_ID]})
                return
            if content in LETTER_IDS:
                self._send({"tokens": [LETTER_IDS[content]]})
                return
            self._send({"tokens": [100 + (ord(c) % 50) for c in content]})
        elif self.path == "/completion":
            MockHandler.calls += 1
            if MockHandler.mode == "sharp":
                lp = {"A": -0.05, "B": -4.0, "C": -4.0, "D": -4.0}
            elif MockHandler.mode == "mild":
                lp = {"A": -0.02, "B": -1.8, "C": -1.8, "D": -1.8}
            else:
                lp = {"A": -0.5, "B": -0.7, "C": -0.9, "D": -1.1}
            letters = [_lp(LETTER_IDS[k], k, lp[k]) for k in "ABCD"]
            tlp = [_lp(YES_ID, "yes", -0.1), _lp(NO_ID, "no", -2.0)] + letters
            self._send({"completion_probabilities": [{"top_logprobs": tlp}]})
        else:
            self._send({}, 404)


ITEM = {"id": "mock-mode-1", "input": "Q?", "options": [
    {"key": "A", "text": "x"}, {"key": "B", "text": "y"},
    {"key": "C", "text": "z"}, {"key": "D", "text": "w"}], "answer": "A"}


def _serve(mode="uniform"):
    MockHandler.mode = mode
    MockHandler.calls = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _run(mode_server, mode_use, servers_n=2):
    srv = _serve(mode_server)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        servers = [base] if servers_n == 1 else [base, base]
        rows, summ = evaluate_mode([ITEM], servers, mode=mode_use,
                                   concurrency=1, log=lambda *a: None)
        return rows, summ, MockHandler.calls
    finally:
        MockHandler.mode = "uniform"
        MockHandler.calls = 0
        srv.shutdown()


# ---------------------------------------------------------------------------
# pure gate logic
# ---------------------------------------------------------------------------

def test_thresholds():
    assert modes_mod.release_threshold("fast") is None
    assert modes_mod.release_threshold("general") == 0.636
    assert modes_mod.release_threshold("high") == 0.838
    assert modes_mod.release_threshold("high", tau=0.9) == 0.9
    try:
        modes_mod.release_threshold("nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_top1_and_combine():
    a, c = modes_mod.top1({"A": 2.0, "B": 1.0, "C": None, "D": 1.0})
    assert a == "A" and abs(c - 0.5) < 1e-9, (a, c)
    m = modes_mod.combine_dual({"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0})
    assert abs(m["A"] - 0.5) < 1e-9 and abs(m["B"] - 0.5) < 1e-9, m


# ---------------------------------------------------------------------------
# staged pipeline (mock servers)
# ---------------------------------------------------------------------------

def test_fast_mode_single_server():
    rows, summ, calls = _run("sharp", "fast", servers_n=1)
    assert calls == 1, f"fast must read exactly once per item, got {calls}"
    assert rows[0]["stage"] == "quick" and rows[0]["pred"] == "A", rows
    assert summ["mode"] == "fast" and summ["gate_tau"] is None, summ
    assert summ["accuracy"] == 1.0, summ
    # wave accounting: 1 wave/item vs 4 for the full path -> 75 % saving
    assert abs(summ["est_saving_vs_full_path"] - 0.75) < 1e-9, summ


def test_general_releases_high_confidence():
    rows, summ, calls = _run("sharp", "general")
    assert calls == 1, f"released item must not trigger the full path, got {calls}"
    assert rows[0]["stage"] == "quick", rows
    assert summ["stage_share"] == {"quick": 1.0, "full": 0.0}, summ
    assert abs(summ["est_saving_vs_full_path"] - 0.75) < 1e-9, summ


def test_general_escalates_low_confidence():
    rows, summ, calls = _run("uniform", "general")
    # 1 (B choice) + 1 (A choice) + 4 (A pair) + 4 (B pair) = 10
    assert calls == 10, f"escalated item must run the full path, got {calls}"
    assert rows[0]["stage"] == "full", rows
    assert summ["stage_share"] == {"quick": 0.0, "full": 1.0}, summ
    assert abs(summ["est_reading_waves_per_item"] - 4.0) < 1e-9, summ
    assert summ["est_saving_vs_full_path"] == 0.0, summ


def test_high_is_stricter_than_general():
    # mild mock: top-1 ≈ 0.664 -> general releases, high escalates
    rows, summ, calls = _run("mild", "general")
    assert rows[0]["stage"] == "quick" and calls == 1, (rows[0]["stage"], calls)
    rows_h, summ_h, calls_h = _run("mild", "high")
    assert rows_h[0]["stage"] == "full" and calls_h == 10, (rows_h[0]["stage"], calls_h)


if __name__ == "__main__":
    test_thresholds()
    test_top1_and_combine()
    test_fast_mode_single_server()
    test_general_releases_high_confidence()
    test_general_escalates_low_confidence()
    test_high_is_stricter_than_general()
    print("mode tests OK ✓ (6/6)")
