"""Reader tests against a tiny mock llama-server (no models needed).

Run:  python3 tests/test_reader_mock.py
Part of Open Medical Jev (Apache-2.0).
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# Local mock servers are loopback; make sure proxy env vars don't capture them
# (the package also bypasses proxies for loopback explicitly — belt and braces).
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

from open_medical_jev.evaluate import evaluate  # noqa: E402
from open_medical_jev.reader import Reader, verify_tokenizer  # noqa: E402

YES_ID, NO_ID = 4242, 4243
LETTER_IDS = {"A": 51, "B": 52, "C": 53, "D": 54}


def _lp(tid, token, logprob):
    return {"id": tid, "token": token, "logprob": logprob}


class MockHandler(BaseHTTPRequestHandler):
    """Canned llama-server responses.

    mode="uniform": the same flat distribution for every request (default).
    mode="cycle":   rotating distribution — one strong "yes" every 4th
                    request, weak otherwise (deterministic with
                    concurrency=1; simulates a clear winner).
    """

    mode = "uniform"
    counter = 0

    def log_message(self, *a):  # silence
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
            letters = [_lp(LETTER_IDS[k], k, -0.5 - 0.2 * i)
                       for i, k in enumerate("ABCD")]
            if MockHandler.mode == "cycle":
                pos = MockHandler.counter % 4
                MockHandler.counter += 1
                if pos == 0:
                    tlp = [_lp(YES_ID, "yes", -0.05), _lp(NO_ID, "no", -4.0)] + letters
                else:
                    tlp = [_lp(YES_ID, "yes", -3.6), _lp(NO_ID, "no", -0.2)] + letters
            else:
                tlp = [_lp(YES_ID, "yes", -0.1), _lp(NO_ID, "no", -2.0)] + letters
            self._send({"completion_probabilities": [{"top_logprobs": tlp}]})
        else:
            self._send({}, 404)


ITEM = {"id": "mock-1", "input": "Q?", "options": [
    {"key": "A", "text": "x"}, {"key": "B", "text": "y"},
    {"key": "C", "text": "z"}, {"key": "D", "text": "w"}], "answer": "A"}


def _serve():
    MockHandler.mode = "uniform"
    MockHandler.counter = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_reader_pair():
    srv = _serve()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        out = Reader(base).read_pair(ITEM)
        assert all(v is not None for v in out["p_yes"].values()), out
        # delta = -0.1 - (-2.0) = 1.9  ->  sigmoid ≈ 0.870
        assert abs(out["p_yes"]["A"] - 0.8701) < 0.005, out
    finally:
        srv.shutdown()


def test_reader_choice():
    srv = _serve()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        out = Reader(base).read_choice(ITEM)
        assert out["probs"]["A"] > out["probs"]["D"] > 0, out
    finally:
        srv.shutdown()


def test_evaluate_plumbing():
    """Full pipeline with a clear winner: auto-release + singleton set."""
    srv = _serve()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        MockHandler.mode = "cycle"
        MockHandler.counter = 0
        rows, summ = evaluate([ITEM], [base, base], concurrency=1, log=lambda *a: None)
        assert summ["n_complete"] == 1, summ
        assert rows[0]["pred"] == "A", rows
        assert rows[0]["agree"] is True and rows[0]["auto"] is True, rows
        assert rows[0]["conformal_set"] == ["A"], rows
        assert rows[0]["correct"] == 1, rows
    finally:
        MockHandler.mode = "uniform"
        MockHandler.counter = 0
        srv.shutdown()


def test_evaluate_uniform_is_low_confidence():
    """Uniform probabilities must NOT auto-release (router semantics)."""
    srv = _serve()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        rows, summ = evaluate([ITEM], [base, base], concurrency=1, log=lambda *a: None)
        assert rows[0]["auto"] is False, rows
        assert rows[0]["confidence"] == 0.25, rows
        assert len(rows[0]["conformal_set"]) == 4, rows
    finally:
        srv.shutdown()


def test_verify_tokenizer_graceful():
    srv = _serve()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        out = verify_tokenizer(base, "Qwen/Qwen3.5-27B")
        # without transformers/cache: hf_error is reported, server side works
        assert out["server_len"] > 0
        assert ("equal" in out)
    finally:
        srv.shutdown()


if __name__ == "__main__":
    test_reader_pair()
    test_reader_choice()
    test_evaluate_plumbing()
    test_evaluate_uniform_is_low_confidence()
    test_verify_tokenizer_graceful()
    print("mock tests OK ✓ (5/5)")
