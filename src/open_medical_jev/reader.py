"""Reader — query llama.cpp servers and turn them into per-option signals.

Part of Open Medical Jev (Apache-2.0).

Two readout structures are supported, matching the measured protocol (see
``docs/protocol.md``):

* ``pair``   — per candidate option, ask "is this correct?" and read yes/no
  probabilities at the first output position.
  ``delta = max_logprob(yes) - max_logprob(no)``;  ``p_yes = sigmoid(delta)``.
* ``choice`` — one forward pass with all options listed; read the probability
  mass on each option letter.

Token ids are obtained either from the llama.cpp server itself
(``/tokenize`` endpoint — default, no extra dependencies) or from a
HuggingFace tokenizer (``--tokenizer-backend hf``). Both paths are equivalent
for the Qwen3.5 family; check your setup with
``python -m open_medical_jev verify-tokenizer``.

Servers are plain ``llama-server`` instances, e.g.::

    llama-server -m Qwen3.5-27B-UD-Q4_K_XL.gguf -ngl 99 -c 8192 --port 10361
"""

from __future__ import annotations

import json
import math
import urllib.request

from . import protocol

__all__ = [
    "ServerTokenizer", "HFTokenizer", "Reader",
    "server_health", "verify_tokenizer", "sigmoid", "lse",
]


def sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _opener(url: str):
    """Request opener that bypasses any configured proxy for loopback addresses.

    Python's urllib honours proxy environment variables but does not exempt
    localhost reliably on every platform — a proxied loopback request usually
    fails with HTTP 503. Serving here is loopback by design, so the bypass is
    explicit; remote servers still honour proxy settings.
    """
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    if host in _LOOPBACK_HOSTS:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _post(url: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with _opener(url).open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def server_health(server: str) -> bool:
    try:
        req = urllib.request.Request(server.rstrip("/") + "/health")
        with _opener(server).open(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# tokenizers
# ---------------------------------------------------------------------------

class ServerTokenizer:
    """Tokenize through the llama.cpp server (``/tokenize``)."""

    def __init__(self, server: str, timeout: int = 60):
        self.server = server.rstrip("/")
        self.timeout = timeout

    def ids(self, text: str) -> list:
        d = _post(self.server + "/tokenize",
                  {"content": text, "add_special": False, "parse_special": True},
                  timeout=self.timeout)
        toks = d.get("tokens") if isinstance(d, dict) else d
        out = []
        for t in toks:
            out.append(int(t) if isinstance(t, int) else int(t.get("id")))
        return out

    def single_token_ids(self, word: str) -> list:
        out = []
        for w in (word, " " + word):
            ids = self.ids(w)
            if len(ids) == 1:
                out.append(ids[0])
        return sorted(set(out))


class HFTokenizer:
    """Tokenize with a HuggingFace tokenizer (optional dependency)."""

    def __init__(self, repo: str):
        from transformers import AutoTokenizer  # lazy import
        self.tok = AutoTokenizer.from_pretrained(repo)

    def ids(self, text: str) -> list:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def single_token_ids(self, word: str) -> list:
        out = []
        for w in (word, " " + word):
            t = self.tok(w, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                out.append(t[0])
        return sorted(set(out))

    def token_string_of(self, token_id: int):
        try:
            return self.tok.convert_ids_to_tokens(token_id)
        except Exception:
            return None


def make_tokenizer(backend: str, server: str, hf_repo: str):
    if backend == "hf":
        return HFTokenizer(hf_repo)
    return ServerTokenizer(server)


# ---------------------------------------------------------------------------
# completion parsing
# ---------------------------------------------------------------------------

def _first_top_logprobs(out: dict):
    cp = out.get("completion_probabilities") or []
    if not cp:
        return []
    return cp[0].get("top_logprobs") or []


def _lut_by_id(out: dict):
    return {int(x["id"]): float(x["logprob"]) for x in _first_top_logprobs(out)}


def _lut_by_token(out: dict):
    return {x.get("token"): float(x["logprob"]) for x in _first_top_logprobs(out)}


def _letter_variants(tokenizer, lab) -> list:
    s = str(lab)
    out = [s, " " + s, s + ".", " " + s + ".", s.lower(), " " + s.lower()]
    if hasattr(tokenizer, "token_string_of"):
        ids = tokenizer.ids(s)
        if ids:
            p = tokenizer.token_string_of(ids[0])
            for c in (p, (p or "").lstrip("\u0120\u2581"), (p or "").strip()):
                if c:
                    out.append(c)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

class Reader:
    """Read per-option signals from one llama.cpp server."""

    def __init__(self, server: str, tokenizer=None, n_probs: int = 200,
                 timeout: int = 600, max_tokens: int = 0):
        self.server = server.rstrip("/")
        self.tokenizer = tokenizer if tokenizer is not None else ServerTokenizer(server)
        self.n_probs = n_probs
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._yes_ids = None
        self._no_ids = None

    # -- internals ---------------------------------------------------------

    def _yesno_ids(self):
        if self._yes_ids is None:
            self._yes_ids = self.tokenizer.single_token_ids("yes")
            self._no_ids = self.tokenizer.single_token_ids("no")
            if not self._yes_ids or not self._no_ids:
                raise RuntimeError("could not locate single-token yes/no ids")
        return self._yes_ids, self._no_ids

    def _completion(self, ids):
        if self.max_tokens and len(ids) > self.max_tokens:
            ids = ids[-self.max_tokens:]  # keep the tail (most recent context)
        return _post(self.server + "/completion",
                     {"prompt": ids, "n_predict": 1, "n_probs": self.n_probs,
                      "temperature": 0.0, "cache_prompt": False},
                     timeout=self.timeout)

    # -- pair readout ------------------------------------------------------

    def read_pair_option(self, item: dict, option_text: str):
        """One candidate option -> (delta, p_yes). None if unreadable."""
        text = protocol.pair_prompt_string(item, option_text)
        ids = self.tokenizer.ids(text)
        lut = _lut_by_id(self._completion(ids))
        yes_ids, no_ids = self._yesno_ids()
        ys = [lut[t] for t in yes_ids if t in lut]
        ns = [lut[t] for t in no_ids if t in lut]
        if not ys or not ns:
            return None
        delta = max(ys) - max(ns)
        return delta, sigmoid(delta)

    def read_pair(self, item: dict) -> dict:
        """All options of one item -> {"delta", "p_yes", "errors"}."""
        deltas, probs, errors = {}, {}, []
        for o in item["options"]:
            try:
                r = self.read_pair_option(item, o["text"])
            except Exception as e:  # network/parse errors
                r = None
                errors.append(f"{o['key']}: {str(e)[:120]}")
            if r is None:
                deltas[o["key"]] = None
                probs[o["key"]] = None
                errors.append(f"{o['key']}: yes/no not in top-{self.n_probs}")
            else:
                deltas[o["key"]], probs[o["key"]] = r
        return {"delta": deltas, "p_yes": probs, "errors": errors}

    # -- choice readout ----------------------------------------------------

    def read_choice(self, item: dict) -> dict:
        """All options in one pass -> {"probs", "errors"} (softmax over letters)."""
        text = protocol.choice_prompt_string(item)
        ids = self.tokenizer.ids(text)
        lut = _lut_by_token(self._completion(ids))
        got = {}
        for o in item["options"]:
            vals = [lut[s] for s in _letter_variants(self.tokenizer, o["key"]) if s in lut]
            got[o["key"]] = lse(vals) if vals else None
        if not any(v is not None for v in got.values()):
            return {"probs": {k: None for k in got}, "errors": ["labels not in top-%d" % self.n_probs]}
        mx = max(v for v in got.values() if v is not None)
        ex = {k: (math.exp(v - mx) if v is not None else 0.0) for k, v in got.items()}
        s = sum(ex.values()) or 1.0
        return {"probs": {k: v / s for k, v in ex.items()}, "errors": []}


# ---------------------------------------------------------------------------
# tokenizer equivalence check
# ---------------------------------------------------------------------------

def verify_tokenizer(server: str, hf_repo: str, sample_text: str | None = None) -> dict:
    """Compare server-side and HuggingFace tokenization of a sample prompt."""
    sample_text = sample_text or protocol.pair_prompt_string(
        {"input": "What is 2+2?"}, "4")
    st = ServerTokenizer(server)
    server_ids = st.ids(sample_text)
    try:
        ht = HFTokenizer(hf_repo)
        hf_ids = ht.ids(sample_text)
        equal = server_ids == hf_ids
        first_diff = next((i for i, (a, b) in enumerate(zip(server_ids, hf_ids)) if a != b), None)
        return {"server_len": len(server_ids), "hf_len": len(hf_ids),
                "equal": equal, "first_diff_index": first_diff,
                "server_ids_head": server_ids[:8], "hf_ids_head": hf_ids[:8]}
    except Exception as e:
        return {"server_len": len(server_ids), "hf_error": str(e)[:200], "equal": None}
