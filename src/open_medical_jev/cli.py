"""Command line interface.

Part of Open Medical Jev (Apache-2.0).

Usage examples::

    # dependency-free self check
    python -m open_medical_jev selftest

    # check a running llama.cpp server
    python -m open_medical_jev check-server --server http://127.0.0.1:10361
    python -m open_medical_jev verify-tokenizer --server http://127.0.0.1:10361 \
        --hf-repo Qwen/Qwen3.5-27B

    # single item through both models + the router
    python -m open_medical_jev read --item examples/item.json \
        --servers http://127.0.0.1:10361,http://127.0.0.1:10362

    # batch evaluation on your own labelled data
    python -m open_medical_jev evaluate --items mydata.jsonl \
        --servers http://127.0.0.1:10361,http://127.0.0.1:10362 \
        --out results/run1

    # recalibrate the conformal table from your own results
    python -m open_medical_jev calibrate --results results/run1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys



def _load_items(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            d = json.load(f)
            return d if isinstance(d, list) else [d]
        return [json.loads(l) for l in f if l.strip()]


def cmd_selftest(args):
    from .selftest import run
    return run()


def cmd_demo(args):
    from .router import Router
    r = Router()
    cases = [
        ("high-confidence agreement", {"A": .02, "B": .95, "C": .02, "D": .01},
         {"A": .03, "B": .93, "C": .03, "D": .01}),
        ("medium agreement", {"A": .10, "B": .75, "C": .10, "D": .05},
         {"A": .12, "B": .72, "C": .10, "D": .06}),
        ("models disagree", {"A": .02, "B": .95, "C": .02, "D": .01},
         {"A": .90, "B": .05, "C": .03, "D": .02}),
    ]
    print("demo (cost_ratio=0.10 | epsilon=0.05)")
    for name, a, b in cases:
        print(f"  [{name}] {json.dumps(r.route(a, b), ensure_ascii=False)}")
    return 0


def cmd_check_server(args):
    from .reader import ServerTokenizer, server_health
    ok = server_health(args.server)
    print(f"health: {'OK' if ok else 'FAILED'} ({args.server})")
    if not ok:
        return 1
    ids = ServerTokenizer(args.server).ids("<|im_start|>user\nping<|im_end|>\n")
    print(f"tokenize: OK ({len(ids)} tokens; head={ids[:6]})")
    return 0


def cmd_verify_tokenizer(args):
    from .reader import verify_tokenizer
    out = verify_tokenizer(args.server, args.hf_repo)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("equal") else 1


def cmd_read(args):
    from . import fusion as fusion_mod
    from .evaluate import read_all
    from .router import Router
    items = _load_items(args.item)
    assert len(items) == 1, "--item expects exactly one item"
    item = items[0]
    servers = args.servers.split(",")
    assert len(servers) == 2, "--servers expects two URLs: A,B"
    res = read_all([item], servers[0], args.readout, args.tokenizer_backend,
                   args.hf_repo, concurrency=1, log=None)
    res_b = read_all([item], servers[1], args.readout, args.tokenizer_backend,
                     args.hf_repo, concurrency=1, log=None)
    pa = res[0]["probs"] if args.readout == "choice" else res[0]["p_yes"]
    pb = res_b[0]["probs"] if args.readout == "choice" else res_b[0]["p_yes"]
    out = {"id": item.get("id"), "p_a": pa, "p_b": pb,
           "errors_a": res[0].get("errors", []), "errors_b": res_b[0].get("errors", [])}
    if all(v is not None for v in pa.values()) and all(v is not None for v in pb.values()):
        out["fused"] = fusion_mod.mean_probs([pa, pb])
        out["decision"] = Router().route(pa, pb, cost_ratio=args.cost_ratio,
                                         epsilon=args.epsilon)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args):
    from .evaluate import evaluate
    items = _load_items(args.items)
    servers = args.servers.split(",")
    rows, summ = evaluate(
        items, servers, readout=args.readout,
        tokenizer_backend=args.tokenizer_backend, hf_repo=args.hf_repo,
        concurrency=args.concurrency, n_probs=args.n_probs,
        max_tokens=args.max_tokens, cost_ratio=args.cost_ratio,
        epsilon=args.epsilon)
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out + ".jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(args.out + ".summary.json", "w", encoding="utf-8") as f:
            json.dump(summ, f, ensure_ascii=False, indent=2)
        print(f"written: {args.out}.jsonl / {args.out}.summary.json")
    return 0


def cmd_calibrate(args):
    from .calibration import fit_quantile_table
    rows = _load_items(args.results)
    items = [{"probs": r["fused"], "gold": r["gold"]}
             for r in rows if r.get("fused") and r.get("gold")]
    if not items:
        print("no labelled rows with fused probabilities found")
        return 1
    table = fit_quantile_table(items)
    print(f"calibration items: {len(items)}")
    print("CONFORMAL_Q = " + json.dumps(table))
    if args.write:
        payload = {"epsilon_to_q": table, "n_calibration": len(items),
                   "note": "recalibrated from user-supplied labelled results"}
        with open(args.write, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"written: {args.write}")
    return 0


def cmd_flip(args):
    from .evaluate import read_all
    from .reader import Reader, make_tokenizer
    import random
    from collections import Counter
    items = _load_items(args.items)[: args.limit or None]
    rng = random.Random(20260925)
    server = args.server
    flip = agree = tot = 0
    for it in items:
        base = [dict(o) for o in it["options"]]
        answers = []
        for s in range(args.n):
            perm = base[:]
            if s > 0:
                rng.shuffle(perm)
            new_item = dict(it)
            new_item["options"] = [{"key": chr(65 + i), "text": t["text"]}
                                   for i, t in enumerate(perm)]
            res = read_all([new_item], server, "choice", args.tokenizer_backend,
                           args.hf_repo, concurrency=1, log=None)[0]
            probs = res["probs"]
            if any(v is None for v in probs.values()):
                continue
            best = max(probs, key=lambda k: probs[k])
            answers.append([o["text"] for o in new_item["options"] if o["key"] == best][0])
        if len(answers) >= 2:
            tot += 1
            c = Counter(answers)
            if c.most_common(1)[0][1] == len(answers):
                agree += 1
            else:
                flip += 1
    if not tot:
        print("no complete items")
        return 1
    print(f"[flip] items={tot} | flip_rate={flip / tot:.4f} | consistent={agree / tot:.4f}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="open_medical_jev",
                                description="Open Medical Jev — routing-based decisions over frozen models")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("selftest", help="dependency-free sanity checks")
    sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("demo", help="router demo on synthetic inputs")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("check-server", help="health + tokenize check")
    sp.add_argument("--server", required=True)
    sp.set_defaults(func=cmd_check_server)

    sp = sub.add_parser("verify-tokenizer", help="compare server vs HF tokenization")
    sp.add_argument("--server", required=True)
    sp.add_argument("--hf-repo", default="Qwen/Qwen3.5-27B")
    sp.set_defaults(func=cmd_verify_tokenizer)

    sp = sub.add_parser("read", help="one item through both models + router")
    sp.add_argument("--item", required=True, help="JSON file with a single item")
    sp.add_argument("--servers", required=True, help="two llama.cpp URLs: A,B")
    sp.add_argument("--readout", choices=["pair", "choice"], default="pair")
    sp.add_argument("--tokenizer-backend", choices=["server", "hf"], default="server")
    sp.add_argument("--hf-repo", default="Qwen/Qwen3.5-27B")
    sp.add_argument("--cost-ratio", type=float, default=0.10)
    sp.add_argument("--epsilon", type=float, default=0.05)
    sp.set_defaults(func=cmd_read)

    sp = sub.add_parser("evaluate", help="batch evaluation on labelled JSONL")
    sp.add_argument("--items", required=True)
    sp.add_argument("--servers", required=True)
    sp.add_argument("--readout", choices=["pair", "choice"], default="pair")
    sp.add_argument("--tokenizer-backend", choices=["server", "hf"], default="server")
    sp.add_argument("--hf-repo", default="Qwen/Qwen3.5-27B")
    sp.add_argument("--concurrency", type=int, default=8)
    sp.add_argument("--n-probs", type=int, default=200)
    sp.add_argument("--max-tokens", type=int, default=0)
    sp.add_argument("--cost-ratio", type=float, default=0.10)
    sp.add_argument("--epsilon", type=float, default=0.05)
    sp.add_argument("--out", default="")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("calibrate", help="recalibrate conformal table from results")
    sp.add_argument("--results", required=True, help="rows JSONL from evaluate")
    sp.add_argument("--write", default="", help="write JSON table to this path")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("flip", help="option-order flip rate (choice readout)")
    sp.add_argument("--items", required=True)
    sp.add_argument("--server", required=True)
    sp.add_argument("--n", type=int, default=4)
    sp.add_argument("--limit", type=int, default=200)
    sp.add_argument("--tokenizer-backend", choices=["server", "hf"], default="server")
    sp.add_argument("--hf-repo", default="Qwen/Qwen3.5-27B")
    sp.set_defaults(func=cmd_flip)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
