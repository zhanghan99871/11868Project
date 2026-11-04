"""Speculative decoding benchmark: draft (GPT-2 small or distilgpt2) -> GPT-2 medium (target), MiniTorch.

Every config decodes the same WikiText-2 prompts. Results are appended as JSON
lines so a run can be split across several short GPU allocations.

Usage:
  python -m specdec.bench --configs ar_target ar_draft spec_k4 --out results/x.jsonl
  config names: ar_target | ar_draft | spec_k{K} | spec_k{K}_t{TEMP}
"""
import argparse
import json
import os
import re
import socket
import time

import numpy as np

from . import common
from .decoding import autoregressive, speculative


def run_config(cfg, target, draft, prompts, n_new, seed, draft_name):
    m = re.fullmatch(r"spec_k(\d+)(?:_t([\d.]+))?", cfg)
    temp = float(m.group(2)) if m and m.group(2) else 0.0
    rows = []
    for pi, ids in enumerate(prompts):
        rng = np.random.default_rng(seed + pi)
        if cfg == "ar_target":
            out, st = autoregressive(target, ids, n_new)
        elif cfg == "ar_draft":
            out, st = autoregressive(draft, ids, n_new)
        else:
            out, st = speculative(target, draft, ids, n_new, int(m.group(1)), temperature=temp, rng=rng)
        rows.append({
            "config": cfg, "draft": draft_name, "prompt": pi, "temperature": temp, "n_new": n_new, "tokens": out,
            "total_time": st.total_time, "prefill_time": st.prefill_time,
            "decode_time": st.total_time - st.prefill_time,
            "target_calls": st.target_calls, "draft_calls": st.draft_calls,
            "target_time": st.target_time, "draft_time": st.draft_time,
            "iterations": st.iterations, "drafted": st.drafted, "accepted": st.accepted,
            "accepted_per_iter": st.accepted_per_iter,
        })
        print(f"{cfg} prompt {pi}: decode {rows[-1]['decode_time']:.2f}s "
              f"acc {st.accepted}/{st.drafted}", flush=True)
    return rows


def micro(target, draft, ids, out, draft_name):
    """Per-call latency of one forward vs number of new tokens (cache already holds the prompt)."""
    rows = []
    for name, m in [("gpt2-medium", target), (draft_name, draft)]:
        for T in [1, 2, 3, 5, 7, 9]:
            times = []
            for rep in range(6):
                c = m.new_cache()
                m.forward(ids, c, last_only=True)
                t0 = time.perf_counter()
                m.forward(ids[:T], c, last_only=(T == 1))
                times.append(time.perf_counter() - t0)
            rows.append({"config": "micro", "resident": m.resident, "model": name, "T": T, "median_s": float(np.median(times[1:]))})
            print(f"micro {name} T={T}: {rows[-1]['median_s'] * 1e3:.1f} ms", flush=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft", default="gpt2", choices=["gpt2", "distilgpt2"])
    ap.add_argument("--no-resident", action="store_true", help="copy weights host->device on every matmul")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    prompts = [p[:args.prompt_len] for p in common.prompts(args.n_prompts)]
    resident = not args.no_resident
    target, draft = common.load("gpt2-medium", resident=resident), common.load(args.draft, resident=resident)
    autoregressive(target, prompts[0], 4)  # warm up CUDA context / allocator
    autoregressive(draft, prompts[0], 4)
    print(f"host {socket.gethostname()}, {len(prompts)} prompts, resident={resident}, draft={args.draft}", flush=True)

    for cfg in args.configs:
        if cfg == "micro":
            micro(target, draft, prompts[0], args.out, args.draft)
            continue
        rows = run_config(cfg, target, draft, prompts, args.n_new, args.seed, args.draft)
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    common.assert_no_torch()


if __name__ == "__main__":
    main()
