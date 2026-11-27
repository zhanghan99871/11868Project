"""Speculative decoding benchmark in MiniTorch.

Targets: GPT-2 medium (default) or the MLA verifier (--target mla-gpt2-medium, exported with
torch_train/export_mla_verifier.py to common.MLA_VERIFIER_DIR). Drafters: a classic autoregressive
draft LM (--draft gpt2 | distilgpt2, its own KV cache), or, with the MLA verifier, the H-Spec
block drafter (common.HSPEC_DRAFTER_DIR or --hspec-drafter; no cache of its own). A fine-tuned
classic drafter is loaded from its save_pretrained directory with --draft-dir.

Every config decodes the same WikiText-2 prompts. Results are appended as JSON
lines so a run can be split across several short GPU allocations.

Usage:
  python -m specdec.bench --configs ar_target ar_draft spec_k4 --out results/x.jsonl
  python -m specdec.bench --target mla-gpt2-medium --configs micro --n-prompts 1 --out results/mla_micro.jsonl
  python -m specdec.bench --target mla-gpt2-medium --draft distilgpt2 \
      --configs ar_target spec_k2 hspec hspec_k3 --out results/hspec.jsonl
  python -m specdec.bench --target mla-gpt2-medium --draft-dir runs/distil-drafter-gen-s32/best \
      --draft-label distil-gen --configs spec_k3 --out results/eval.jsonl --skip-done
Rows record the drafter in "draft" (--draft-label / --hspec-label), so several drafters can share
one file; --skip-done skips configs whose rows for every prompt are already in --out (eval.sh).
config names:
  ar_target | ar_draft             target / classic draft LM alone
  spec_k{K}[_t{TEMP}]              classic speculative decoding, K drafts per step
  hspec[_k{K}][_t{TEMP}]           H-Spec block drafting, K <= block size (default: block size)
  micro                            forward latency vs new tokens (+ one H-Spec drafter pass)
"""
import argparse
import json
import os
import re
import socket
import time
from collections import defaultdict

import numpy as np

from . import common
from .decoding import autoregressive, speculative, speculative_block

SPEC = re.compile(r"spec_k(\d+)(?:_t([\d.]+))?")
HSPEC = re.compile(r"hspec(?:_k(\d+))?(?:_t([\d.]+))?")
HSPEC_NAME = "hspec"  # default "draft" label of H-Spec rows (--hspec-label)


def is_config(cfg):
    return cfg in ("ar_target", "ar_draft", "micro") or bool(SPEC.fullmatch(cfg) or HSPEC.fullmatch(cfg))


def row_key(config, target, draft):
    """What identifies a config's rows: ar_target does not depend on the drafter."""
    return (config, target) if config == "ar_target" else (config, target, draft)


def done_configs(path, n_prompts):
    """row_keys of the configs in `path` (a bench JSONL) that have rows for prompts 0..n_prompts-1."""
    seen = defaultdict(set)
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["config"] != "micro":
                    seen[row_key(r["config"], r.get("target", "gpt2-medium"), r.get("draft"))].add(r["prompt"])
    return {key for key, got in seen.items() if set(range(n_prompts)) <= got}


def draft_label(draft, draft_dir=None):
    """Default "draft" label: the --draft name, or the run directory of --draft-dir
    (runs/distil-drafter-gen/best -> distil-drafter-gen)."""
    if not draft_dir:
        return draft
    path = os.path.normpath(draft_dir)
    name = os.path.basename(path)
    if name in ("best", "minitorch"):
        return os.path.basename(os.path.dirname(path)) or name
    return name


def run_config(cfg, target, draft, prompts, n_new, seed, draft_name, target_name="gpt2-medium", hspec=None,
               hspec_name=HSPEC_NAME):
    spec, block = SPEC.fullmatch(cfg), HSPEC.fullmatch(cfg)
    if cfg not in ("ar_target", "ar_draft") and not spec and not block:
        raise ValueError(f"unknown config {cfg!r}")
    if block and hspec is None:
        raise ValueError(f"{cfg} needs the H-Spec drafter (use --target {common.MLA_VERIFIER})")
    if (cfg == "ar_draft" or spec) and draft is None:
        raise ValueError(f"{cfg} needs a classic draft model")
    m = spec or block
    temp = float(m.group(2)) if m and m.group(2) else 0.0
    rows = []
    for pi, ids in enumerate(prompts):
        rng = np.random.default_rng(seed + pi)
        if cfg == "ar_target":
            out, st = autoregressive(target, ids, n_new)
        elif cfg == "ar_draft":
            out, st = autoregressive(draft, ids, n_new)
        elif block:
            k = int(block.group(1)) if block.group(1) else None
            out, st = speculative_block(target, hspec, ids, n_new, k=k, temperature=temp, rng=rng)
        else:
            out, st = speculative(target, draft, ids, n_new, int(spec.group(1)), temperature=temp, rng=rng)
        rows.append({
            "config": cfg, "target": target_name, "draft": hspec_name if block else draft_name,
            "prompt": pi, "temperature": temp, "n_new": n_new, "tokens": out,
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


def micro(target, draft, ids, out, draft_name, target_name="gpt2-medium", hspec=None):
    """Per-call latency of one forward vs number of new tokens (cache already holds the prompt),
    and of one H-Spec drafter pass (block_size tokens) after the prompt."""
    rows = []
    models = [(target_name, target)] + ([(draft_name, draft)] if draft is not None else [])
    for name, m in models:
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
    if hspec is not None:
        times = []
        for rep in range(6):
            c = target.new_cache()
            _, hidden = target.forward(ids[:-1], c, last_only=True, hidden_layers=hspec.cfg.tap_layers)
            t0 = time.perf_counter()
            hspec.draft_from_cache(ids[-1], c, hidden[:, -1])
            times.append(time.perf_counter() - t0)
        rows.append({"config": "micro", "resident": hspec.resident, "model": "hspec-drafter",
                     "T": hspec.cfg.block_size, "median_s": float(np.median(times[1:]))})
        print(f"micro hspec-drafter (one pass, {hspec.cfg.block_size} tokens): {rows[-1]['median_s'] * 1e3:.1f} ms",
              flush=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft", default="gpt2", choices=["gpt2", "distilgpt2"])
    ap.add_argument("--draft-dir", default=None,
                    help="classic drafter from a save_pretrained directory (torch_train/finetune_distilgpt2.py "
                         "best/) instead of --draft")
    ap.add_argument("--draft-label", default=None,
                    help="'draft' of classic-drafter rows (default: --draft, or the --draft-dir run name)")
    ap.add_argument("--target", default="gpt2-medium", choices=list(common.TARGETS))
    ap.add_argument("--hspec-drafter", default=common.HSPEC_DRAFTER_DIR,
                    help="exported H-Spec drafter (torch_train/export_hspec_drafter.py)")
    ap.add_argument("--hspec-label", default=HSPEC_NAME, help="'draft' of H-Spec rows")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip configs whose rows (same target and draft label) for all prompts are in --out")
    ap.add_argument("--no-resident", action="store_true", help="copy weights host->device on every matmul")
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for cfg in args.configs:  # fail before loading models
        if not is_config(cfg):
            ap.error(f"unknown config {cfg!r}")
    label = args.draft_label or draft_label(args.draft, args.draft_dir)
    if args.skip_done:
        done = done_configs(args.out, args.n_prompts)
        todo = [c for c in args.configs
                if c == "micro" or row_key(c, args.target, args.hspec_label if HSPEC.fullmatch(c) else label) not in done]
        for c in args.configs:
            if c not in todo:
                print(f"{c} [{args.hspec_label if HSPEC.fullmatch(c) else label}]: already in {args.out}, skipped",
                      flush=True)
        if not todo:
            return
        args.configs = todo

    prompts = [p[:args.prompt_len] for p in common.prompts(args.n_prompts)]
    resident = not args.no_resident
    wants_hspec = any(HSPEC.fullmatch(c) for c in args.configs)
    if wants_hspec and args.target != common.MLA_VERIFIER:
        ap.error(f"hspec configs need --target {common.MLA_VERIFIER}")
    wants_draft = any(c in ("ar_draft", "micro") or SPEC.fullmatch(c) for c in args.configs)

    target = common.load(args.target, resident=resident)
    draft = None
    if wants_draft:
        draft = (common.load_gpt2_dir(args.draft_dir, resident=resident) if args.draft_dir
                 else common.load(args.draft, resident=resident))
    hspec = None
    if args.target == common.MLA_VERIFIER and (wants_hspec or "micro" in args.configs):
        if wants_hspec or os.path.isfile(os.path.join(args.hspec_drafter, "config.json")):
            hspec = common.load_hspec_drafter(target, args.hspec_drafter, resident=resident)
        else:
            print(f"no H-Spec drafter at {args.hspec_drafter}; micro skips it", flush=True)
    autoregressive(target, prompts[0], 4)  # warm up CUDA context / allocator
    if draft is not None:
        autoregressive(draft, prompts[0], 4)
    if hspec is not None:
        speculative_block(target, hspec, prompts[0], 4)
    print(f"host {socket.gethostname()}, {len(prompts)} prompts, resident={resident}, target={args.target}, "
          f"draft={label + ' (' + (args.draft_dir or args.draft) + ')' if draft is not None else '-'}, "
          f"hspec={args.hspec_label + ' (' + args.hspec_drafter + ')' if hspec else '-'}", flush=True)

    for cfg in args.configs:
        if cfg == "micro":
            micro(target, draft, prompts[0], args.out, label, args.target, hspec)
            continue
        rows = run_config(cfg, target, draft, prompts, args.n_new, args.seed, label, args.target, hspec,
                          hspec_name=args.hspec_label)
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    common.assert_no_torch()


if __name__ == "__main__":
    main()
