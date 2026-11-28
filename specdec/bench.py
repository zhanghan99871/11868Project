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
one file, and the bench run in "run_id" (host, start time, random suffix): specdec.summarize divides by
the ar_target of the same run, so give each run its own ar_target. --skip-done skips configs whose
rows for every prompt are already in --out (eval.sh); ar_target is re-timed whenever other configs
still run, so they get a same-run baseline.
config names:
  ar_target | ar_draft             target / classic draft LM alone
  spec_k{K}[_t{TEMP}]              classic speculative decoding, K drafts per step
  hspec[_k{K}][_t{TEMP}]           H-Spec block drafting, K <= block size (default: block size)
  micro                            forward latency vs new tokens (+ one H-Spec drafter pass)
  memory                           memory each loaded model holds: GPU bytes of its resident weights
                                   (measured around loading), host bytes, parameters and its cache
                                   (specdec.memory has the exact accounting from configs)
"""
import argparse
import gc
import json
import os
import re
import socket
import time
import uuid
from collections import defaultdict

import numpy as np

from . import common
from .decoding import autoregressive, speculative, speculative_block

SPEC = re.compile(r"spec_k(\d+)(?:_t([\d.]+))?")
HSPEC = re.compile(r"hspec(?:_k(\d+))?(?:_t([\d.]+))?")
HSPEC_NAME = "hspec"  # default "draft" label of H-Spec rows (--hspec-label)


AUX = ("micro", "memory")  # configs that do not decode


def is_config(cfg):
    return cfg in ("ar_target", "ar_draft") + AUX or bool(SPEC.fullmatch(cfg) or HSPEC.fullmatch(cfg))


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
                key = row_key(r["config"], r.get("target", "gpt2-medium"), r.get("draft"))
                if r["config"] == "memory":
                    seen[key] = set(range(n_prompts))  # measured once per drafter
                elif r["config"] != "micro":
                    seen[key].add(r["prompt"])
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


def loads_classic_draft(configs, draft_dir=None):
    """Whether a run of `configs` loads the classic draft model (memory alone measures the
    --draft-dir drafter, or else the H-Spec drafter)."""
    wants_hspec = any(HSPEC.fullmatch(c) for c in configs)
    return (any(c in ("ar_draft", "micro") or SPEC.fullmatch(c) for c in configs)
            or ("memory" in configs and not wants_hspec and draft_dir is not None))


def todo_configs(configs, done, target, label, hspec_label, mem_label=None):
    """The configs of `configs` that are not in `done` (done_configs). ar_target counts as done only
    when every other decoding config is, so new rows always come with a same-run baseline. Memory
    rows carry `mem_label` (default: the classic label if the run loads a classic drafter, else the
    H-Spec label)."""
    if mem_label is None:
        mem_label = label if loads_classic_draft(configs) else hspec_label

    def key(c):
        return row_key(c, target, hspec_label if HSPEC.fullmatch(c) else mem_label if c == "memory" else label)

    rest = [c for c in configs if c != "ar_target" and (c == "micro" or key(c) not in done)]
    decoding = [c for c in rest if c not in AUX]
    return [c for c in configs if c in rest or (c == "ar_target" and (decoding or key(c) not in done))]


def memory_in_use():
    """(GPU bytes in use on the device or None without CUDA, this process's resident host bytes)."""
    gc.collect()
    device = None
    try:
        import pycuda.driver as cuda
        free, total = cuda.mem_get_info()
        device = total - free
    except Exception:  # no pycuda, no context, no GPU
        pass
    try:
        with open("/proc/self/statm") as f:
            host = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        import resource
        host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return device, host


def cache_bytes(cache):
    """(bytes allocated, bytes per cached token) of a verifier or draft-model cache."""
    allocated = sum(a.nbytes for a in vars(cache).values() if isinstance(a, np.ndarray))
    return allocated, allocated // cache.max_len


def model_memory(model):
    """specdec.memory accounting of a loaded MiniTorch model."""
    from . import memory
    if hasattr(model, "draft_logits"):
        return memory.hspec_drafter(model.cfg)
    if hasattr(model, "cfg"):
        return memory.mla_verifier(model.cfg)
    return memory.gpt2_drafter({"n_layer": model.n_layer, "n_embd": model.n_embd, "vocab_size": model.vocab_size,
                                "n_positions": model.wpe.shape[0]})


def memory_rows(loaded, measured, label, target_name, extra=None):
    """One row per loaded model: device/host bytes measured around its loading (None if not
    measured), parameters and its cache. `loaded`: [(name, role, model)]."""
    rows = []
    for name, role, model in loaded:
        acct = model_memory(model)
        device, host = measured.get(name, (None, None))
        allocated, per_token = cache_bytes(model.new_cache()) if hasattr(model, "new_cache") else (0, 0)
        rows.append({"config": "memory", "target": target_name, "draft": label, "model": name, "role": role,
                     "kind": acct["kind"], "params": acct["params"], "shared_params": acct["shared_params"],
                     "shares_head": bool(getattr(model, "shares_head", False)), "resident": model.resident,
                     "device_bytes": device, "host_bytes": host, "cache_bytes_allocated": allocated,
                     "cache_bytes_per_token": per_token, **(extra or {})})
        print(f"memory {name} ({role}): {acct['params'] / 1e6:.1f}M params, device "
              f"{'-' if device is None else f'{device / 2**20:.0f} MB'}, host {'-' if host is None else f'{host / 2**20:.0f} MB'}, "
              f"cache {per_token / 1024:.1f} KB/token", flush=True)
    return rows


def run_config(cfg, target, draft, prompts, n_new, seed, draft_name, target_name="gpt2-medium", hspec=None,
               hspec_name=HSPEC_NAME, extra=None):
    """Rows of one config on every prompt; `extra` (e.g. host, run_id) is added to each row."""
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
            "accepted_per_iter": st.accepted_per_iter, **(extra or {}),
        })
        print(f"{cfg} prompt {pi}: decode {rows[-1]['decode_time']:.2f}s "
              f"acc {st.accepted}/{st.drafted}", flush=True)
    return rows


def micro(target, draft, ids, out, draft_name, target_name="gpt2-medium", hspec=None, extra=None):
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
            f.write(json.dumps({**r, **(extra or {})}) + "\n")


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
    wants_draft = loads_classic_draft(args.configs, args.draft_dir)  # decided on all requested configs
    mem_label = label if wants_draft else args.hspec_label
    if args.skip_done:
        todo = todo_configs(args.configs, done_configs(args.out, args.n_prompts), args.target, label, args.hspec_label,
                            mem_label)
        for c in args.configs:
            if c not in todo:
                shown = args.hspec_label if HSPEC.fullmatch(c) else mem_label if c == "memory" else label
                print(f"{c} [{shown}]: already in {args.out}, skipped", flush=True)
        if not todo:
            return
        args.configs = todo

    host = socket.gethostname()
    extra = {"host": host, "run_id": f"{host}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"}
    prompts = [p[:args.prompt_len] for p in common.prompts(args.n_prompts)]
    resident = not args.no_resident
    wants_hspec = any(HSPEC.fullmatch(c) for c in args.configs)
    if wants_hspec and args.target != common.MLA_VERIFIER:
        ap.error(f"hspec configs need --target {common.MLA_VERIFIER}")
    wants_memory = "memory" in args.configs
    measured = {}

    def timed_load(name, load):
        before = memory_in_use() if wants_memory else None
        model = load()
        if wants_memory:
            after = memory_in_use()
            measured[name] = tuple(None if a is None or b is None else a - b for a, b in zip(after, before))
        return model

    if wants_memory:
        common.backend()  # create the CUDA context first, so it is not counted as the verifier's
    target = timed_load(args.target, lambda: common.load(args.target, resident=resident))
    draft = None
    if wants_draft:
        draft = timed_load(label, lambda: common.load_gpt2_dir(args.draft_dir, resident=resident) if args.draft_dir
                           else common.load(args.draft, resident=resident))
    hspec = None
    if args.target == common.MLA_VERIFIER and (wants_hspec or "micro" in args.configs
                                               or (wants_memory and draft is None)):
        if wants_hspec or os.path.isfile(os.path.join(args.hspec_drafter, "config.json")):
            hspec = timed_load(args.hspec_label, lambda: common.load_hspec_drafter(target, args.hspec_drafter,
                                                                                    resident=resident))
        else:
            print(f"no H-Spec drafter at {args.hspec_drafter}; micro/memory skip it", flush=True)
    autoregressive(target, prompts[0], 4)  # warm up CUDA context / allocator
    if draft is not None:
        autoregressive(draft, prompts[0], 4)
    if hspec is not None:
        speculative_block(target, hspec, prompts[0], 4)
    print(f"host {host}, run {extra['run_id']}, {len(prompts)} prompts, resident={resident}, target={args.target}, "
          f"draft={label + ' (' + (args.draft_dir or args.draft) + ')' if draft is not None else '-'}, "
          f"hspec={args.hspec_label + ' (' + args.hspec_drafter + ')' if hspec else '-'}", flush=True)

    for cfg in args.configs:
        if cfg == "micro":
            micro(target, draft, prompts[0], args.out, label, args.target, hspec, extra=extra)
            continue
        if cfg == "memory":
            loaded = [(args.target, "verifier", target)]
            loaded += [(label, "drafter", draft)] if draft is not None else []
            loaded += [(args.hspec_label, "drafter", hspec)] if hspec is not None else []
            with open(args.out, "a") as f:
                for r in memory_rows(loaded, measured, mem_label, args.target, extra):
                    f.write(json.dumps(r) + "\n")
            continue
        rows = run_config(cfg, target, draft, prompts, args.n_new, args.seed, label, args.target, hspec,
                          hspec_name=args.hspec_label, extra=extra)
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    common.assert_no_torch()


if __name__ == "__main__":
    main()
