"""Correctness gates, run before any benchmark.

1. MiniTorch GPT-2 logits vs an independent float64 numpy reference (specdec/reference.py).
2. Incremental (chunked, KV-cached) forward vs one full forward.
3. Cache crop + re-feed reproduces the uncropped result.
4. Greedy speculative decoding output == target-only greedy output, for several k.

Usage: python -m specdec.test_correctness
"""
import sys

import numpy as np

from . import common
from .decoding import autoregressive, speculative
from .reference import gpt2_logits


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="gpt2", choices=["gpt2", "distilgpt2"])
    args = ap.parse_args()
    tok = common.tokenizer()
    prompts = [p[:48] for p in common.prompts()[-3:]]  # held out from the default benchmark prompts
    ok = True
    models = {}
    for name in [args.draft, "gpt2-medium"]:
        m = common.load(name)
        models[name] = m
        ids = prompts[0]
        got = m.forward(ids, m.new_cache())
        # MiniTorch's fused LayerNorm kernel hard-codes eps=1e-8 (GPT-2 uses 1e-5); that alone moves
        # gpt2-medium logits by ~5e-2, so the tight check uses a reference with the same eps.
        ref = gpt2_logits(common.model_dir(name), ids, eps=1e-8)
        err = np.abs(got - ref).max()
        ok &= check(f"{name}: logits vs numpy fp64 ref (eps=1e-8)", err < 1e-2, f"max|diff|={err:.2e}")
        ref = gpt2_logits(common.model_dir(name), ids)
        err = np.abs(got - ref).max()
        agree = (got.argmax(-1) == ref.argmax(-1)).mean()
        ok &= check(f"{name}: argmax vs numpy fp64 ref (checkpoint eps=1e-5)", agree == 1.0,
                    f"max|diff|={err:.2e} argmax_agree={agree:.3f} (|logit| max {np.abs(ref).max():.1f})")

        c = m.new_cache()
        parts = [m.forward(ids[:20], c), m.forward(ids[20:21], c), m.forward(ids[21:30], c), m.forward(ids[30:], c)]
        inc = np.concatenate(parts, 0)
        err = np.abs(inc - got).max()
        ok &= check(f"{name}: incremental vs full", err < 1e-3, f"max|diff|={err:.2e}")

        c = m.new_cache()
        m.forward(ids[:30], c)
        m.forward(tok.encode(" garbage tokens that must be rolled back").ids, c)
        c.crop(30)
        err = np.abs(m.forward(ids[30:], c) - got[30:]).max()
        ok &= check(f"{name}: crop + refeed", err < 1e-3, f"max|diff|={err:.2e}")

    target, draft = models["gpt2-medium"], models[args.draft]
    for ids in prompts:
        ref, _ = autoregressive(target, ids, 24)
        for k in [1, 3, 5]:
            out, st = speculative(target, draft, ids, 24, k)
            same = out == ref
            first_diff = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), None)
            ok &= check(f"greedy spec == target greedy (k={k})", same,
                        f"accepted {st.accepted}/{st.drafted}" + ("" if same else f" first diff at {first_diff}"))

    rng = np.random.default_rng(0)
    out, st = speculative(target, draft, prompts[0], 24, 4, temperature=1.0, rng=rng)
    ok &= check("sampling spec runs", len(out) == 24, f"accepted {st.accepted}/{st.drafted}")

    ref, _ = autoregressive(target, prompts[0], 32)
    print("prompt  :", tok.decode(prompts[0]))
    print("medium  :", tok.decode(ref))
    common.assert_no_torch()
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
