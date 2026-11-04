"""Why is a T=2 verification call slower inside speculative decoding than in isolation?

Times target.forward for T=1 and T=2 at several cache lengths, then cProfiles one
speculative run (k=1) to see where the extra time goes.
Usage: python -m specdec.profile_spec
"""
import cProfile
import pstats
import time

import numpy as np

from . import common
from .decoding import speculative


def main():
    target, draft = common.load("gpt2-medium"), common.load("gpt2")
    ids = common.prompts(1)[0]
    speculative(target, draft, ids, 8, 1)  # warm-up
    for P in [64, 96, 127]:
        for T, last_only in [(1, True), (2, False), (2, True)]:
            ts = []
            for _ in range(4):
                c = target.new_cache()
                target.forward((ids * 3)[:P], c, last_only=True)
                t0 = time.perf_counter()
                target.forward(ids[:T], c, last_only=last_only)
                ts.append(time.perf_counter() - t0)
            print(f"target P={P} T={T} last_only={last_only}: {np.median(ts[1:]) * 1e3:.1f} ms", flush=True)

    pr = cProfile.Profile()
    pr.enable()
    out, st = speculative(target, draft, ids, 32, 1)
    pr.disable()
    print(f"spec k=1: target {st.target_time / st.target_calls * 1e3:.1f} ms/call, "
          f"draft {st.draft_time / st.draft_calls * 1e3:.1f} ms/call", flush=True)
    pstats.Stats(pr).sort_stats("tottime").print_stats(15)
    common.assert_no_torch()


if __name__ == "__main__":
    main()
