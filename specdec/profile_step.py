"""cProfile one decode step (T=1) and one verification step (T=5) of gpt2-medium.

Usage: python -m specdec.profile_step
"""
import cProfile
import pstats
import time

from . import common


def main():
    m = common.load("gpt2-medium")
    ids = common.prompts(1)[0]
    c = m.new_cache()
    m.forward(ids, c, last_only=True)
    m.forward(ids[:1], c, last_only=True)  # warm-up
    for T in [1, 5]:
        c.crop(len(ids))
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        m.forward(ids[:T], c, last_only=(T == 1))
        pr.disable()
        print(f"\n===== T={T}: {time.perf_counter() - t0:.3f}s =====")
        pstats.Stats(pr).sort_stats("tottime").print_stats(18)
    common.assert_no_torch()


if __name__ == "__main__":
    main()
