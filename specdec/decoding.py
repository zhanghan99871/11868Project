"""Autoregressive and speculative decoding (Leviathan et al., 2023) for batch size 1.

Both models expose `forward(ids, cache, last_only)` and `new_cache()`.

Cache invariant: a cache holds K/V for a prefix of `tokens`; whatever suffix is
missing is fed at the next call. After verification both caches are cropped to
the accepted prefix, so rejected draft tokens never leak into later steps.

temperature == 0 -> greedy; output must equal target-only greedy decoding.
temperature  > 0 -> rejection sampling; output is distributed as target sampling.
"""
import time
from dataclasses import dataclass, field

import numpy as np


def _probs(logits, temperature):
    z = logits.astype(np.float64) / temperature
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def _pick(logits, temperature, rng):
    """Return (token, prob vector or None)."""
    if temperature == 0:
        return int(np.argmax(logits)), None
    p = _probs(logits, temperature)
    return int(rng.choice(len(p), p=p)), p


@dataclass
class Stats:
    target_calls: int = 0
    draft_calls: int = 0
    target_time: float = 0.0
    draft_time: float = 0.0
    prefill_time: float = 0.0
    total_time: float = 0.0
    iterations: int = 0
    drafted: int = 0
    accepted: int = 0
    accepted_per_iter: list = field(default_factory=list)


def _timed(model, ids, cache, last_only, stats, which):
    t0 = time.perf_counter()
    out = model.forward(ids, cache, last_only=last_only)
    dt = time.perf_counter() - t0
    if which == "target":
        stats.target_calls += 1
        stats.target_time += dt
    else:
        stats.draft_calls += 1
        stats.draft_time += dt
    return out


def autoregressive(model, prompt, n_new, temperature=0.0, rng=None):
    stats = Stats()
    t_start = time.perf_counter()
    cache = model.new_cache()
    logits = _timed(model, prompt, cache, True, stats, "target")
    stats.prefill_time = time.perf_counter() - t_start
    out = []
    for i in range(n_new):
        tok, _ = _pick(logits[-1], temperature, rng)
        out.append(tok)
        if i + 1 < n_new:
            logits = _timed(model, [tok], cache, True, stats, "target")
    stats.total_time = time.perf_counter() - t_start
    return out, stats


def speculative(target, draft, prompt, n_new, k, temperature=0.0, rng=None):
    stats = Stats()
    t_start = time.perf_counter()
    tc, dc = target.new_cache(), draft.new_cache()
    tokens = list(prompt)
    # prefill everything except the last prompt token, which is fed with the first drafts
    if len(prompt) > 1:
        _timed(target, tokens[:-1], tc, True, stats, "target")
        _timed(draft, tokens[:-1], dc, True, stats, "draft")
    stats.prefill_time = time.perf_counter() - t_start
    stats.target_calls = stats.draft_calls = 0
    stats.target_time = stats.draft_time = 0.0

    produced = 0
    while produced < n_new:
        kk = min(k, n_new - produced - 1)  # never draft past the requested length
        drafts, qs = [], []
        feed = tokens[dc.length:]
        for _ in range(kk):
            lg = _timed(draft, feed, dc, True, stats, "draft")[-1]
            tok, q = _pick(lg, temperature, rng)
            drafts.append(tok)
            qs.append(q)
            feed = [tok]

        # one target forward scores every draft position plus the bonus position
        lg = _timed(target, tokens[tc.length:] + drafts, tc, False, stats, "target")[-(kk + 1):]

        n = 0
        nxt = None
        if temperature == 0:
            while n < kk and int(np.argmax(lg[n])) == drafts[n]:
                n += 1
            nxt = int(np.argmax(lg[n]))
        else:
            while n < kk:
                p = _probs(lg[n], temperature)
                d = drafts[n]
                if rng.random() < min(1.0, p[d] / qs[n][d]):
                    n += 1
                    continue
                resid = np.maximum(p - qs[n], 0.0)
                nxt = int(rng.choice(len(resid), p=resid / resid.sum()))
                break
            if nxt is None:
                nxt = int(rng.choice(lg.shape[1], p=_probs(lg[kk], temperature)))

        tc.crop(len(tokens) + n)
        dc.crop(min(dc.length, len(tokens) + n))
        tokens += drafts[:n] + [nxt]
        produced += n + 1
        stats.iterations += 1
        stats.drafted += kk
        stats.accepted += n
        stats.accepted_per_iter.append(n)

    stats.total_time = time.perf_counter() - t_start
    return tokens[len(prompt):len(prompt) + n_new], stats
