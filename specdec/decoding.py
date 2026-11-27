"""Autoregressive and speculative decoding (Leviathan et al., 2023) for batch size 1.

Both models expose `forward(ids, cache, last_only)` and `new_cache()`.
`speculative` drafts autoregressively with a second LM (its own cache); `speculative_block`
drafts a whole block per pass with an H-Spec-style drafter that reads the verifier's cache.

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


def _accept(lg, drafts, qs, temperature, rng):
    """Verify drafts against the target's logits lg [len(drafts) + 1, V] (row i follows drafts[:i]).

    Returns (n, nxt): the number of accepted drafts and the token that follows them (the target's
    correction at the first rejection, or the bonus token if all are accepted). Greedy: accept
    while the draft is the target's argmax. Sampling (Leviathan et al. 2023): accept drafts[i]
    with probability min(1, p_i(d) / q_i(d)), where qs[i] is the distribution it was sampled from;
    on rejection resample from max(p_i - q_i, 0). The output is distributed as target sampling.
    """
    kk = len(drafts)
    n, nxt = 0, None
    if temperature == 0:
        while n < kk and int(np.argmax(lg[n])) == drafts[n]:
            n += 1
        return n, int(np.argmax(lg[n]))
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
    return n, nxt


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
        n, nxt = _accept(lg, drafts, qs, temperature, rng)

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


def speculative_block(target, drafter, prompt, n_new, k=None, temperature=0.0, rng=None):
    """Speculative decoding with a block-parallel drafter that keeps no cache (H-Spec style).

    `target` is the MLA verifier (mla_model.MiniTorchMLAGPT2: forward(..., hidden_layers=...));
    `drafter` is project/minitorch_drafter.MiniTorchHSpecDrafter (or anything with
    `cfg.block_size`, `cfg.tap_layers` and `draft_from_cache(anchor, cache, summary)`).

    Invariant at the top of each iteration: the verifier's cache holds tokens[:-1]; tokens[-1] is
    the anchor (not yet cached), and `summary` is the verifier's hidden state at the last cached
    position after each tap layer. One drafter pass drafts up to k tokens at once from that cache;
    one verifier pass scores [anchor] + drafts, appending their latents and returning their hidden
    states; the cache is cropped back to the anchor and the n accepted drafts, and the hidden
    state at the last accepted position becomes the next summary.

    k defaults to the drafter's block size (and may not exceed it). Needs >= 2 prompt tokens: the
    first summary is the verifier's hidden state at the token before the anchor.
    """
    taps = list(drafter.cfg.tap_layers)
    k = drafter.cfg.block_size if k is None else k
    if not 1 <= k <= drafter.cfg.block_size:
        raise ValueError(f"k must be in [1, {drafter.cfg.block_size}], got {k}")
    if len(prompt) < 2:
        raise ValueError("speculative_block needs at least 2 prompt tokens")
    stats = Stats()
    t_start = time.perf_counter()
    cache = target.new_cache()
    tokens = list(prompt)
    _, hidden = target.forward(tokens[:-1], cache, last_only=True, hidden_layers=taps)
    summary = hidden[:, -1]
    stats.prefill_time = time.perf_counter() - t_start

    produced = 0
    while produced < n_new:
        kk = min(k, n_new - produced - 1)  # never draft past the requested length
        drafts, qs = [], []
        if kk > 0:
            t0 = time.perf_counter()
            for row in drafter.draft_from_cache(tokens[-1], cache, summary)[:kk]:
                tok, q = _pick(row, temperature, rng)
                drafts.append(tok)
                qs.append(q)
            stats.draft_calls += 1
            stats.draft_time += time.perf_counter() - t0

        P = cache.length
        t0 = time.perf_counter()
        lg, hidden = target.forward([tokens[-1]] + drafts, cache, last_only=False, hidden_layers=taps)
        stats.target_calls += 1
        stats.target_time += time.perf_counter() - t0
        n, nxt = _accept(lg, drafts, qs, temperature, rng)

        cache.crop(P + 1 + n)  # the anchor and the n accepted drafts
        summary = hidden[:, n]  # the last accepted position, whose output is nxt
        tokens += drafts[:n] + [nxt]
        produced += n + 1
        stats.iterations += 1
        stats.drafted += kk
        stats.accepted += n
        stats.accepted_per_iter.append(n)

    stats.total_time = time.perf_counter() - t_start
    return tokens[len(prompt):len(prompt) + n_new], stats
