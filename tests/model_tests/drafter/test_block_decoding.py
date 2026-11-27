"""specdec.decoding.speculative_block: H-Spec block drafting with the MLA verifier (CPU numpy
backends; the CUDA test is skipped without a GPU), and its benchmark configs."""
import dataclasses
import io
import json
from contextlib import redirect_stdout

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from .helpers import make_drafter, make_verifier  # noqa: E402  (sets sys.path)

from export_hspec_drafter import export_drafter  # noqa: E402
from export_mla_verifier import export_state as export_verifier  # noqa: E402
from minitorch_drafter import MiniTorchHSpecDrafter  # noqa: E402
from specdec import bench, summarize  # noqa: E402
from specdec.decoding import autoregressive, speculative_block  # noqa: E402
from specdec.mla_model import MiniTorchMLAGPT2  # noqa: E402

PROMPT = [int(t) for t in np.random.default_rng(3).integers(0, 97, 9)]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """(numpy MLA verifier, numpy H-Spec drafter with block size 4) from tiny PyTorch models."""
    tmp = tmp_path_factory.mktemp("block")
    verifier = make_verifier()
    target = MiniTorchMLAGPT2(str(export_verifier(verifier.state_dict(), dataclasses.asdict(verifier.cfg), tmp / "v")))
    drafter = MiniTorchHSpecDrafter.from_verifier(str(export_drafter(make_drafter(verifier), tmp / "d")), target)
    return target, drafter


class OracleDrafter:
    """Drafts the target's own greedy continuation (known in advance), with a scheduled number of
    leading correct tokens per call; the rest are wrong. Records every call for plumbing checks."""

    def __init__(self, cfg, greedy_seq, n_correct=lambda call: 10 ** 9, vocab=97):
        self.cfg, self.seq, self.n_correct, self.vocab = cfg, greedy_seq, n_correct, vocab
        self.calls = []

    def draft_from_cache(self, anchor, cache, summary):
        P = cache.length
        self.calls.append({"anchor": anchor, "P": P, "summary": np.array(summary),
                           "latents": [tuple(np.array(a) for a in cache.latents(m)) for m in self.cfg.target_layers]})
        good = self.n_correct(len(self.calls) - 1)
        logits = np.zeros((self.cfg.block_size, self.vocab), dtype=np.float32)
        for j in range(self.cfg.block_size):
            right = self.seq[P + 1 + j] if P + 1 + j < len(self.seq) else 0
            logits[j, right if j < good else (right + 1) % self.vocab] = 10.0
        return logits


def greedy_sequence(target, n_new):
    out, _ = autoregressive(target, PROMPT, n_new)
    return PROMPT + out


# --- greedy: output is exactly the verifier's ---------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("n_new", [1, 2, 13])
def test_greedy_equals_autoregressive(models, k, n_new):
    target, drafter = models
    ref, _ = autoregressive(target, PROMPT, n_new)
    out, stats = speculative_block(target, drafter, PROMPT, n_new, k=k)
    assert out == ref
    assert stats.accepted + stats.iterations == n_new  # every iteration adds n accepted + 1
    assert stats.drafted == sum(min(k, n_new - p - 1) for p in np.cumsum([0] + [a + 1 for a in stats.accepted_per_iter])[:-1])


def test_perfect_drafts_are_all_accepted(models):
    target, drafter = models
    n_new, k = 20, drafter.cfg.block_size
    oracle = OracleDrafter(drafter.cfg, greedy_sequence(target, n_new))
    out, stats = speculative_block(target, oracle, PROMPT, n_new)
    assert out == greedy_sequence(target, n_new)[len(PROMPT):]
    assert stats.accepted == stats.drafted and stats.iterations == 4  # 5 + 5 + 5 + 5 tokens
    assert stats.accepted_per_iter == [k] * 4 and stats.target_calls == 4 and stats.draft_calls == 4


def test_wrong_drafts_still_give_the_verifier_output(models):
    target, drafter = models
    oracle = OracleDrafter(drafter.cfg, greedy_sequence(target, 12), n_correct=lambda call: 0)
    out, stats = speculative_block(target, oracle, PROMPT, 12)
    assert out == greedy_sequence(target, 12)[len(PROMPT):]
    assert stats.accepted == 0 and stats.iterations == 12


def test_each_call_sees_the_right_anchor_cache_and_summary(models):
    """With partial acceptance (0, 1, 2, 3, 4, 0, ... correct drafts), every drafter call gets:
    the anchor at position P = cache.length, cache latents equal to a fresh verifier pass over
    the first P tokens (so crops after rejections are right), and the summary = the verifier's
    hidden state at P - 1 (the last accepted position)."""
    target, drafter = models
    n_new = 25
    seq = greedy_sequence(target, n_new)
    oracle = OracleDrafter(drafter.cfg, seq, n_correct=lambda call: call % (drafter.cfg.block_size + 1))
    out, stats = speculative_block(target, oracle, PROMPT, n_new)
    assert out == seq[len(PROMPT):]
    assert set(stats.accepted_per_iter) >= {0, 1, 2, 3}  # crops of every size were exercised
    taps = list(drafter.cfg.tap_layers)
    for call in oracle.calls:
        P = call["P"]
        assert call["anchor"] == seq[P]
        fresh = target.new_cache()
        _, hidden = target.forward(seq[:P], fresh, hidden_layers=taps)
        np.testing.assert_allclose(call["summary"], hidden[:, -1], atol=1e-5, rtol=1e-5)
        for (c_kv, k_rope), m in zip(call["latents"], drafter.cfg.target_layers):
            ref_c, ref_k = fresh.latents(m)
            np.testing.assert_allclose(c_kv, ref_c, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(k_rope, ref_k, atol=1e-5, rtol=1e-5)


def test_input_checks(models):
    target, drafter = models
    with pytest.raises(ValueError, match="k must be"):
        speculative_block(target, drafter, PROMPT, 5, k=drafter.cfg.block_size + 1)
    with pytest.raises(ValueError, match="k must be"):
        speculative_block(target, drafter, PROMPT, 5, k=0)
    with pytest.raises(ValueError, match="2 prompt tokens"):
        speculative_block(target, drafter, PROMPT[:1], 5)
    assert speculative_block(target, drafter, PROMPT, 0)[0] == []


# --- sampling: output distribution equals the verifier's ----------------------------------------

# The tiny verifier's logits reach ~27, so at temperature 1 it is effectively deterministic (top
# probability 1.000). At 4 its next-token entropy is ~2.6 nats, so sampling is really exercised.
TEMPERATURE = 4.0


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


class BoostedVerifierDrafter:
    """Drafts from the verifier's own next-token distribution with its 2nd..9th most likely tokens
    boosted by e^1: q overlaps p substantially (TV ~0.2), unlike an untrained drafter. (Computes p
    by feeding the anchor to the verifier's cache and cropping it back.)"""

    def __init__(self, cfg, target, temperature=TEMPERATURE):
        self.cfg, self.target, self.temperature = cfg, target, temperature

    def draft_from_cache(self, anchor, cache, summary):
        P = cache.length
        logits = self.target.forward([anchor], cache, last_only=True)[-1]
        cache.crop(P)
        boosted = np.isin(np.arange(len(logits)), np.argsort(-logits)[1:9])
        row = logits + self.temperature * 1.0 * boosted
        return np.tile(row, (self.cfg.block_size, 1)).astype(np.float32)


def first_draft_distribution(target, drafter, temperature):
    """q for the first drafted token, computed exactly as speculative_block's first iteration does."""
    cache = target.new_cache()
    _, hidden = target.forward(PROMPT[:-1], cache, last_only=True, hidden_layers=list(drafter.cfg.tap_layers))
    return softmax(drafter.draft_from_cache(PROMPT[-1], cache, hidden[:, -1])[0] / temperature)


def exact_marginals(target, temperature):
    """Exact distributions of the first two sampled tokens under the verifier alone."""
    def probs(logits):
        z = logits.astype(np.float64) / temperature
        p = np.exp(z - z.max())
        return p / p.sum()

    cache = target.new_cache()
    p1 = probs(target.forward(PROMPT, cache, last_only=True)[-1])
    p2 = np.zeros_like(p1)
    for t in range(len(p1)):
        p2 += p1[t] * probs(target.forward([t], cache, last_only=True)[-1])
        cache.crop(len(PROMPT))
    return p1, p2


@pytest.mark.parametrize("which", ["trained_drafter", "boosted_verifier"])
def test_sampling_matches_the_verifier_distribution(models, which):
    """Rejection sampling with parallel drafts is lossless: the first two output tokens follow the
    verifier's exact marginals.

    Besides whole-distribution TV checks, it tests the sharpest consequence: on the set S of tokens
    the drafter over-weights (q > p), the output frequency must be p(S). Accepting too often
    gives q(S); resampling a rejection from p instead of max(p - q, 0) gives (1 + TV(p, q)) p(S),
    which these drafters shift by 17 and 6 standard deviations.
    """
    target, drafter = models
    N = 3000
    if which == "boosted_verifier":
        drafter = BoostedVerifierDrafter(drafter.cfg, target)
    p1, p2 = exact_marginals(target, TEMPERATURE)
    q1 = first_draft_distribution(target, drafter, TEMPERATURE)
    over = q1 > p1
    assert 0.1 < 0.5 * np.abs(p1 - q1).sum() < 0.9  # q overlaps p partly: both paths matter

    rng = np.random.default_rng(0)
    counts = np.zeros((2, len(p1)))
    accepted = 0
    for _ in range(N):
        out, stats = speculative_block(target, drafter, PROMPT, 2, temperature=TEMPERATURE, rng=rng)
        counts[0, out[0]] += 1
        counts[1, out[1]] += 1
        accepted += stats.accepted
    assert 0.05 * N < accepted < 0.95 * N  # both accept and reject paths were taken often

    p_over = p1[over].sum()
    sigma = np.sqrt(p_over * (1 - p_over) / N)
    assert abs(counts[0, over].sum() / N - p_over) < 4 * sigma + 0.005, (counts[0, over].sum() / N, p_over)

    tv = [0.5 * np.abs(counts[i] / N - p).sum() for i, p in enumerate((p1, p2))]
    direct = np.random.default_rng(1)  # the same statistic for N exact samples, as a noise scale
    noise = [0.5 * np.abs(np.bincount(direct.choice(len(p), N, p=p), minlength=len(p)) / N - p).sum()
             for p in (p1, p2)]
    assert tv[0] < 1.5 * noise[0] + 0.01 and tv[1] < 1.5 * noise[1] + 0.01, (tv, noise)


# --- benchmark plumbing ---------------------------------------------------------------------------

def test_bench_configs_and_summary(models, tmp_path):
    target, drafter = models
    prompts = [PROMPT, PROMPT[::-1]]
    rows = bench.run_config("ar_target", target, None, prompts, 10, 0, "distilgpt2", "mla-gpt2-medium")
    for cfg in ("hspec", "hspec_k2", "hspec_t1.0"):
        rows += bench.run_config(cfg, target, None, prompts, 10, 0, "distilgpt2", "mla-gpt2-medium", hspec=drafter)
    assert {r["draft"] for r in rows if r["config"] != "ar_target"} == {"hspec"}
    assert all(r["target"] == "mla-gpt2-medium" for r in rows)
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    buf = io.StringIO()
    with redirect_stdout(buf):
        summarize.main([str(path)])
    table = buf.getvalue()
    for cfg in ("hspec [hspec]", "hspec_k2 [hspec]"):
        line = next(l for l in table.splitlines() if l.startswith(f"| {cfg} |"))
        assert line.rstrip().endswith("| 2/2 |"), line  # identical to the verifier's greedy output
    assert "n/a (sampling)" in next(l for l in table.splitlines() if l.startswith("| hspec_t1.0 [hspec] |"))

    with pytest.raises(ValueError, match="unknown config"):
        bench.run_config("hspecc", target, None, prompts, 4, 0, "x")
    with pytest.raises(ValueError, match="H-Spec drafter"):
        bench.run_config("hspec", target, None, prompts, 4, 0, "x")
    with pytest.raises(ValueError, match="classic draft"):
        bench.run_config("spec_k2", target, None, prompts, 4, 0, "x")


def test_micro_times_one_drafter_pass(models, tmp_path):
    target, drafter = models
    out = tmp_path / "micro.jsonl"
    bench.micro(target, None, PROMPT, str(out), "distilgpt2", "mla-gpt2-medium", hspec=drafter)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["model"] for r in rows} == {"mla-gpt2-medium", "hspec-drafter"}
    hrow = next(r for r in rows if r["model"] == "hspec-drafter")
    assert hrow["T"] == drafter.cfg.block_size and hrow["median_s"] > 0


# --- CUDA ---------------------------------------------------------------------------------------

def test_cuda_greedy_equals_autoregressive(cuda_backend, tmp_path):
    verifier = make_verifier()
    target = MiniTorchMLAGPT2(str(export_verifier(verifier.state_dict(), dataclasses.asdict(verifier.cfg), tmp_path / "v")),
                              backend=cuda_backend)
    drafter = MiniTorchHSpecDrafter.from_verifier(str(export_drafter(make_drafter(verifier), tmp_path / "d")), target,
                                                  backend=cuda_backend)
    ref, _ = autoregressive(target, PROMPT, 16)
    for k in (1, drafter.cfg.block_size):
        assert speculative_block(target, drafter, PROMPT, 16, k=k)[0] == ref
    oracle = OracleDrafter(drafter.cfg, PROMPT + ref, n_correct=lambda call: call % 3)
    out, stats = speculative_block(target, oracle, PROMPT, 16)
    assert out == ref and stats.accepted > 0
