"""The verifier's model logic against the PyTorch MLA model, using the numpy backend (CPU).

The CUDA backend runs the same forward() code with different ops; test_cuda_verifier.py
compares it against this backend.
"""
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdec.mla_model import LatentCache, mla_attention  # noqa: E402

from .helpers import (ROOT, numpy_verifier, token_ids, torch_cache_and_hidden,  # noqa: E402
                      torch_logits)

ATOL = RTOL = 1e-4


def test_logits_match_pytorch(exported):
    model, out = exported
    verifier = numpy_verifier(out)
    ids = token_ids(24)
    np.testing.assert_allclose(verifier.forward(ids, verifier.new_cache()), torch_logits(model, ids),
                               atol=ATOL, rtol=RTOL)


def test_cache_holds_the_pytorch_latents(exported):
    model, out = exported
    verifier = numpy_verifier(out)
    ids = token_ids(24)
    cache = verifier.new_cache()
    verifier.forward(ids, cache)
    assert cache.length == len(ids)
    ref, _ = torch_cache_and_hidden(model, ids)
    for layer, (c_kv, k_rope) in enumerate(ref):
        got_c, got_k = cache.latents(layer)
        np.testing.assert_allclose(got_c, c_kv, atol=1e-5, rtol=1e-5, err_msg=f"c_kv layer {layer}")
        np.testing.assert_allclose(got_k, k_rope, atol=1e-5, rtol=1e-5, err_msg=f"k_rope layer {layer}")


def test_hidden_taps_match_pytorch_for_every_position(exported):
    model, out = exported
    verifier = numpy_verifier(out)
    ids = token_ids(20)
    layers = [2, 0]  # returned in the requested order
    logits, hidden = verifier.forward(ids, verifier.new_cache(), last_only=True, hidden_layers=layers)
    assert logits.shape == (1, verifier.vocab_size)
    assert hidden.shape == (len(layers), len(ids), verifier.n_embd)
    _, ref = torch_cache_and_hidden(model, ids, layers)
    for j, layer in enumerate(layers):
        np.testing.assert_allclose(hidden[j], ref[layer], atol=ATOL, rtol=RTOL, err_msg=f"layer {layer}")


def test_incremental_matches_full_forward(exported):
    _, out = exported
    verifier = numpy_verifier(out)
    ids = token_ids(30)
    full_cache = verifier.new_cache()
    full = verifier.forward(ids, full_cache)

    cache = verifier.new_cache()
    parts = [verifier.forward(ids[a:b], cache) for a, b in [(0, 9), (9, 10), (10, 17), (17, 30)]]
    np.testing.assert_allclose(np.concatenate(parts), full, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(cache.c_kv[:, :30], full_cache.c_kv[:, :30], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cache.k_rope[:, :30], full_cache.k_rope[:, :30], atol=1e-5, rtol=1e-5)


def test_crop_then_refeed_matches_uncropped(exported):
    _, out = exported
    verifier = numpy_verifier(out)
    ids = token_ids(30)
    full = verifier.forward(ids, verifier.new_cache())

    cache = verifier.new_cache()
    verifier.forward(ids[:20], cache)
    verifier.forward(token_ids(6, seed=5), cache)  # rejected drafts
    cache.crop(20)
    np.testing.assert_allclose(verifier.forward(ids[20:], cache), full[20:], atol=ATOL, rtol=RTOL)


def test_last_only_returns_the_last_row(exported):
    _, out = exported
    verifier = numpy_verifier(out)
    ids = token_ids(15)
    full = verifier.forward(ids, verifier.new_cache())
    np.testing.assert_allclose(verifier.forward(ids, verifier.new_cache(), last_only=True), full[-1:],
                               atol=1e-6, rtol=1e-6)


def test_limits_and_errors(exported):
    _, out = exported
    verifier = numpy_verifier(out)
    cfg = verifier.cfg
    assert verifier.max_len == cfg.n_positions  # requested 1024, clipped to the position table
    assert verifier.new_cache().floats_per_token == cfg.n_layer * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
    with pytest.raises(AssertionError):
        verifier.forward(token_ids(cfg.n_positions + 1), verifier.new_cache())
    with pytest.raises(ValueError, match="hidden_layers"):
        verifier.forward(token_ids(3), verifier.new_cache(), hidden_layers=[cfg.n_layer])
    cache = verifier.new_cache()
    verifier.forward(token_ids(5), cache)
    with pytest.raises(AssertionError):
        cache.crop(6)
    with pytest.raises(AssertionError):
        cache.latents(0, 6)


def test_gpt2_medium_mla_cache_size():
    # the trained verifier: 24 layers, kv_rank 512, rope_dim 32 (runs/mla-gpt2-medium/arguments.json)
    assert LatentCache(24, 512, 32, max_len=1).floats_per_token == 13_056  # vs 2 * 24 * 1024 for MHA


def test_absorbed_attention_equals_materialized():
    """The algebra the fused CUDA kernel relies on, checked against the materialized reference.

    The kernel folds W_K into the query (scores = (q_nope W_K) c^T + q_rope k_rope^T), drops the
    key bias (constant per query, cancels in softmax), and applies W_V after pooling the latents.
    """
    rng = np.random.default_rng(0)
    H, T, S, Dn, Dr, Dv, R, start = 3, 4, 9, 5, 4, 6, 7, 5
    q_nope = rng.standard_normal((H, T, Dn)).astype(np.float32)
    q_rope = rng.standard_normal((H, T, Dr)).astype(np.float32)
    c_kv = rng.standard_normal((S, R)).astype(np.float32)
    k_rope = rng.standard_normal((S, Dr)).astype(np.float32)
    w_up = rng.standard_normal((H * (Dn + Dv), R)).astype(np.float32)
    b_up = rng.standard_normal(H * (Dn + Dv)).astype(np.float32)
    ref = mla_attention(q_nope, q_rope, c_kv, k_rope, w_up, b_up, start)

    w = w_up.reshape(H, Dn + Dv, R)
    w_k, w_v, b_v = w[:, :Dn], w[:, Dn:], b_up.reshape(H, Dn + Dv)[:, Dn:]
    scores = ((q_nope @ w_k) @ c_kv.T + q_rope @ k_rope.T) * (Dn + Dr) ** -0.5
    scores[:, np.arange(S)[None, :] > (start + np.arange(T))[:, None]] = -np.inf
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    absorbed = (p @ c_kv) @ w_v.transpose(0, 2, 1) + b_v[:, None, :]
    np.testing.assert_allclose(absorbed, ref, atol=1e-4, rtol=1e-4)


def test_numpy_verifier_does_not_import_torch(exported):
    """specdec must stay torch-free: load and run the verifier in a fresh interpreter."""
    _, out = exported
    code = ("import sys; from specdec.mla_model import MiniTorchMLAGPT2; "
            f"v = MiniTorchMLAGPT2({str(out)!r}); v.forward([1, 2, 3], v.new_cache()); "
            "assert 'torch' not in sys.modules, 'torch was imported'")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
