"""The MiniTorch CUDA verifier (fused LayerNorm + fused MLA kernels) against the numpy backend
and the PyTorch model. Skipped without torch, pycuda, a GPU or the compiled kernels."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdec.decoding import autoregressive, speculative  # noqa: E402
from specdec.mla_model import MiniTorchMLAGPT2  # noqa: E402

from .helpers import (export, make_torch_model, numpy_verifier, token_ids,  # noqa: E402
                      torch_cache_and_hidden, torch_logits)

# fused kernels: float32 tiled matmuls, __expf softmax, E[x^2]-E[x]^2 LayerNorm
ATOL, RTOL = 2e-3, 1e-3


@pytest.fixture
def verifiers(exported, cuda_backend):
    model, out = exported
    return model, MiniTorchMLAGPT2(str(out), backend=cuda_backend), numpy_verifier(out)


def test_cuda_logits_match_numpy_and_pytorch(verifiers):
    model, gpu, cpu = verifiers
    assert gpu.resident and gpu.ops.name == "minitorch"
    ids = token_ids(24)
    got = gpu.forward(ids, gpu.new_cache())
    np.testing.assert_allclose(got, cpu.forward(ids, cpu.new_cache()), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(got, torch_logits(model, ids), atol=ATOL, rtol=RTOL)


def test_cuda_cache_and_hidden_taps_match_pytorch(verifiers):
    model, gpu, _ = verifiers
    ids = token_ids(20)
    cache = gpu.new_cache()
    _, hidden = gpu.forward(ids, cache, hidden_layers=[0, 2])
    ref_cache, ref_hidden = torch_cache_and_hidden(model, ids, [0, 2])
    for layer, (c_kv, k_rope) in enumerate(ref_cache):
        got_c, got_k = cache.latents(layer)
        np.testing.assert_allclose(got_c, c_kv, atol=ATOL, rtol=RTOL, err_msg=f"c_kv layer {layer}")
        np.testing.assert_allclose(got_k, k_rope, atol=ATOL, rtol=RTOL, err_msg=f"k_rope layer {layer}")
    for j, layer in enumerate([0, 2]):
        np.testing.assert_allclose(hidden[j], ref_hidden[layer], atol=ATOL, rtol=RTOL)


def test_cuda_incremental_and_crop(verifiers):
    _, gpu, _ = verifiers
    ids = token_ids(30)
    full = gpu.forward(ids, gpu.new_cache())

    cache = gpu.new_cache()
    parts = [gpu.forward(ids[a:b], cache) for a, b in [(0, 9), (9, 10), (10, 17), (17, 30)]]
    np.testing.assert_allclose(np.concatenate(parts), full, atol=ATOL, rtol=RTOL)

    cache = gpu.new_cache()
    gpu.forward(ids[:20], cache)
    gpu.forward(token_ids(6, seed=5), cache)
    cache.crop(20)
    np.testing.assert_allclose(gpu.forward(ids[20:], cache), full[20:], atol=ATOL, rtol=RTOL)


def test_resident_and_copied_weights_agree(exported, cuda_backend):
    _, out = exported
    ids = token_ids(16)
    resident = MiniTorchMLAGPT2(str(out), backend=cuda_backend, resident=True)
    copied = MiniTorchMLAGPT2(str(out), backend=cuda_backend, resident=False)
    assert not copied.resident
    np.testing.assert_allclose(resident.forward(ids, resident.new_cache()),
                               copied.forward(ids, copied.new_cache()), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("k", [1, 4])
def test_cuda_greedy_speculative_equals_autoregressive(verifiers, cuda_backend, tmp_path, k):
    _, target, _ = verifiers
    draft = MiniTorchMLAGPT2(str(export(make_torch_model("gpt2_style", seed=1), tmp_path / "draft")),
                             backend=cuda_backend)
    prompt = token_ids(8, seed=3)
    ref, _ = autoregressive(target, prompt, 16)
    out, _ = speculative(target, draft, prompt, 16, k)
    assert out == ref
