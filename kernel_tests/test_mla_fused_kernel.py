"""Regression tests for fused vs unfused MiniTorch MLA.

Install at model_tests/test_modules_mla_fused_switch.py and run:
  python -m pytest model_tests/test_modules_mla_fused_switch.py -v
"""
from pathlib import Path
import numpy as np
import pytest

SO = Path(__file__).resolve().parents[1] / "minitorch" / "cuda_kernels" / "mla_kernel.so"


@pytest.fixture(scope="module")
def model_backend():
    if not SO.exists():
        pytest.skip(f"Compile MLA CUDA kernel: {SO}")
    import minitorch
    from minitorch.cuda_kernel_ops import CudaKernelOps
    from minitorch.modules_mla import MLAConfig, DecoderLM
    backend = minitorch.TensorBackend(CudaKernelOps)
    cfg = MLAConfig(
        n_vocab=41, n_embd=32, n_head=4, n_layer=2,
        n_positions=32, q_lora_rank=None, kv_lora_rank=12,
        qk_nope_head_dim=8, qk_rope_head_dim=4,
        v_head_dim=8, p_dropout=0.0, use_kv_norm=False,
    )
    np.random.seed(7)
    return DecoderLM(cfg, backend, use_fused_kernel=True), backend


def _tokens(backend, ids):
    import minitorch
    return minitorch.tensor_from_numpy(np.asarray(ids, dtype=np.float32), backend=backend)


def _switch(model, fused):
    for layer in model.layers:
        layer.attention.use_fused_kernel = fused


@pytest.mark.parametrize("ids", [[[1, 3, 5]], [[1, 3, 5], [2, 4, 6]]])
def test_fused_matches_unfused_logits(model_backend, ids):
    model, backend = model_backend
    model.eval()
    x = _tokens(backend, ids)
    _switch(model, False)
    expected = model(x).to_numpy()
    _switch(model, True)
    actual = model(x).to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_fused_matches_unfused_cached(model_backend):
    model, backend = model_backend
    model.eval()
    ids = [1, 3, 5, 7]
    results = {}
    for fused in (False, True):
        _switch(model, fused)
        cache = None
        pieces = []
        for token in ids:
            logits, cache = model(_tokens(backend, [[token]]),
                                  past_key_values=cache, use_cache=True)
            pieces.append(logits.to_numpy())
        results[fused] = np.concatenate(pieces, axis=1)
    np.testing.assert_allclose(results[True], results[False], rtol=2e-3, atol=2e-3)
