"""Smoke checks for MiniTorch fused MLA module. Requires compiled MLA CUDA .so."""
from pathlib import Path
import numpy as np
import pytest

SO = Path(__file__).resolve().parents[1] / 'minitorch' / 'cuda_kernels' / 'mla_kernel.so'

@pytest.fixture(scope='module')
def setup_mla():
    if not SO.is_file():
        pytest.skip(f'CUDA MLA library not compiled at {SO}')
    import minitorch
    from minitorch.cuda_kernel_ops import CudaKernelOps
    from minitorch.modules_mla import MLAConfig, DecoderLM
    cfg = MLAConfig(n_vocab=41, n_embd=32, n_head=4, n_layer=2,
                    n_positions=32, q_lora_rank=None, kv_lora_rank=12,
                    qk_nope_head_dim=8, qk_rope_head_dim=4,
                    v_head_dim=8, p_dropout=0, use_kv_norm=False)
    backend = minitorch.TensorBackend(CudaKernelOps)
    np.random.seed(7)
    return DecoderLM(cfg, backend), backend


def tokens(data, backend):
    import minitorch
    return minitorch.tensor_from_numpy(np.asarray(data, dtype=np.float32), backend=backend)


def test_forward_shape(setup_mla):
    model, backend = setup_mla
    model.eval()
    idx = tokens([[1, 3, 5], [2, 4, 6]], backend)
    assert model(idx).shape == (2, 3, 41)


def test_cached_matches_full(setup_mla):
    model, backend = setup_mla
    model.eval()
    ids = [3, 5, 7, 9]
    full = model(tokens([ids], backend)).to_numpy()
    cache = None
    slices = []
    for i in ids:
        out, cache = model(tokens([[i]], backend), past_key_values=cache,
                           use_cache=True)
        slices.append(out.to_numpy())
    cached = np.concatenate(slices, axis=1)
    np.testing.assert_allclose(full, cached, rtol=1e-3, atol=1e-3)
    assert len(cache) == model.cfg.n_layer
    for c_kv, k_rope in cache:
        assert c_kv.shape == (1, len(ids), model.cfg.kv_lora_rank)
        assert k_rope.shape == (1, len(ids), model.cfg.qk_rope_head_dim)


def test_backward_reaches_kv_down(setup_mla):
    model, backend = setup_mla
    model.train()
    logits = model(tokens([[1, 2, 3]], backend))
    logits.sum().backward()
    grad = model.layers[0].attention.kv_down.weights.value.grad
    assert grad is not None
    assert np.isfinite(grad.to_numpy()).all()
