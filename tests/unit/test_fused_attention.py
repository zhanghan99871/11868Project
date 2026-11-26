"""CUDA-only correctness and mutation checks for the fused softmax."""
import numpy as np
import pytest
import torch
from minitorch.tensor import tensor_from_numpy
from minitorch.tensor_functions import Attn_Softmax


def mt(arr, backend):
    return tensor_from_numpy(np.asarray(arr, dtype=np.float32), backend=backend)


def test_fused_causal_softmax_matches_torch(cuda_backend):
    # Small inputs so a full [B,H,T,T] causal mask fits in both implementations.
    b,h,t = 2,2,4
    rng = np.random.default_rng(42)
    x = rng.normal(size=(b,h,t,t)).astype(np.float32)
    mask = np.broadcast_to(np.triu(np.full((t,t), -1e8, dtype=np.float32), 1), x.shape).copy()
    expected = torch.softmax(torch.from_numpy(x + mask), dim=-1).numpy()
    got = Attn_Softmax.apply(mt(x, cuda_backend), mt(mask, cuda_backend)).to_numpy()
    np.testing.assert_allclose(got, expected, atol=2e-4, rtol=2e-4)


def test_softmax_forward_does_not_mutate_logits(cuda_backend):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(1,2,4,4)).astype(np.float32)
    original = mt(x, cuda_backend)
    mask = mt(np.zeros((1,2,4,4)), cuda_backend)
    _ = Attn_Softmax.apply(original, mask)
    np.testing.assert_array_equal(original.to_numpy(), x,
        err_msg="Fused softmax unexpectedly overwrote logits")


def test_softmax_backward_does_not_mutate_upstream_gradient(cuda_backend):
    x = np.arange(16, dtype=np.float32).reshape(1,1,4,4)/10
    prob = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
    upstream_np = np.linspace(-1,1,16,dtype=np.float32).reshape(1,1,4,4)
    upstream = mt(upstream_np, cuda_backend)
    _ = upstream.f.attn_softmax_bw(upstream, mt(prob, cuda_backend))
    np.testing.assert_array_equal(upstream.to_numpy(), upstream_np,
        err_msg="Fused softmax backward overwrote upstream gradient")


def test_fused_noncausal_softmax_matches_torch(cuda_backend):
    rng = np.random.default_rng(6)
    x = rng.normal(size=(2,2,4,4)).astype(np.float32)
    mask = np.zeros_like(x)
    expected = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
    got = Attn_Softmax.apply(mt(x, cuda_backend), mt(mask, cuda_backend)).to_numpy()
    np.testing.assert_allclose(got, expected, atol=2e-4, rtol=2e-4)
