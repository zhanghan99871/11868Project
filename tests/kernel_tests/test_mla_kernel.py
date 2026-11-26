"""CUDA forward regression test for mla_kernel.cu (requires nvcc + NVIDIA GPU).

Build:
    nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC \
      -o mla_kernel.so mla_kernel.cu
Run:
    python -m pytest test_mla_kernel.py -v
"""
import ctypes
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

SO_PATH = (
    REPO_ROOT
    / "minitorch"
    / "cuda_kernels"
    / "mla_kernel.so"
)

def call_kernel(lib, qn, qr, c, kr, weight, bias, query_start):
    # All inputs are CPU contiguous float32 NumPy arrays.
    b, h, q, dn = qn.shape
    k, r = c.shape[1:]
    dr = qr.shape[-1]
    dv = weight.shape[0] // h - dn
    out = np.empty((b, h, q, dv), dtype=np.float32)
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    func = lib.launch_mla_forward
    func.argtypes = [ptr, ptr, ptr, ptr, ptr, ctypes.c_void_p, ptr] + [ctypes.c_int] * 9 + [ctypes.c_float, ctypes.c_void_p]
    func.restype = ctypes.c_int
    bias_ptr = None if bias is None else bias.ctypes.data
    status = func(qn, qr, c, kr, weight, bias_ptr, out,
                  b, h, q, k, dn, dr, dv, r, query_start,
                  (dn + dr) ** -0.5, None)
    assert status == 0, f"CUDA launcher returned {status}"
    return out


@pytest.mark.parametrize(
    "query_start,q,k",
    [(0, 5, 5), (5, 1, 6), (4, 3, 7)]
)
def test_mla_forward_matches_reconstructed_reference(query_start, q, k):
    if not torch.cuda.is_available():
        pytest.skip("Requires an NVIDIA GPU")

    if not SO_PATH.exists():
        pytest.fail(f"CUDA library not found: {SO_PATH}")

    lib = ctypes.CDLL(str(SO_PATH))
    rng = np.random.default_rng(123)
    b, h, dn, dr, dv, r = 2, 4, 12, 8, 16, 32

    def rand(*shape):
        return rng.normal(size=shape).astype(np.float32)

    qn = rand(b,h,q,dn)
    qr = rand(b,h,q,dr)
    c = rand(b,k,r)
    kr = rand(b,k,dr)
    w = rand(h*(dn+dv),r)
    bias = rand(h*(dn+dv))
    got = call_kernel(lib,qn,qr,c,kr,w,bias,query_start)

    device = 'cuda'
    qnt = torch.from_numpy(qn).to(device)
    qrt = torch.from_numpy(qr).to(device)
    ct = torch.from_numpy(c).to(device)
    krt = torch.from_numpy(kr).to(device)
    wt = torch.from_numpy(w).to(device)
    bt = torch.from_numpy(bias).to(device)
    expanded = F.linear(ct, wt, bt).view(b,k,h,dn+dv).transpose(1,2)
    kn, v = expanded.split((dn,dv),dim=-1)
    keys = torch.cat([kn,krt[:,None,:,:].expand(b,h,k,dr)],dim=-1)
    queries = torch.cat([qnt,qrt],dim=-1)
    scores = (queries @ keys.transpose(-1,-2)) * (dn+dr)**-0.5
    future = torch.arange(k,device=device)[None,:] > (query_start + torch.arange(q,device=device)[:,None])
    scores = scores.masked_fill(future[None,None,:,:],float('-inf'))
    expected = (scores.softmax(-1) @ v).cpu().numpy()
    np.testing.assert_allclose(got,expected,rtol=2e-4,atol=2e-4)
