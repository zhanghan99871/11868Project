"""Run from repository root after compiling mla_kernel.so.
python -m pytest kernel_tests/test_mla_backward.py -v
"""
import ctypes
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SO = ROOT / 'minitorch' / 'cuda_kernels' / 'mla_kernel.so'


def ptr_array():
    return np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')


def cuda_backward(lib, go, qn, qr, c, kr, w, bias, query_start):
    B, H, Q, Dn = qn.shape
    K, R = c.shape[1:]
    Dr = qr.shape[-1]
    Dv = w.shape[0] // H - Dn

    arrays = [
        np.zeros_like(x)
        for x in (qn, qr, c, kr, w, bias)
    ]

    f = lib.launch_mla_backward

    f.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int] * 9
        + [ctypes.c_float, ctypes.c_void_p]
    )
    f.restype = ctypes.c_int

    def ptr(x):
        assert x.dtype == np.float32
        assert x.flags["C_CONTIGUOUS"]
        return ctypes.c_void_p(x.ctypes.data)

    status = f(
        ptr(go),
        ptr(qn),
        ptr(qr),
        ptr(c),
        ptr(kr),
        ptr(w),
        ptr(bias),

        ptr(arrays[0]),  # dq_nope
        ptr(arrays[1]),  # dq_rope
        ptr(arrays[2]),  # dc_kv
        ptr(arrays[3]),  # dk_rope
        ptr(arrays[4]),  # dw_up
        ptr(arrays[5]),  # db_up

        B, H, Q, K, Dn, Dr, Dv, R, query_start,
        (Dn + Dr) ** -0.5,
        None,
    )

    assert status == 0, f"CUDA backward returned {status}"

    return arrays


@pytest.mark.parametrize('query_start,Q,K', [(0, 3, 3), (3, 1, 4), (2, 2, 4)])
def test_backward_matches_pytorch(query_start, Q, K):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    assert SO.exists(), f'Compile {SO} first'
    lib = ctypes.CDLL(str(SO))
    rng = np.random.default_rng(43)
    B, H, Dn, Dr, Dv, R = 2, 3, 4, 2, 5, 7
    def rand(*shape):
        return np.ascontiguousarray(rng.normal(size=shape).astype('float32') * .25)
    vals = [rand(B,H,Q,Dn), rand(B,H,Q,Dr), rand(B,K,R),
            rand(B,K,Dr), rand(H*(Dn+Dv),R), rand(H*(Dn+Dv))]
    go = rand(B,H,Q,Dv)
    actual = cuda_backward(lib, go, *vals, query_start)

    qn,qr,c,kr,w,bias = [torch.tensor(v, device='cuda', requires_grad=True) for v in vals]
    expanded = F.linear(c, w, bias).view(B,K,H,Dn+Dv).transpose(1,2)
    kn, vv = expanded.split((Dn,Dv), dim=-1)
    keys = torch.cat((kn,kr[:,None].expand(B,H,K,Dr)), dim=-1)
    q = torch.cat((qn,qr), dim=-1)
    scores = q @ keys.transpose(-1,-2) * ((Dn+Dr)**-0.5)
    mask = torch.arange(K,device='cuda')[None,:] > (
        query_start+torch.arange(Q,device='cuda')[:,None])
    scores = scores.masked_fill(mask[None,None], -torch.inf)
    out = torch.softmax(scores, dim=-1) @ vv
    (out * torch.tensor(go, device='cuda')).sum().backward()
    for name, got, tensor in zip(('qn','qr','c','kr','w','bias'), actual,
                                 (qn,qr,c,kr,w,bias)):
        np.testing.assert_allclose(got, tensor.grad.detach().cpu().numpy(),
                                   atol=3e-4, rtol=3e-4, err_msg=name)
