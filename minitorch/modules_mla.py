"""MiniTorch decoder-only Transformer with Multi-head Latent Attention (MLA).

Designed for the host-backed CudaKernelOps.mla_fw / mla_bw from the MLA Step 2
kernel. FP32 implementation; attention probabilities are not exposed, so
attention-probability dropout is unsupported (p_dropout must be 0 for training).

Cache is (compressed_c_kv, rotated_shared_k_rope) for each layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .autodiff import Context
from .cuda_kernel_ops import CudaKernelOps
from .module import Module, Parameter
from .modules_basic import Dropout, Embedding, LayerNorm1d, Linear
from .nn import GELU, softmax
from .tensor import Tensor
from .tensor_functions import Function, ones, tensor_from_numpy, zeros
from .tensor_ops import TensorBackend

KVCache = Tuple[Tensor, Tensor]


def _constant(value: float, backend: TensorBackend) -> Tensor:
    return tensor_from_numpy(np.asarray([value], dtype=np.float32), backend=backend)


def _from_array(array: np.ndarray, backend: TensorBackend) -> Tensor:
    return tensor_from_numpy(np.ascontiguousarray(array, dtype=np.float32), backend=backend)


class _SliceLast(Function):
    """Slice [start:start+length] along last axis, with differentiable scatter-back."""

    @staticmethod
    def forward(ctx: Context, x: Tensor, start: Tensor, length: Tensor) -> Tensor:
        i, n = int(start.item()), int(length.item())
        if i < 0 or n <= 0 or i + n > x.shape[-1]:
            raise ValueError("invalid last-dimension slice")
        ctx.save_for_backward(x.shape, i, n, x.backend)
        return _from_array(x.to_numpy()[..., i:i+n], x.backend)

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        shape, i, n, backend = ctx.saved_values
        result = np.zeros(shape, dtype=np.float32)
        result[..., i:i+n] = grad.to_numpy()
        return _from_array(result, backend), 0.0, 0.0


def _slice_last(x: Tensor, start: int, length: int) -> Tensor:
    return _SliceLast.apply(x, _constant(start, x.backend), _constant(length, x.backend))


class _ConcatTime(Function):
    """Concatenate two tensors along time (axis 1), preserving gradients."""

    @staticmethod
    def forward(ctx: Context, old: Tensor, new: Tensor) -> Tensor:
        if len(old.shape) != 3 or len(new.shape) != 3:
            raise ValueError("KV caches must be [B,T,D]")
        if old.shape[0] != new.shape[0] or old.shape[2] != new.shape[2]:
            raise ValueError("incompatible KV cache tensors")
        ctx.save_for_backward(old.shape[1], old.backend)
        return _from_array(np.concatenate([old.to_numpy(), new.to_numpy()], axis=1), old.backend)

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        offset, backend = ctx.saved_values
        arr = grad.to_numpy()
        return (_from_array(arr[:, :offset, :], backend),
                _from_array(arr[:, offset:, :], backend))


class _RepeatHeads(Function):
    """Expand [B,K,D] to [B,H,K,D]; backward sums over the head axis."""

    @staticmethod
    def forward(ctx: Context, x: Tensor, n_heads: Tensor) -> Tensor:
        if len(x.shape) != 3:
            raise ValueError("shared rotary keys must have shape [B,K,D]")
        h = int(n_heads.item())
        if h <= 0:
            raise ValueError("n_heads must be positive")
        ctx.save_for_backward(x.backend)
        return _from_array(np.repeat(x.to_numpy()[:, None, :, :], h, axis=1), x.backend)

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        (backend,) = ctx.saved_values
        return _from_array(grad.to_numpy().sum(axis=1), backend), 0.0


class _RoPE(Function):
    """Interleaved partial RoPE; accepts [B,H,T,D] or [B,T,D]."""

    @staticmethod
    def forward(ctx: Context, x: Tensor, start: Tensor) -> Tensor:
        offset = int(start.item())
        d = x.shape[-1]
        if d <= 0 or d % 2:
            raise ValueError("RoPE dimension must be positive and even")
        if len(x.shape) not in (3, 4):
            raise ValueError("RoPE expects [B,T,D] or [B,H,T,D]")
        t = x.shape[-2]
        angles = ((offset + np.arange(t, dtype=np.float32))[:, None]
                  * (10000.0 ** (-np.arange(0, d, 2, dtype=np.float32) / d))[None, :])
        cos = np.cos(angles)
        sin = np.sin(angles)
        if len(x.shape) == 4:
            cos, sin = cos[None, None], sin[None, None]
        else:
            cos, sin = cos[None], sin[None]
        pair = x.to_numpy().reshape(*x.shape[:-1], d // 2, 2)
        even, odd = pair[..., 0], pair[..., 1]
        out = np.stack((even * cos - odd * sin,
                        even * sin + odd * cos), axis=-1).reshape(x.shape)
        ctx.save_for_backward(offset, d, len(x.shape), x.backend)
        return _from_array(out, x.backend)

    @staticmethod
    def backward(ctx: Context, grad: Tensor):
        offset, d, ndim, backend = ctx.saved_values
        t = grad.shape[-2]
        angles = ((offset + np.arange(t, dtype=np.float32))[:, None]
                  * (10000.0 ** (-np.arange(0, d, 2, dtype=np.float32) / d))[None, :])
        cos = np.cos(angles)
        sin = np.sin(angles)
        if ndim == 4:
            cos, sin = cos[None, None], sin[None, None]
        else:
            cos, sin = cos[None], sin[None]
        pair = grad.to_numpy().reshape(*grad.shape[:-1], d // 2, 2)
        even, odd = pair[..., 0], pair[..., 1]
        # Transpose (inverse) of the forward rotation.
        dx = np.stack((even * cos + odd * sin,
                       -even * sin + odd * cos), axis=-1).reshape(grad.shape)
        return _from_array(dx, backend), 0.0


def apply_rope(x: Tensor, start_position: int = 0) -> Tensor:
    return _RoPE.apply(x, _constant(start_position, x.backend))


class _MLAFused(Function):
    """MiniTorch autograd adapter around the native forward/backward launchers."""

    @staticmethod
    def forward(ctx: Context, q_nope: Tensor, q_rope: Tensor,
                c_kv: Tensor, k_rope: Tensor, w_up: Tensor,
                b_up: Tensor, query_start: Tensor) -> Tensor:
        start = int(query_start.item())
        ctx.save_for_backward(q_nope, q_rope, c_kv, k_rope, w_up, b_up, start)
        return CudaKernelOps.mla_fw(
            q_nope, q_rope, c_kv, k_rope, w_up, b_up, query_start=start)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        qn, qr, ck, kr, w, b, start = ctx.saved_values
        gqn, gqr, gck, gkr, gw, gb = CudaKernelOps.mla_bw(
            grad_output, qn, qr, ck, kr, w, b, query_start=start)
        return gqn, gqr, gck, gkr, gw, gb, 0.0


@dataclass
class MLAConfig:
    n_vocab: int = 50257
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_positions: int = 1024
    p_dropout: float = 0.0
    ln_eps: float = 1e-5
    bias: bool = True
    middle_dim: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    use_kv_norm: bool = True

    def __post_init__(self):
        for field in ("n_vocab", "n_embd", "n_head", "n_layer", "n_positions",
                      "kv_lora_rank", "qk_nope_head_dim", "v_head_dim"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.q_lora_rank is not None and self.q_lora_rank <= 0:
            raise ValueError("q_lora_rank must be positive")
        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be positive and even")
        if self.ln_eps <= 0 or not (0 <= self.p_dropout < 1):
            raise ValueError("invalid ln_eps or p_dropout")
        if self.middle_dim is None:
            self.middle_dim = 4 * self.n_embd


class RMSNorm(Module):
    def __init__(self, dim: int, backend: TensorBackend, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weights = Parameter(ones((dim,), backend=backend))

    def forward(self, x: Tensor) -> Tensor:
        variance = (x * x).mean(dim=len(x.shape) - 1)
        return x * ((variance + self.eps) ** -0.5) * self.weights.value


class MLAAttention(Module):
    def __init__(self, cfg: MLAConfig, backend: TensorBackend,
                 use_fused_kernel: bool = True):
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.use_fused_kernel = use_fused_kernel
        self.attn_dropout = Dropout(cfg.p_dropout)
        h, dn, dr, dv = (cfg.n_head, cfg.qk_nope_head_dim,
                         cfg.qk_rope_head_dim, cfg.v_head_dim)
        q_out = h * (dn + dr)
        if cfg.q_lora_rank is None:
            self.q_proj = Linear(cfg.n_embd, q_out, bias=cfg.bias, backend=backend)
        else:
            self.q_down = Linear(cfg.n_embd, cfg.q_lora_rank, bias=cfg.bias, backend=backend)
            self.q_norm = RMSNorm(cfg.q_lora_rank, backend)
            self.q_up = Linear(cfg.q_lora_rank, q_out, bias=cfg.bias, backend=backend)

        self.kv_down = Linear(cfg.n_embd, cfg.kv_lora_rank + dr, bias=cfg.bias, backend=backend)
        if cfg.use_kv_norm:
            self.kv_norm = RMSNorm(cfg.kv_lora_rank, backend)
        self.kv_up = Linear(cfg.kv_lora_rank, h * (dn + dv), bias=cfg.bias, backend=backend)
        self.out_projection = Linear(h * dv, cfg.n_embd, bias=cfg.bias, backend=backend)
        self.out_dropout = Dropout(cfg.p_dropout)

    def forward(self, x: Tensor, past_key_value: Optional[KVCache] = None,
                use_cache: bool = False) -> Tuple[Tensor, Optional[KVCache]]:
        if len(x.shape) != 3 or x.shape[2] != self.cfg.n_embd:
            raise ValueError("x must be [B,T,n_embd]")
        b, q_len, emb = x.shape
        past_len = 0 if past_key_value is None else past_key_value[0].shape[1]
        k_len = past_len + q_len
        if k_len > self.cfg.n_positions:
            raise ValueError("sequence exceeds n_positions")
        if self.use_fused_kernel and self.training and self.cfg.p_dropout:
            raise ValueError("fused MLA does not implement attention-probability dropout; set p_dropout=0 or use_fused_kernel=False")

        flat = x.contiguous().view(b * q_len, emb)
        if self.cfg.q_lora_rank is None:
            q = self.q_proj(flat)
        else:
            q = self.q_up(self.q_norm(self.q_down(flat)))
        h, dn, dr, dv = (self.cfg.n_head, self.cfg.qk_nope_head_dim,
                         self.cfg.qk_rope_head_dim, self.cfg.v_head_dim)
        q = q.contiguous().view(b, q_len, h, dn + dr)
        qn = _slice_last(q, 0, dn).permute(0, 2, 1, 3).contiguous()
        qr = _slice_last(q, dn, dr).permute(0, 2, 1, 3).contiguous()
        qr = apply_rope(qr, past_len)

        kv = self.kv_down(flat).contiguous().view(
            b, q_len, self.cfg.kv_lora_rank + dr)
        ck = _slice_last(kv, 0, self.cfg.kv_lora_rank)
        kr = _slice_last(kv, self.cfg.kv_lora_rank, dr)
        if self.cfg.use_kv_norm:
            ck = self.kv_norm(ck)
        kr = apply_rope(kr, past_len)
        if past_key_value is not None:
            old_ck, old_kr = past_key_value
            if old_ck.shape != (b, past_len, self.cfg.kv_lora_rank) or old_kr.shape != (b, past_len, dr):
                raise ValueError("invalid KV cache shape")
            ck = _ConcatTime.apply(old_ck, ck)
            kr = _ConcatTime.apply(old_kr, kr)
        present = (ck, kr) if use_cache else None

        if self.use_fused_kernel:
            # MiniTorch Linear stores weights [in_features,out_features]; CUDA
            # MLA expects W_up [out_features,in_features]. Its transpose is
            # differentiable, so gradients reach kv_up.weights.
            w_up = self.kv_up.weights.value.permute(1, 0).contiguous()
            if self.kv_up.bias is None:
                b_up = zeros((h * (dn + dv),), backend=self.backend)
            else:
                b_up = self.kv_up.bias.value
            context = _MLAFused.apply(qn, qr, ck, kr, w_up, b_up,
                                      _constant(past_len, self.backend))
        else:
            # Materialize per-head K_nope and V, matching the PyTorch reference.
            expanded = self.kv_up(ck.contiguous().view(b * k_len, self.cfg.kv_lora_rank))
            expanded = expanded.contiguous().view(b, k_len, h, dn + dv)
            expanded = expanded.permute(0, 2, 1, 3).contiguous()
            kn = _slice_last(expanded, 0, dn)
            v = _slice_last(expanded, dn, dv)
            kr_heads = _RepeatHeads.apply(kr, _constant(h, self.backend))

            # This avoids concatenation: QK^T = Q_nope K_nope^T + Q_rope K_rope^T.
            scores = (qn @ kn.permute(0, 1, 3, 2).contiguous()
                      + qr @ kr_heads.permute(0, 1, 3, 2).contiguous())
            scores = scores * ((dn + dr) ** -0.5)
            # Absolute-position causal mask handles both prefill and KV caching.
            future = np.arange(k_len)[None, :] > (past_len + np.arange(q_len)[:, None])
            causal_mask = np.broadcast_to(
                np.where(future, -np.finfo(np.float32).max, 0.0)[None, None, :, :],
                (b, h, q_len, k_len),
            )
            scores = scores + _from_array(causal_mask, self.backend)
            probs = self.attn_dropout(softmax(scores, dim=3))
            context = probs @ v
        context = context.permute(0, 2, 1, 3).contiguous().view(b*q_len, h*dv)
        out = self.out_projection(context).view(b, q_len, emb)
        return self.out_dropout(out), present


class FeedForward(Module):
    def __init__(self, cfg: MLAConfig, backend: TensorBackend):
        super().__init__()
        self.linear_in = Linear(cfg.n_embd, cfg.middle_dim, bias=cfg.bias, backend=backend)
        self.linear_out = Linear(cfg.middle_dim, cfg.n_embd, bias=cfg.bias, backend=backend)
        self.dropout = Dropout(cfg.p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, e = x.shape
        out = GELU(self.linear_in(x.contiguous().view(b*t, e)))
        return self.dropout(self.linear_out(out)).view(b, t, e)


class TransformerLayer(Module):
    def __init__(self, cfg: MLAConfig, backend: TensorBackend,
                 use_fused_layernorm: bool = False,
                 use_fused_kernel: bool = True):
        super().__init__()
        self.ln_1 = LayerNorm1d(cfg.n_embd, eps=cfg.ln_eps,
                                backend=backend, use_fused_kernel=use_fused_layernorm)
        self.attention = MLAAttention(cfg, backend, use_fused_kernel=use_fused_kernel)
        self.ln_2 = LayerNorm1d(cfg.n_embd, eps=cfg.ln_eps,
                                backend=backend, use_fused_kernel=use_fused_layernorm)
        self.ff = FeedForward(cfg, backend)

    def forward(self, x: Tensor, past_key_value: Optional[KVCache] = None,
                use_cache: bool = False) -> Tuple[Tensor, Optional[KVCache]]:
        b, t, e = x.shape
        a, present = self.attention(self.ln_1(x.contiguous().view(b*t, e)).view(b, t, e),
                                    past_key_value=past_key_value, use_cache=use_cache)
        x = x + a
        x = x + self.ff(self.ln_2(x.contiguous().view(b*t, e)).view(b, t, e))
        return x, present


class DecoderLM(Module):
    def __init__(self, cfg: MLAConfig, backend: TensorBackend,
                 use_fused_layernorm: bool = False,
                 use_fused_kernel: bool = True):
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.token_embeddings = Embedding(cfg.n_vocab, cfg.n_embd, backend=backend)
        self.position_embeddings = Embedding(cfg.n_positions, cfg.n_embd, backend=backend)
        self.dropout = Dropout(cfg.p_dropout)
        self.layers: List[TransformerLayer] = []
        for i in range(cfg.n_layer):
            layer = TransformerLayer(cfg, backend,
                                     use_fused_layernorm=use_fused_layernorm,
                                     use_fused_kernel=use_fused_kernel)
            setattr(self, f"t_layer_{i + 1}", layer)  # register in MiniTorch Module
            self.layers.append(layer)
        self.ln = LayerNorm1d(cfg.n_embd, eps=cfg.ln_eps,
                              backend=backend, use_fused_kernel=use_fused_layernorm)
        self.lm_head = Linear(cfg.n_embd, cfg.n_vocab, bias=False, backend=backend)
        # Not tied by default: MiniTorch Embedding.weights is [vocab,embd]
        # and Linear.weights is [embd,vocab], and Parameter registrations differ.

    def forward(self, idx: Tensor,
                past_key_values: Optional[List[KVCache]] = None,
                use_cache: bool = False):
        if len(idx.shape) != 2:
            raise ValueError("idx must have shape [B,T]")
        b, t = idx.shape
        if past_key_values is not None and len(past_key_values) != self.cfg.n_layer:
            raise ValueError("one KV cache entry per layer required")
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        if past_len + t > self.cfg.n_positions:
            raise ValueError("sequence exceeds n_positions")
        positions = _from_array(np.arange(past_len, past_len+t, dtype=np.float32)[None, :], self.backend)
        x = self.dropout(self.token_embeddings(idx) + self.position_embeddings(positions))
        new_cache: List[KVCache] = []
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[i]
            x, present = layer(x, past_key_value=past, use_cache=use_cache)
            if use_cache:
                assert present is not None
                new_cache.append(present)
        logits = self.lm_head(self.ln(x.contiguous().view(b*t, self.cfg.n_embd)))
        logits = logits.view(b, t, self.cfg.n_vocab)
        return (logits, new_cache) if use_cache else logits
