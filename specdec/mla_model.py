"""MLA GPT-2 inference (the speculative-decoding verifier) with a per-request latent KV cache.

The model is the PyTorch MLA decoder in torch_train/decoder_mla.py (GPT-2 medium converted to
Multi-head Latent Attention and fine-tuned), exported with torch_train/export_mla_verifier.py.

The cache stores only the MLA latents of each layer: c_kv [max_len, kv_lora_rank] and the shared
rotary key k_rope [max_len, qk_rope_head_dim], stored after RoPE like decoder_mla's cache.
Per-head K/V are never materialized: the fused attention kernel folds the key/value
up-projection into the query and output. For GPT-2 medium (24 layers, rank 512, RoPE dim 32)
the cache holds 13,056 floats per token, vs 49,152 for its MHA cache.

Two numeric backends run the same model code:
  backend=TensorBackend(CudaKernelOps)  MiniTorch matmuls with device-resident weights, the fused
      LayerNorm kernel and the fused MLA attention kernel (CudaKernelOps.mla_fw). Used for
      decoding and benchmarks.
  backend=None  a float32 numpy reference of every op. Runs on CPU; used by tests/verifier.

For an H-Spec-style drafter, forward() can also return the residual stream after selected
layers ("hidden taps"), and LatentCache.latents(layer) exposes a layer's cached latents so the
drafter can attend to them in place.
"""
import json
import os
from dataclasses import dataclass, fields
from typing import Optional, Sequence

import numpy as np

FORMAT = "mla-gpt2-minitorch-v1"  # written by torch_train/export_mla_verifier.py
ROPE_BASE = 10000.0
RMS_NORM_EPS = 1e-6  # decoder_mla.RMSNorm default


@dataclass(frozen=True)
class MLAConfig:
    """The fields of torch_train/decoder_mla.MLAConfig (p_dropout is training-only)."""
    n_vocab: int
    n_embd: int
    n_head: int
    n_layer: int
    n_positions: int
    ln_eps: float
    bias: bool
    middle_dim: int
    q_lora_rank: Optional[int]
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    use_kv_norm: bool
    p_dropout: float = 0.0

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        missing = names - set(d) - {"p_dropout"}
        unknown = set(d) - names
        if missing or unknown:
            raise ValueError(f"bad MLA config: missing {sorted(missing)}, unknown {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.qk_rope_head_dim <= 0 or cfg.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be positive and even")
        return cfg


def expected_shapes(cfg: MLAConfig) -> dict:
    """PyTorch state_dict names and shapes of decoder_mla.DecoderLM, without the tied lm_head."""
    D, H = cfg.n_embd, cfg.n_head
    Dn, Dr, Dv, R = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    shapes = {"token_embeddings.weight": (cfg.n_vocab, D),
              "position_embeddings.weight": (cfg.n_positions, D),
              "ln.weight": (D,), "ln.bias": (D,)}

    def linear(name, n_in, n_out):
        shapes[name + ".weight"] = (n_out, n_in)
        if cfg.bias:
            shapes[name + ".bias"] = (n_out,)

    for i in range(cfg.n_layer):
        p = f"layers.{i}."
        for ln in ("ln_1", "ln_2"):
            shapes[p + ln + ".weight"] = shapes[p + ln + ".bias"] = (D,)
        a = p + "attention."
        if cfg.q_lora_rank is None:
            linear(a + "q_proj", D, H * (Dn + Dr))
        else:
            linear(a + "q_down", D, cfg.q_lora_rank)
            shapes[a + "q_norm.weight"] = (cfg.q_lora_rank,)
            linear(a + "q_up", cfg.q_lora_rank, H * (Dn + Dr))
        linear(a + "kv_down", D, R + Dr)
        if cfg.use_kv_norm:
            shapes[a + "kv_norm.weight"] = (R,)
        linear(a + "kv_up", R, H * (Dn + Dv))
        linear(a + "out_projection", H * Dv, D)
        linear(p + "ff.linear_in", D, cfg.middle_dim)
        linear(p + "ff.linear_out", cfg.middle_dim, D)
    return shapes


# ---------------------------------------------------------------------------
# float32 numpy helpers (both backends use these for the small host-side steps)
# ---------------------------------------------------------------------------

def apply_rope(x: np.ndarray, start: int, base: float = ROPE_BASE) -> np.ndarray:
    """Interleaved RoPE (decoder_mla.apply_rope) on x [..., T, d] at positions start..start+T-1."""
    d, t = x.shape[-1], x.shape[-2]
    inv_freq = (base ** (-np.arange(0, d, 2, dtype=np.float32) / d)).astype(np.float32)
    angles = np.arange(start, start + t, dtype=np.float32)[:, None] * inv_freq[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    pair = x.reshape(*x.shape[:-1], d // 2, 2)
    even, odd = pair[..., 0], pair[..., 1]
    out = np.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1)
    return out.reshape(x.shape).astype(np.float32)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS) -> np.ndarray:
    inv_rms = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps))
    return ((x * inv_rms) * weight).astype(np.float32)


def layer_norm(x, gamma, beta, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + np.float32(eps)) * gamma + beta).astype(np.float32)


def gelu(x):
    return (0.5 * x * (1 + np.tanh(np.float32(np.sqrt(2 / np.pi)) * (x + 0.044715 * x ** 3)))).astype(np.float32)


def mla_attention(q_nope, q_rope, c_kv, k_rope, w_up, b_up, query_start):
    """Reference MLA attention with K/V materialized from the latents (decoder_mla.MLAAttention).

    q_nope [H,T,Dn], q_rope [H,T,Dr] (RoPE applied), c_kv [S,R], k_rope [S,Dr] (RoPE applied),
    w_up [H*(Dn+Dv), R] (nn.Linear layout), b_up [H*(Dn+Dv)] or None.
    Query t sits at absolute position query_start + t and sees keys 0..query_start + t.
    Returns the attention context [H,T,Dv].
    """
    H, T, Dn = q_nope.shape
    S, Dr = k_rope.shape
    up = c_kv @ w_up.T
    if b_up is not None:
        up = up + b_up
    up = up.reshape(S, H, -1).transpose(1, 0, 2)  # [H,S,Dn+Dv]
    k_nope, v = up[..., :Dn], up[..., Dn:]
    scores = (q_nope @ k_nope.transpose(0, 2, 1) + q_rope @ k_rope.T[None]) * np.float32((Dn + Dr) ** -0.5)
    future = np.arange(S)[None, :] > (query_start + np.arange(T))[:, None]
    scores = np.where(future[None], -np.inf, scores)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return (probs @ v).astype(np.float32)


# ---------------------------------------------------------------------------
# numeric backends
# ---------------------------------------------------------------------------

class _NumpyOps:
    """float32 numpy reference of every op (CPU)."""
    name = "numpy"

    def param(self, arr, matmul=False):
        return np.ascontiguousarray(arr, dtype=np.float32)

    tensor = param

    def numpy(self, x):
        return x

    def linear(self, x, w, b):
        y = x @ w
        return y if b is None else y + b

    def add(self, x, y):
        return x + y

    def gelu(self, x):
        return gelu(x)

    def layernorm(self, x, gamma, beta, eps):
        return layer_norm(x, gamma, beta, eps)

    def mla(self, q_nope, q_rope, c_kv, k_rope, w_up, b_up, query_start):
        return mla_attention(q_nope, q_rope, c_kv, k_rope, w_up, b_up, query_start)


class _MiniTorchOps:
    """MiniTorch CudaKernelOps: matmuls, fused LayerNorm and fused MLA attention kernels."""
    name = "minitorch"

    def __init__(self, backend, resident, ln_eps):
        from minitorch.cuda_kernel_ops import CudaKernelOps, make_resident
        from minitorch.nn import GELU
        from minitorch.tensor_functions import tensor_from_numpy

        from .common import MINITORCH_LN_EPS
        if ln_eps != MINITORCH_LN_EPS:
            raise ValueError(f"model needs LayerNorm eps={ln_eps}, but the fused kernel uses {MINITORCH_LN_EPS}")
        self.backend, self.resident = backend, resident
        self._kernels, self._make_resident = CudaKernelOps, make_resident
        self._gelu, self._from_numpy = GELU, tensor_from_numpy

    def param(self, arr, matmul=False):
        t = self.tensor(arr)
        if matmul and self.resident:
            self._make_resident(t)  # uploaded once; `x @ t` then skips the per-call copy
        return t

    def tensor(self, arr):
        return self._from_numpy(np.ascontiguousarray(arr, dtype=np.float32), backend=self.backend)

    def numpy(self, x):
        return x.to_numpy()

    def linear(self, x, w, b):
        y = x @ w
        return y if b is None else y + b

    def add(self, x, y):
        return x + y

    def gelu(self, x):
        return self._gelu(x)

    def layernorm(self, x, gamma, beta, eps):
        return x.contiguous().layernorm(gamma, beta)

    def mla(self, q_nope, q_rope, c_kv, k_rope, w_up, b_up, query_start):
        out = self._kernels.mla_fw(self.tensor(q_nope[None]), self.tensor(q_rope[None]),
                                   self.tensor(c_kv[None]), self.tensor(k_rope[None]),
                                   w_up, b_up, query_start=query_start)
        return out.to_numpy()[0]


# ---------------------------------------------------------------------------
# cache and model
# ---------------------------------------------------------------------------

class LatentCache:
    """Preallocated MLA latents per layer: c_kv [n_layer, max_len, R], k_rope [n_layer, max_len, Dr]."""

    def __init__(self, n_layer, rank, rope_dim, max_len):
        self.c_kv = np.zeros((n_layer, max_len, rank), dtype=np.float32)
        self.k_rope = np.zeros((n_layer, max_len, rope_dim), dtype=np.float32)
        self.max_len = max_len
        self.length = 0

    def crop(self, length):
        """Roll back to the first `length` positions (rejected drafts are dropped)."""
        assert 0 <= length <= self.length, (length, self.length)
        self.length = length

    def latents(self, layer, end=None):
        """Views of one layer's cached (c_kv, k_rope) for positions [0, end), default the whole cache."""
        end = self.length if end is None else end
        assert 0 <= end <= self.length, (end, self.length)
        return self.c_kv[layer, :end], self.k_rope[layer, :end]

    @property
    def floats_per_token(self):
        return self.c_kv.shape[0] * (self.c_kv.shape[2] + self.k_rope.shape[2])


class MiniTorchMLAGPT2:
    """MLA decoder inference with the same interface as model.MiniTorchGPT2 (new_cache / forward)."""

    def __init__(self, model_dir, backend=None, max_len=1024, resident=True):
        from safetensors.numpy import load_file

        with open(os.path.join(model_dir, "config.json")) as f:
            meta = json.load(f)
        if meta.get("format") != FORMAT:
            raise ValueError(f"{model_dir}: expected format {FORMAT!r}, got {meta.get('format')!r}")
        self.cfg = cfg = MLAConfig.from_dict(meta["config"])
        self.n_layer, self.n_head, self.n_embd = cfg.n_layer, cfg.n_head, cfg.n_embd
        self.vocab_size = cfg.n_vocab
        self.max_len = min(max_len, cfg.n_positions)
        self.resident = backend is not None and resident
        self.ops = _NumpyOps() if backend is None else _MiniTorchOps(backend, resident, cfg.ln_eps)
        self._load(load_file(os.path.join(model_dir, "model.safetensors")))

    def _load(self, w):
        cfg, ops = self.cfg, self.ops
        shapes = expected_shapes(cfg)
        if "lm_head.weight" in w:  # untied head
            shapes["lm_head.weight"] = (cfg.n_vocab, cfg.n_embd)
        missing, unexpected = shapes.keys() - w.keys(), w.keys() - shapes.keys()
        wrong = {k: (tuple(w[k].shape), shapes[k]) for k in shapes.keys() & w.keys()
                 if tuple(w[k].shape) != shapes[k]}
        if missing or unexpected or wrong:
            raise ValueError(f"weights do not match the config: missing {sorted(missing)}, "
                             f"unexpected {sorted(unexpected)}, wrong shapes {wrong}")

        def linear(name):  # nn.Linear [out, in] -> MiniTorch [in, out]
            bias = w.get(name + ".bias")
            return ops.param(w[name + ".weight"].T, matmul=True), None if bias is None else ops.param(bias)

        self.wte = np.ascontiguousarray(w["token_embeddings.weight"], dtype=np.float32)  # gathered on host
        self.wpe = np.ascontiguousarray(w["position_embeddings.weight"], dtype=np.float32)
        self.lm_head = ops.param(w.get("lm_head.weight", self.wte).T, matmul=True)  # [D, V]
        self.ln_f = (ops.param(w["ln.weight"]), ops.param(w["ln.bias"]))
        self.layers = []
        for i in range(cfg.n_layer):
            p, a = f"layers.{i}.", f"layers.{i}.attention."
            L = {"ln1": (ops.param(w[p + "ln_1.weight"]), ops.param(w[p + "ln_1.bias"])),
                 "ln2": (ops.param(w[p + "ln_2.weight"]), ops.param(w[p + "ln_2.bias"])),
                 "kv_down": linear(a + "kv_down"),
                 # the attention kernel takes kv_up in nn.Linear layout [H*(Dn+Dv), R]
                 "w_up": ops.param(w[a + "kv_up.weight"]),
                 "b_up": ops.param(w[a + "kv_up.bias"]) if cfg.bias else None,
                 "kv_norm": w[a + "kv_norm.weight"] if cfg.use_kv_norm else None,  # host
                 "out": linear(a + "out_projection"),
                 "fc": linear(p + "ff.linear_in"),
                 "proj": linear(p + "ff.linear_out")}
            if cfg.q_lora_rank is None:
                L["q"] = linear(a + "q_proj")
            else:
                L["q_down"], L["q_up"] = linear(a + "q_down"), linear(a + "q_up")
                L["q_norm"] = w[a + "q_norm.weight"]  # host
            self.layers.append(L)

    def new_cache(self):
        cfg = self.cfg
        return LatentCache(cfg.n_layer, cfg.kv_lora_rank, cfg.qk_rope_head_dim, self.max_len)

    def forward(self, ids, cache, last_only=False, hidden_layers: Sequence[int] = ()):
        """Run `ids` (token ids) after the `cache.length` cached positions; append their latents.

        Returns host logits of shape (len(ids), V), or (1, V) for the last position if
        `last_only`. If `hidden_layers` is non-empty, returns (logits, hidden) where hidden has
        shape (len(hidden_layers), len(ids), n_embd): the residual stream after each listed
        layer (0-indexed) at every fed position, regardless of `last_only`.
        """
        cfg, ops = self.cfg, self.ops
        H, Dn, Dr, Dv = cfg.n_head, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        R = cfg.kv_lora_rank
        T = len(ids)
        P = cache.length
        S = P + T
        assert T > 0 and S <= cache.max_len, (P, T, cache.max_len)
        hidden_layers = list(hidden_layers)
        if any(not 0 <= m < cfg.n_layer for m in hidden_layers):
            raise ValueError(f"hidden_layers must be in [0, {cfg.n_layer}): {hidden_layers}")
        taps = {}

        x = ops.tensor(self.wte[np.asarray(ids)] + self.wpe[P:S])  # [T, D]
        for li, L in enumerate(self.layers):
            h = ops.layernorm(x, *L["ln1"], cfg.ln_eps)
            if cfg.q_lora_rank is None:
                q = ops.linear(h, *L["q"])
            else:
                q = ops.linear(h, *L["q_down"])
                q = ops.linear(ops.tensor(rms_norm(ops.numpy(q), L["q_norm"])), *L["q_up"])
            q = ops.numpy(q).reshape(T, H, Dn + Dr).transpose(1, 0, 2)  # [H, T, Dn+Dr]
            q_nope = np.ascontiguousarray(q[..., :Dn])
            q_rope = apply_rope(np.ascontiguousarray(q[..., Dn:]), P)

            kv = ops.numpy(ops.linear(h, *L["kv_down"]))  # [T, R+Dr]
            c_kv = kv[:, :R] if L["kv_norm"] is None else rms_norm(kv[:, :R], L["kv_norm"])
            cache.c_kv[li, P:S] = c_kv
            cache.k_rope[li, P:S] = apply_rope(np.ascontiguousarray(kv[:, R:]), P)

            ctx = ops.mla(q_nope, q_rope, cache.c_kv[li, :S], cache.k_rope[li, :S],
                          L["w_up"], L["b_up"], P)  # [H, T, Dv]
            ctx = ctx.transpose(1, 0, 2).reshape(T, H * Dv)
            x = ops.add(x, ops.linear(ops.tensor(ctx), *L["out"]))

            h = ops.layernorm(x, *L["ln2"], cfg.ln_eps)
            x = ops.add(x, ops.linear(ops.gelu(ops.linear(h, *L["fc"])), *L["proj"]))
            if li in hidden_layers:
                taps[li] = np.array(ops.numpy(x), dtype=np.float32)

        cache.length = S
        if last_only:
            x = ops.tensor(ops.numpy(x)[-1:])
        x = ops.layernorm(x, *self.ln_f, cfg.ln_eps)
        logits = np.array(ops.numpy(ops.linear(x, self.lm_head, None)), dtype=np.float32)
        if hidden_layers:
            return logits, np.stack([taps[m] for m in hidden_layers])
        return logits
