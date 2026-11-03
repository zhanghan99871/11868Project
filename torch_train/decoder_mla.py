"""GPT-style decoder-only Transformer with Multi-head Latent Attention (MLA).

MLA architecture is inspired by DeepSeek's compressed KV representation and
partially rotary-positioned query/key dimensions. This is an educational
PyTorch implementation, not a drop-in replica of DeepSeek-V2/V3 or GPT-2.

The cache stores compressed c_KV [B, T, kv_lora_rank] and shared rotary keys
[B, T, qk_rope_head_dim] per layer. Full per-head K and V are reconstructed
for attention. This reduces cache storage but is not compute-optimal.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

KVCache = Tuple[Tensor, Tensor]  # (compressed_kv, rotated_shared_key)


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
    q_lora_rank: Optional[int] = None  # None -> full Q projection
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32  # even, partial RoPE
    v_head_dim: int = 64
    use_kv_norm: bool = True

    def __post_init__(self):
        for name in ("n_vocab", "n_embd", "n_head", "n_layer", "n_positions",
                     "kv_lora_rank", "qk_nope_head_dim", "v_head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.q_lora_rank is not None and self.q_lora_rank <= 0:
            raise ValueError("q_lora_rank must be positive when supplied")
        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be positive and even")
        if not 0.0 <= self.p_dropout < 1.0:
            raise ValueError("p_dropout must be in [0, 1)")
        if self.ln_eps <= 0:
            raise ValueError("ln_eps must be positive")
        if self.middle_dim is None:
            self.middle_dim = 4 * self.n_embd


class RMSNorm(nn.Module):
    """RMS normalization for the compressed latent query/key vectors."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_float = x.float()
        inv_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inv_rms).to(x.dtype) * self.weight


def apply_rope(x: Tensor, positions: Tensor, base: float = 10000.0) -> Tensor:
    """Apply interleaved rotary embeddings to [..., T, rotary_dim].

    Positions are absolute token positions [T]. Works for x [B,H,T,D]
    or [B,1,T,D].
    """
    dim = x.size(-1)
    if dim % 2:
        raise ValueError("RoPE dimension must be even")
    idx = torch.arange(0, dim, 2, device=x.device, dtype=torch.float32)
    inverse_freq = base ** (-idx / dim)
    angles = positions.to(torch.float32)[:, None] * inverse_freq[None, :]
    cos = angles.cos().to(x.dtype)[None, None, :, :]
    sin = angles.sin().to(x.dtype)[None, None, :, :]
    pair = x.reshape(*x.shape[:-1], dim // 2, 2)
    even, odd = pair[..., 0], pair[..., 1]
    out = torch.stack((even * cos - odd * sin,
                       even * sin + odd * cos), dim=-1)
    return out.flatten(-2)


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (MLA) with decoupled partial RoPE.

    Queries consist of per-head non-RoPE and RoPE dimensions.
    Keys/values are expanded from a shared low-rank compressed c_KV.
    One additional shared rotary key is produced per token.

    Cache stores (c_KV, rotated k_R), NOT expanded per-head K/V.
    """

    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.d_nope = cfg.qk_nope_head_dim
        self.d_rope = cfg.qk_rope_head_dim
        self.d_value = cfg.v_head_dim
        self.d_qk = self.d_nope + self.d_rope

        q_out = cfg.n_head * self.d_qk
        if cfg.q_lora_rank is None:
            self.q_proj = nn.Linear(cfg.n_embd, q_out, bias=cfg.bias)
            self.q_down = None
            self.q_norm = None
            self.q_up = None
        else:
            self.q_proj = None
            self.q_down = nn.Linear(cfg.n_embd, cfg.q_lora_rank, bias=cfg.bias)
            self.q_norm = RMSNorm(cfg.q_lora_rank)
            self.q_up = nn.Linear(cfg.q_lora_rank, q_out, bias=cfg.bias)

        # KV down-projection produces compressed latent plus shared rotary key.
        self.kv_down = nn.Linear(
            cfg.n_embd, cfg.kv_lora_rank + self.d_rope, bias=cfg.bias
        )
        self.kv_norm = (
            RMSNorm(cfg.kv_lora_rank)
            if cfg.use_kv_norm
            else nn.Identity()
        )
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_head * (self.d_nope + self.d_value),
            bias=cfg.bias,
        )
        self.out_projection = nn.Linear(
            cfg.n_head * self.d_value, cfg.n_embd, bias=cfg.bias
        )
        self.attn_dropout = nn.Dropout(cfg.p_dropout)
        self.out_dropout = nn.Dropout(cfg.p_dropout)
        self.scale = self.d_qk ** -0.5

    def forward(
        self,
        x: Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        batch, query_len, _ = x.shape
        past_len = 0 if past_key_value is None else past_key_value[0].size(1)
        total_len = past_len + query_len
        if total_len > self.cfg.n_positions:
            raise ValueError("Sequence length exceeds n_positions")

        positions = torch.arange(past_len, total_len, device=x.device)

        if self.q_proj is not None:
            q = self.q_proj(x)
        else:
            q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.view(batch, query_len, self.n_head, self.d_qk).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.d_nope, self.d_rope], dim=-1)
        q_rope = apply_rope(q_rope, positions)

        kv_down = self.kv_down(x)
        c_kv, k_rope = torch.split(
            kv_down, [self.cfg.kv_lora_rank, self.d_rope], dim=-1
        )
        c_kv = self.kv_norm(c_kv)
        # A *shared* rotary key, independent of head.
        k_rope = apply_rope(k_rope[:, None, :, :], positions).squeeze(1)

        if past_key_value is not None:
            old_c_kv, old_k_rope = past_key_value
            if old_c_kv.size(0) != batch or old_k_rope.size(0) != batch:
                raise ValueError("Cache batch size mismatch")
            c_kv = torch.cat((old_c_kv, c_kv), dim=1)
            k_rope = torch.cat((old_k_rope, k_rope), dim=1)

        present = (c_kv, k_rope) if use_cache else None

        # Materialize per-head K_nope and V from compressed c_KV.
        expanded = self.kv_up(c_kv).view(
            batch, total_len, self.n_head, self.d_nope + self.d_value
        ).transpose(1, 2)
        k_nope, v = torch.split(
            expanded, [self.d_nope, self.d_value], dim=-1
        )
        k_rope_heads = k_rope[:, None, :, :].expand(
            batch, self.n_head, total_len, self.d_rope
        )

        keys = torch.cat((k_nope, k_rope_heads), dim=-1)
        queries = torch.cat((q_nope, q_rope), dim=-1)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        # Absolute-position causal mask also works with cached decoding.
        key_positions = torch.arange(total_len, device=x.device)
        future = key_positions[None, :] > positions[:, None]
        scores = scores.masked_fill(future[None, None, :, :], float("-inf"))
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        probs = self.attn_dropout(probs)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(
            batch, query_len, self.n_head * self.d_value
        )
        return self.out_dropout(self.out_projection(context)), present


class FeedForward(nn.Module):
    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.linear_in = nn.Linear(cfg.n_embd, cfg.middle_dim, bias=cfg.bias)
        self.linear_out = nn.Linear(cfg.middle_dim, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.linear_out(F.gelu(self.linear_in(x), approximate="tanh")))


class TransformerLayer(nn.Module):
    """GPT-2-like pre-LN block, replacing MHA with MLA."""

    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.ln_eps)
        self.attention = MLAAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.ln_eps)
        self.ff = FeedForward(cfg)

    def forward(
        self, x: Tensor, past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        attn, present = self.attention(
            self.ln_1(x), past_key_value=past_key_value, use_cache=use_cache
        )
        x = x + attn
        x = x + self.ff(self.ln_2(x))
        return x, present


class DecoderLM(nn.Module):
    """Decoder-only language model with MLA and optionally compressed KV caching.

    Default return is logits [B,T,vocab]. If use_cache=True, returns
    (logits, list_of_per_layer_kv_caches).
    """

    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embeddings = nn.Embedding(cfg.n_vocab, cfg.n_embd)
        self.position_embeddings = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.p_dropout)
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(cfg.n_embd, eps=cfg.ln_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.n_vocab, bias=False)
        # GPT-like tied embedding / output classifier weights.
        self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        idx: Tensor,
        past_key_values: Optional[List[KVCache]] = None,
        use_cache: bool = False,
    ):
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch, sequence]")
        batch, seq_len = idx.shape
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("Expected one past KV cache per layer")
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(1)
        if past_len + seq_len > self.cfg.n_positions:
            raise ValueError("Sequence length exceeds n_positions")
        positions = torch.arange(past_len, past_len + seq_len, device=idx.device)
        x = self.dropout(
            self.token_embeddings(idx) + self.position_embeddings(positions)[None]
        )
        new_cache: List[KVCache] = []
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[i]
            x, present = layer(x, past_key_value=past, use_cache=use_cache)
            if use_cache:
                assert present is not None
                new_cache.append(present)
        logits = self.lm_head(self.ln(x))
        return (logits, new_cache) if use_cache else logits


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = MLAConfig(n_vocab=1000, n_embd=128, n_head=4, n_layer=2,
                    n_positions=128, kv_lora_rank=32, q_lora_rank=64,
                    qk_nope_head_dim=24, qk_rope_head_dim=8,
                    v_head_dim=32, p_dropout=0.0)
    model = DecoderLM(cfg).eval()
    ids = torch.randint(0, cfg.n_vocab, (2, 9))
    with torch.no_grad():
        full = model(ids)
        cache = None
        steps = []
        for j in range(ids.shape[1]):
            logits, cache = model(ids[:, j:j+1], past_key_values=cache, use_cache=True)
            steps.append(logits)
        cached = torch.cat(steps, dim=1)
    print("Output shape:", tuple(full.shape))
    print("Full vs cached max absolute error:", (full - cached).abs().max().item())
    assert torch.allclose(full, cached, atol=1e-4, rtol=1e-4)
    loss = model(ids).float().square().mean()
    loss.backward()
    print("Backward gradients available:", model.layers[0].attention.kv_down.weight.grad is not None)
