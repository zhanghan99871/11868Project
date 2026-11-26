"""H-Spec-style parallel drafter for the MLA verifier, in PyTorch (used for training).

One forward pass drafts a whole block. The input is [anchor, MASK x k] at absolute positions
P..P+k, where the anchor is the last token the verifier produced (not yet in its cache); the
outputs at the k MASK positions are the drafted tokens for positions P+1..P+k. Like H-Spec
(arXiv 2609.24197) the drafter keeps no KV cache of its own. It is conditioned on the verifier in
two ways:

* In-place latent reuse. Drafter layer i attends to the cached MLA latents (c_kv, k_rope) of
  verifier layer cfg.target_layers[i] at positions < P, plus the block's own latents (causal).
  Because MLA shares a 512-d latent rather than per-head K/V, the drafter can be narrower than
  the verifier (768 vs 1024): its own kv_up maps latents to K/V for its heads. kv_up starts as a
  copy of that verifier layer's (init_from_verifier).
* Last-position summary. The verifier's residual stream at position P-1 after cfg.tap_layers is
  normalized, fused and projected, then added to the anchor's input embedding. (H-Spec uses it as
  Mamba's initial state; this drafter is a GPT-2-style transformer instead.)

The trunk is GPT-2-small shaped (pre-LN blocks, GELU MLP) and can start from distilgpt2
(init_from_gpt2). The output head is the verifier's tied LM head, frozen, after a learned
n_embd -> target_n_embd projection.

`forward` runs many blocks per sequence in one pass for training (anchors [B, N]; a block mask
keeps blocks from seeing each other or the prefix past their anchor); `draft_block` runs the
same code for one block (inference). project/minitorch_drafter.py ports draft_block to MiniTorch.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROPE_BASE = 10000.0  # decoder_mla.apply_rope
Latents = Tuple[Tensor, Tensor]  # (c_kv, k_rope), k_rope after RoPE, as in the verifier's cache


def default_layer_maps(n_target_layers: int, n_drafter_layers: int, n_taps: int = 5):
    """(target_layers, tap_layers) following H-Spec's choice, scaled to the verifier's depth.

    Taps: n_taps layers evenly spaced over 1..L-2 (H-Spec: 1, 9, 17, 25, 34 of 36 layers).
    Attention reads: evenly spaced over the upper half, L//2..L-2 (H-Spec reuses the deeper
    tapped layers). For the 24-layer verifier and 6 drafter layers: reads 12, 14, ..., 22 and
    taps 1, 6, 11, 16, 22.
    """
    L = n_target_layers
    lo = min(1, L - 1)
    hi = max(L - 2, lo)
    taps = tuple(int(i) for i in np.floor(np.linspace(lo, hi, n_taps)))
    reads = tuple(int(i) for i in np.floor(np.linspace(min(L // 2, hi), hi, n_drafter_layers)))
    return reads, taps


@dataclass
class DrafterConfig:
    # trunk (GPT-2 small / distilgpt2 shaped)
    n_vocab: int = 50257
    n_embd: int = 768
    n_layer: int = 6
    n_positions: int = 1024
    middle_dim: Optional[int] = None  # 4 * n_embd
    ln_eps: float = 1e-5
    bias: bool = True
    block_size: int = 7  # k: tokens drafted per forward pass
    # attention: must match the verifier's MLA layout to read its latents in place
    n_head: int = 16
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    kv_lora_rank: int = 512
    # verifier context
    target_n_embd: int = 1024
    target_layers: Tuple[int, ...] = (12, 14, 16, 18, 20, 22)  # whose latents drafter layer i reads
    tap_layers: Tuple[int, ...] = (1, 6, 11, 16, 22)  # whose hidden state at P-1 is fused
    fusion_rank: Optional[int] = None  # target_n_embd
    use_position_embeddings: bool = True
    summary_to_all_positions: bool = False  # also add the summary to every MASK position

    def __post_init__(self):
        self.target_layers = tuple(int(i) for i in self.target_layers)
        self.tap_layers = tuple(int(i) for i in self.tap_layers)
        if self.middle_dim is None:
            self.middle_dim = 4 * self.n_embd
        if self.fusion_rank is None:
            self.fusion_rank = self.target_n_embd
        for name in ("n_vocab", "n_embd", "n_layer", "n_positions", "middle_dim", "block_size",
                     "n_head", "qk_nope_head_dim", "v_head_dim", "kv_lora_rank", "target_n_embd",
                     "fusion_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be positive and even")
        if len(self.target_layers) != self.n_layer:
            raise ValueError(f"need one target layer per drafter layer: {len(self.target_layers)} != {self.n_layer}")
        if not self.tap_layers:
            raise ValueError("need at least one tap layer")
        if min(self.target_layers + self.tap_layers) < 0:
            raise ValueError("layer indices must be non-negative")

    @classmethod
    def for_verifier(cls, verifier_cfg, n_layer=6, n_embd=768, n_taps=5, **overrides):
        """Attention dims, vocabulary, positions and layer maps taken from a decoder_mla.MLAConfig."""
        reads, taps = default_layer_maps(verifier_cfg.n_layer, n_layer, n_taps)
        fields = dict(n_vocab=verifier_cfg.n_vocab, n_embd=n_embd, n_layer=n_layer,
                      n_positions=verifier_cfg.n_positions, ln_eps=verifier_cfg.ln_eps,
                      n_head=verifier_cfg.n_head, qk_nope_head_dim=verifier_cfg.qk_nope_head_dim,
                      qk_rope_head_dim=verifier_cfg.qk_rope_head_dim, v_head_dim=verifier_cfg.v_head_dim,
                      kv_lora_rank=verifier_cfg.kv_lora_rank, target_n_embd=verifier_cfg.n_embd,
                      target_layers=reads, tap_layers=taps)
        fields.update(overrides)
        return cls(**fields)

    def check_verifier(self, verifier_cfg):
        """Raise ValueError unless this drafter can read `verifier_cfg`'s cache and head."""
        pairs = {"n_vocab": "n_vocab", "target_n_embd": "n_embd", "n_head": "n_head",
                 "qk_nope_head_dim": "qk_nope_head_dim", "qk_rope_head_dim": "qk_rope_head_dim",
                 "v_head_dim": "v_head_dim", "kv_lora_rank": "kv_lora_rank"}
        bad = {mine: (getattr(self, mine), getattr(verifier_cfg, theirs))
               for mine, theirs in pairs.items() if getattr(self, mine) != getattr(verifier_cfg, theirs)}
        if bad:
            raise ValueError(f"drafter does not match the verifier (drafter, verifier): {bad}")
        if max(self.target_layers + self.tap_layers) >= verifier_cfg.n_layer:
            raise ValueError(f"layer index out of range for a {verifier_cfg.n_layer}-layer verifier")


def rope(x: Tensor, positions: Tensor, base: float = ROPE_BASE) -> Tensor:
    """Interleaved RoPE as decoder_mla.apply_rope, with a position per token.

    x is [B, H, Q, D] or [B, Q, D]; positions is [B, Q] (absolute).
    """
    d = x.size(-1)
    idx = torch.arange(0, d, 2, device=x.device, dtype=torch.float32)
    angles = positions.to(torch.float32)[..., None] * (base ** (-idx / d))  # [B, Q, D/2]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    if x.dim() == 4:
        cos, sin = cos[:, None], sin[:, None]
    pair = x.reshape(*x.shape[:-1], d // 2, 2)
    even, odd = pair[..., 0], pair[..., 1]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def block_attention_mask(prefix_len: Tensor, n_prefix_keys: int, block_len: int) -> Tensor:
    """Boolean mask [B, N*block_len, n_prefix_keys + N*block_len], True = may attend.

    Queries are N blocks of block_len tokens. Query j of block b sees the prefix keys
    s < prefix_len[:, b] and keys 0..j of its own block; never another block.
    """
    B, N = prefix_len.shape
    device = prefix_len.device
    Q = N * block_len
    s = torch.arange(n_prefix_keys, device=device)
    prefix = (s[None, None, :] < prefix_len[:, :, None]).repeat_interleave(block_len, dim=1)
    qb, qj = torch.arange(Q, device=device) // block_len, torch.arange(Q, device=device) % block_len
    own = (qb[:, None] == qb[None, :]) & (qj[None, :] <= qj[:, None])
    return torch.cat((prefix, own[None].expand(B, Q, Q)), dim=-1)


class LatentReaderAttention(nn.Module):
    """MLA attention over [verifier latents of the prefix ; the block's own latents].

    Same parameters as decoder_mla.MLAAttention (q_proj, kv_down, kv_up, out_projection) at the
    drafter's width. Scores use the verifier's scale (qk_nope + qk_rope) ** -0.5.
    """

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.cfg = cfg
        d, H, R = cfg.n_embd, cfg.n_head, cfg.kv_lora_rank
        Dn, Dr, Dv = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        self.q_proj = nn.Linear(d, H * (Dn + Dr), bias=cfg.bias)
        self.kv_down = nn.Linear(d, R + Dr, bias=cfg.bias)  # the block's own latents
        self.kv_up = nn.Linear(R, H * (Dn + Dv), bias=cfg.bias)
        self.out_projection = nn.Linear(H * Dv, d, bias=cfg.bias)

    def forward(self, x: Tensor, positions: Tensor, prefix_c: Tensor, prefix_kr: Tensor,
                mask: Tensor) -> Tensor:
        cfg = self.cfg
        H, R = cfg.n_head, cfg.kv_lora_rank
        Dn, Dr, Dv = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        B, Q, _ = x.shape
        S = prefix_c.size(1)

        q_nope, q_rope = self.q_proj(x).view(B, Q, H, Dn + Dr).transpose(1, 2).split([Dn, Dr], dim=-1)
        q = torch.cat((q_nope, rope(q_rope, positions)), dim=-1)

        c_blk, kr_blk = self.kv_down(x).split([R, Dr], dim=-1)
        c = torch.cat((prefix_c, c_blk), dim=1)
        kr = torch.cat((prefix_kr, rope(kr_blk, positions)), dim=1)
        k_nope, v = self.kv_up(c).view(B, S + Q, H, Dn + Dv).transpose(1, 2).split([Dn, Dv], dim=-1)
        k = torch.cat((k_nope, kr[:, None].expand(B, H, S + Q, Dr)), dim=-1)

        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        return self.out_projection(ctx.transpose(1, 2).reshape(B, Q, H * Dv))


class FeedForward(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.linear_in = nn.Linear(cfg.n_embd, cfg.middle_dim, bias=cfg.bias)
        self.linear_out = nn.Linear(cfg.middle_dim, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_out(F.gelu(self.linear_in(x), approximate="tanh"))


class DrafterLayer(nn.Module):
    """GPT-2 pre-LN block with LatentReaderAttention."""

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.ln_eps)
        self.attention = LatentReaderAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.ln_eps)
        self.ff = FeedForward(cfg)

    def forward(self, x, positions, prefix_c, prefix_kr, mask):
        x = x + self.attention(self.ln_1(x), positions, prefix_c, prefix_kr, mask)
        return x + self.ff(self.ln_2(x))


class SummaryFusion(nn.Module):
    """Verifier hidden states at the last input position (one per tap layer) -> one n_embd vector.

    Each tap is layer-normalized (the residual stream grows with depth), then H-Spec's cross-layer
    fusion Proj_r(concat(...)) is followed by a projection to the drafter width.
    """

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.target_n_embd, eps=cfg.ln_eps) for _ in cfg.tap_layers)
        self.down = nn.Linear(len(cfg.tap_layers) * cfg.target_n_embd, cfg.fusion_rank, bias=False)
        self.up = nn.Linear(cfg.fusion_rank, cfg.n_embd, bias=cfg.bias)

    def forward(self, hidden: Tensor) -> Tensor:  # [..., n_taps, target_n_embd] -> [..., n_embd]
        z = torch.cat([norm(hidden[..., i, :]) for i, norm in enumerate(self.norms)], dim=-1)
        return self.up(self.down(z))


class HSpecDrafter(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.n_embd
        self.token_embeddings = nn.Embedding(cfg.n_vocab, d)  # used for the anchor only
        self.position_embeddings = nn.Embedding(cfg.n_positions, d) if cfg.use_position_embeddings else None
        self.mask_embedding = nn.Parameter(torch.zeros(d))
        self.summary = SummaryFusion(cfg)
        self.layers = nn.ModuleList(DrafterLayer(cfg) for _ in range(cfg.n_layer))
        self.ln = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.head_proj = nn.Linear(d, cfg.target_n_embd, bias=False)
        # The verifier's LM head, frozen and not saved with the drafter: set_target_head().
        self.register_buffer("target_head", torch.zeros(cfg.n_vocab, cfg.target_n_embd), persistent=False)
        self.has_target_head = False

        self.apply(self._init_weights)
        nn.init.normal_(self.mask_embedding, std=0.02)
        for layer in self.layers:  # GPT-2: scale residual projections by 1/sqrt(2 * n_layer)
            for proj in (layer.attention.out_projection, layer.ff.linear_out):
                nn.init.normal_(proj.weight, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def set_target_head(self, weight: Tensor):
        """Use the verifier's LM head weight [n_vocab, target_n_embd] (its token embedding, tied)."""
        if tuple(weight.shape) != tuple(self.target_head.shape):
            raise ValueError(f"target head must be {tuple(self.target_head.shape)}, got {tuple(weight.shape)}")
        with torch.no_grad():
            self.target_head.copy_(weight.detach())
        self.has_target_head = True

    def forward(self, tokens: Tensor, anchors: Tensor, latents: Sequence[Latents], hidden: Tensor) -> Tensor:
        """Training forward: one block per anchor, all in one pass.

        tokens [B, T]; anchors [B, N] positions with 1 <= a < T; latents: one (c_kv [B, T, R],
        k_rope [B, T, Dr]) per drafter layer, from verifier layer target_layers[i] (target_context);
        hidden [B, T, n_taps, target_n_embd]. Block b sees latents < anchors[b] and hidden at
        anchors[b] - 1 only. Returns logits [B, N, block_size, n_vocab] for positions a+1..a+k.
        """
        B, T = tokens.shape
        if anchors.dim() != 2 or anchors.size(0) != B:
            raise ValueError("anchors must be [B, N]")
        if anchors.min() < 1 or anchors.max() >= T:
            raise ValueError(f"anchors must be in [1, {T - 1}]")
        rows = torch.arange(B, device=tokens.device)[:, None]
        return self._draft(tokens.gather(1, anchors), anchors, latents, hidden[rows, anchors - 1])

    def draft_block(self, anchor: int, latents: Sequence[Latents], summary: Tensor) -> Tensor:
        """Inference: draft one block. The anchor sits at position P = len(prefix latents).

        latents: one (c_kv [P, R], k_rope [P, Dr]) per drafter layer; summary [n_taps,
        target_n_embd] is the verifier's hidden state at P-1. Returns logits [block_size, n_vocab].
        """
        P = latents[0][0].size(0)
        if any(c.size(0) != P or kr.size(0) != P for c, kr in latents):
            raise ValueError("every layer's latents must cover the same prefix")
        device = self.mask_embedding.device
        anchors = torch.full((1, 1), P, dtype=torch.long, device=device)
        anchor_tok = torch.full((1, 1), int(anchor), dtype=torch.long, device=device)
        batched = [(c[None], kr[None]) for c, kr in latents]
        return self._draft(anchor_tok, anchors, batched, summary[None, None])[0, 0]

    def _draft(self, anchor_tok, anchor_pos, latents, summary):
        cfg = self.cfg
        if not self.has_target_head:
            raise RuntimeError("call set_target_head() (or init_from_verifier) before drafting")
        if len(latents) != cfg.n_layer:
            raise ValueError(f"need latents for {cfg.n_layer} layers, got {len(latents)}")
        k, (B, N), d = cfg.block_size, anchor_tok.shape, cfg.n_embd
        L = k + 1
        if anchor_pos.max() + k >= cfg.n_positions:
            raise ValueError(f"block would pass n_positions={cfg.n_positions}")
        positions = (anchor_pos[..., None] + torch.arange(L, device=anchor_pos.device)).reshape(B, N * L)

        s = self.summary(summary)  # [B, N, d]
        first = (self.token_embeddings(anchor_tok) + s)[:, :, None]
        rest = self.mask_embedding.expand(B, N, k, d)
        if cfg.summary_to_all_positions:
            rest = rest + s[:, :, None]
        x = torch.cat((first, rest), dim=2).reshape(B, N * L, d)
        if self.position_embeddings is not None:
            x = x + self.position_embeddings(positions)

        mask = block_attention_mask(anchor_pos, latents[0][0].size(1), L)
        for layer, (c, kr) in zip(self.layers, latents):
            x = layer(x, positions, c, kr, mask)
        x = x.view(B, N, L, d)[:, :, 1:]  # the MASK positions
        return self.head_proj(self.ln(x)) @ self.target_head.T


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------

def init_from_gpt2(drafter: HSpecDrafter, gpt2_state: dict, layers: Optional[Sequence[int]] = None):
    """Copy a GPT-2-family checkpoint (HF GPT2LMHeadModel/GPT2Model state_dict, e.g. distilgpt2)
    into the trunk: token/position embeddings, each block's LayerNorms and MLP, the final
    LayerNorm. Attention is MLA and keeps its fresh init (kv_up: see init_from_verifier).

    `layers`: source block for each drafter layer; default evenly spaced (distilgpt2 -> 0..5).
    Returns the source layers used.
    """
    cfg = drafter.cfg
    sd = {k.removeprefix("transformer."): v for k, v in gpt2_state.items()}
    n_src = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("h."))
    if layers is None:
        if cfg.n_layer > n_src:
            raise ValueError(f"source has {n_src} blocks, drafter needs {cfg.n_layer}; pass layers=")
        layers = [int(i) for i in np.floor(np.linspace(0, n_src - 1, cfg.n_layer))]
    if len(layers) != cfg.n_layer or not all(0 <= i < n_src for i in layers):
        raise ValueError(f"bad source layers {layers} for {n_src} source blocks")
    if tuple(sd["wte.weight"].shape) != (cfg.n_vocab, cfg.n_embd):
        raise ValueError(f"source embeddings {tuple(sd['wte.weight'].shape)} != ({cfg.n_vocab}, {cfg.n_embd})")
    if tuple(sd["h.0.mlp.c_fc.weight"].shape) != (cfg.n_embd, cfg.middle_dim):
        raise ValueError("source MLP width does not match middle_dim")

    def copy(dst, src):
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device))

    with torch.no_grad():
        copy(drafter.token_embeddings.weight, sd["wte.weight"])
        if drafter.position_embeddings is not None:
            if sd["wpe.weight"].size(0) < cfg.n_positions:
                raise ValueError("source has fewer positions than n_positions")
            copy(drafter.position_embeddings.weight, sd["wpe.weight"][:cfg.n_positions])
        for layer, i in zip(drafter.layers, layers):
            p = f"h.{i}."
            for ln in ("ln_1", "ln_2"):
                copy(getattr(layer, ln).weight, sd[p + ln + ".weight"])
                copy(getattr(layer, ln).bias, sd[p + ln + ".bias"])
            for name, src in (("linear_in", "c_fc"), ("linear_out", "c_proj")):  # Conv1D is [in, out]
                linear = getattr(layer.ff, name)
                copy(linear.weight, sd[p + f"mlp.{src}.weight"].T)
                if linear.bias is not None:
                    copy(linear.bias, sd[p + f"mlp.{src}.bias"])
        copy(drafter.ln.weight, sd["ln_f.weight"])
        copy(drafter.ln.bias, sd["ln_f.bias"])
    return list(layers)


def init_from_verifier(drafter: HSpecDrafter, verifier):
    """Start each drafter layer's kv_up as the verifier layer whose latents it reads (so those
    latents decode to that layer's own keys and values), and attach the verifier's LM head."""
    drafter.cfg.check_verifier(verifier.cfg)
    with torch.no_grad():
        for layer, m in zip(drafter.layers, drafter.cfg.target_layers):
            layer.attention.kv_up.load_state_dict(verifier.layers[m].attention.kv_up.state_dict())
    drafter.set_target_head(verifier.lm_head.weight)


# ---------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def target_context(verifier, tokens: Tensor, cfg: DrafterConfig):
    """One frozen verifier pass over tokens [B, T]: what the drafter reads, plus training targets.

    Returns (latents, hidden, logits): latents is one (c_kv [B, T, R], k_rope [B, T, Dr]) per
    drafter layer (verifier layer target_layers[i]); hidden [B, T, n_taps, n_embd_target] is the
    residual stream after each tap layer; logits [B, T, n_vocab] are the verifier's.
    """
    taps = {}
    hooks = [verifier.layers[m].register_forward_hook(
        lambda mod, inp, out, m=m: taps.__setitem__(m, out[0])) for m in set(cfg.tap_layers)]
    was_training = verifier.training
    verifier.eval()
    try:
        logits, cache = verifier(tokens, use_cache=True)
    finally:
        for h in hooks:
            h.remove()
        verifier.train(was_training)
    latents = [cache[m] for m in cfg.target_layers]
    hidden = torch.stack([taps[m] for m in cfg.tap_layers], dim=2)
    return latents, hidden, logits


def sample_anchors(batch: int, seq_len: int, block_size: int, n: int,
                   generator: Optional[torch.Generator] = None, device=None) -> Tensor:
    """n distinct anchors per row, uniform over [1, seq_len - 1 - block_size], sorted, [batch, n].

    Anchor a needs the prefix [0, a), a summary at a-1 and targets up to a + block_size.
    """
    lo, hi = 1, seq_len - 1 - block_size
    if hi < lo or n > hi - lo + 1:
        raise ValueError(f"cannot pick {n} anchors in [{lo}, {hi}] (seq_len {seq_len}, block {block_size})")
    order = torch.rand(batch, hi - lo + 1, generator=generator).argsort(dim=1)[:, :n]
    return (order.sort(dim=1).values + lo).to(device)


def block_targets(tokens: Tensor, teacher_logits: Tensor, anchors: Tensor, block_size: int):
    """Targets for each block's MASK positions: tokens [B, N, k] at a+1..a+k, and the verifier's
    logits for them [B, N, k, V], which it produced at positions a..a+k-1."""
    B, T = tokens.shape
    idx = anchors[..., None] + torch.arange(1, block_size + 1, device=anchors.device)  # [B, N, k]
    if idx.max() >= T or anchors.min() < 1:
        raise ValueError("every block needs its k targets inside the sequence")
    targets = tokens.gather(1, idx.flatten(1)).view(idx.shape)
    rows = torch.arange(B, device=tokens.device)[:, None, None]
    return targets, teacher_logits[rows, idx - 1]


def position_weights(block_size: int, gamma: float = 4.0) -> Tensor:
    """DFlash/DSpark/H-Spec block-position weights exp(-(j - 1) / gamma), j = 1..block_size."""
    return torch.exp(-torch.arange(block_size, dtype=torch.float32) / gamma)


def drafter_loss(logits: Tensor, targets: Tensor, teacher_logits: Tensor, gamma: float = 4.0,
                 ce_weight: float = 0.1, tv_weight: float = 0.9):
    """H-Spec's loss: ce_weight * CE(targets) + tv_weight * TV(verifier, drafter) per MASK position,
    averaged with normalized position weights. Returns (loss, stats).

    1 - TV(p, q) is the probability that rejection sampling (temperature 1) accepts a draft from q
    at that position, so stats["acceptance"][j] estimates per-position acceptance;
    stats["top1_agree"][j] is the greedy analogue.
    """
    k = logits.size(-2)
    logq = F.log_softmax(logits.float(), dim=-1)
    ce = -logq.gather(-1, targets[..., None]).squeeze(-1)  # [B, N, k]
    tv = 0.5 * (F.softmax(teacher_logits.float(), dim=-1) - logq.exp()).abs().sum(-1)
    w = position_weights(k, gamma).to(logits.device)
    w = w / w.sum()
    loss = ((ce_weight * ce + tv_weight * tv) * w).sum(-1).mean()
    with torch.no_grad():
        stats = {"ce": (ce * w).sum(-1).mean().item(), "tv": (tv * w).sum(-1).mean().item(),
                 "acceptance": (1 - tv).flatten(0, -2).mean(0).tolist(),
                 "top1_agree": (logits.argmax(-1) == teacher_logits.argmax(-1)).float()
                 .flatten(0, -2).mean(0).tolist()}
    return loss, stats
