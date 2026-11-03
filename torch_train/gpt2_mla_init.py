"""Initialize the educational MLA decoder from pretrained Hugging Face GPT-2.

GPT-2 K/V projections are compressed by truncated SVD; this is approximate,
not mathematically equivalent to GPT-2. Do not expect matching logits.
"""

import math
import torch
from decoder_mla import MLAConfig, DecoderLM


def make_gpt2_mla_config(hf_config, kv_rank=512, rope_dim=32, dropout=0.0):
    d = int(hf_config.n_embd)
    heads = int(hf_config.n_head)
    if d % heads:
        raise ValueError("GPT-2 embedding size must be divisible by heads")
    if not 0 < kv_rank <= d:
        raise ValueError("kv_rank must be between 1 and n_embd")
    return MLAConfig(
        n_vocab=int(hf_config.vocab_size), n_embd=d,
        n_head=heads, n_layer=int(hf_config.n_layer),
        n_positions=int(hf_config.n_positions),
        p_dropout=dropout, ln_eps=float(hf_config.layer_norm_epsilon),
        bias=True, middle_dim=4 * d,
        q_lora_rank=None, kv_lora_rank=kv_rank,
        use_kv_norm=False,  # preserve the linear SVD factorization
        qk_nope_head_dim=d // heads,
        qk_rope_head_dim=rope_dim,
        v_head_dim=d // heads,
    )


@torch.no_grad()
def initialize_from_gpt2(model: DecoderLM, hf_model, *, svd_device="cpu", svd_niter=2):
    """In-place initialization. Model and GPT-2 checkpoint should be on CPU.

    The GPT-2 Conv1D weights are stored as [input, output], unlike nn.Linear.
    A rank-r approximation of stacked per-head [K;V] matrices initializes KV.
    Q_nope is copied directly (scaled for the MLA attention temperature).
    Q_rope begins zero, K_rope retains small random weights so it can learn.
    """
    cfg = model.cfg
    if cfg.q_lora_rank is not None or cfg.use_kv_norm:
        raise ValueError("GPT-2 initialization requires q_lora_rank=None, use_kv_norm=False")
    dh = cfg.n_embd // cfg.n_head
    if cfg.qk_nope_head_dim != dh or cfg.v_head_dim != dh:
        raise ValueError("Use original GPT-2 per-head dimensions for Q/K_nope/V")
    if len(model.layers) != hf_model.config.n_layer:
        raise ValueError("Layer count mismatch")
    sd = hf_model.state_dict()

    def copy(dst, src):
        if dst.shape != src.shape:
            raise ValueError(f"Shape mismatch: {tuple(dst.shape)} vs {tuple(src.shape)}")
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    copy(model.token_embeddings.weight, sd['transformer.wte.weight'])
    copy(model.position_embeddings.weight, sd['transformer.wpe.weight'])
    copy(model.ln.weight, sd['transformer.ln_f.weight'])
    copy(model.ln.bias, sd['transformer.ln_f.bias'])
    # lm_head.weight is tied to token_embeddings.weight.

    report = []
    for i, block in enumerate(model.layers):
        p = f'transformer.h.{i}'
        copy(block.ln_1.weight, sd[f'{p}.ln_1.weight'])
        copy(block.ln_1.bias, sd[f'{p}.ln_1.bias'])
        copy(block.ln_2.weight, sd[f'{p}.ln_2.weight'])
        copy(block.ln_2.bias, sd[f'{p}.ln_2.bias'])
        for mod, source in ((block.ff.linear_in, 'c_fc'),
                            (block.ff.linear_out, 'c_proj')):
            copy(mod.weight, sd[f'{p}.mlp.{source}.weight'].T)
            copy(mod.bias, sd[f'{p}.mlp.{source}.bias'])

        attn = block.attention
        copy(attn.out_projection.weight, sd[f'{p}.attn.c_proj.weight'].T)
        copy(attn.out_projection.bias, sd[f'{p}.attn.c_proj.bias'])
        # GPT-2 c_attn weights are [D, 3D] and bias [3D].
        wq, wk, wv = sd[f'{p}.attn.c_attn.weight'].T.chunk(3, dim=0)
        bq, bk, bv = sd[f'{p}.attn.c_attn.bias'].chunk(3, dim=0)

        # Match original QK temperature: original 1/sqrt(dh), MLA uses
        # 1/sqrt(dh + rope_dim). Since Q_rope begins at zero, scale Q_nope.
        q_scale = math.sqrt((dh + cfg.qk_rope_head_dim) / dh)
        qw = attn.q_proj.weight.view(cfg.n_head, dh + cfg.qk_rope_head_dim, cfg.n_embd)
        qb = attn.q_proj.bias.view(cfg.n_head, dh + cfg.qk_rope_head_dim)
        qw.zero_()
        qb.zero_()
        qw[:, :dh].copy_(wq.reshape(cfg.n_head, dh, cfg.n_embd) * q_scale)
        qb[:, :dh].copy_(bq.reshape(cfg.n_head, dh) * q_scale)

        # Build per-head [K_h; V_h] rows matching kv_up view(H, dh + dh).
        kv_matrix = torch.cat(
            (wk.reshape(cfg.n_head, dh, cfg.n_embd),
             wv.reshape(cfg.n_head, dh, cfg.n_embd)), dim=1
        ).reshape(cfg.n_head * (2 * dh), cfg.n_embd)
        kv_bias = torch.cat(
            (bk.reshape(cfg.n_head, dh), bv.reshape(cfg.n_head, dh)), dim=1
        ).reshape(-1)

        r = cfg.kv_lora_rank
        matrix = kv_matrix.to(device=svd_device, dtype=torch.float32)
        # Low-rank randomized SVD; no giant full U matrix materialization.
        q = min(min(matrix.shape), r + 16)
        U, S, V = torch.svd_lowrank(matrix, q=q, niter=svd_niter)
        down = V[:, :r].T.contiguous()   # [r, input]
        up = U[:, :r] * S[:r]         # [output, r]
        # Error estimate for approximate K/V reconstruction.
        original_norm = matrix.square().sum().sqrt()
        err = ((matrix - up @ down).square().sum().sqrt() /
               original_norm.clamp_min(1e-12)).item()
        report.append(err)

        copy(attn.kv_down.weight[:r], down.cpu())
        attn.kv_down.bias[:r].zero_()
        # Retain random initialized rotary-key branch. No rotary effect until
        # Q_rope learns; avoids a permanently zero-gradient branch.
        copy(attn.kv_up.weight, up.cpu())
        copy(attn.kv_up.bias, kv_bias)

        print(f'layer {i + 1:02d}/{len(model.layers)}: K/V rank-{r} relative Frobenius error={err:.4f}', flush=True)
    return report
