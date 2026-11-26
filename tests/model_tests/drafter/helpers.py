"""Tiny MLA verifiers and H-Spec drafters for the drafter tests (test modules importorskip torch first)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "torch_train", ROOT / "project"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from decoder_mla import DecoderLM, MLAConfig  # noqa: E402
from hspec_drafter import DrafterConfig, HSpecDrafter, init_from_verifier  # noqa: E402

# a 4-layer MLA verifier with the trained verifier's layout (full-rank Q, no kv_norm, biases)
VERIFIER = dict(n_vocab=97, n_embd=32, n_head=4, n_layer=4, n_positions=64, kv_lora_rank=12,
                q_lora_rank=None, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
                use_kv_norm=False, p_dropout=0.0)
# the 24-layer GPT-2 medium MLA verifier (runs/mla-gpt2-medium/arguments.json)
TRAINED_VERIFIER = dict(n_vocab=50257, n_embd=1024, n_head=16, n_layer=24, n_positions=1024,
                        kv_lora_rank=512, q_lora_rank=None, qk_nope_head_dim=64, qk_rope_head_dim=32,
                        v_head_dim=64, use_kv_norm=False)
DRAFTER_VARIANTS = {
    "default": {},
    "summary_to_all": dict(summary_to_all_positions=True),
    "no_bias_no_wpe": dict(bias=False, use_position_embeddings=False),
}


def make_verifier(seed=0):
    torch.manual_seed(seed)
    return DecoderLM(MLAConfig(**VERIFIER)).eval()


def drafter_config(verifier_cfg, **overrides):
    """A 2-layer, 24-wide drafter reading 3 taps, drafting blocks of 4."""
    return DrafterConfig.for_verifier(verifier_cfg, **{**dict(n_layer=2, n_embd=24, n_taps=3, block_size=4),
                                                      **overrides})


def make_drafter(verifier, seed=1, **overrides):
    """A tiny drafter for `verifier`, with its norms and biases perturbed so they matter."""
    torch.manual_seed(seed)
    drafter = HSpecDrafter(drafter_config(verifier.cfg, **overrides))
    if drafter.cfg.bias:
        init_from_verifier(drafter, verifier)
    else:  # kv_up has no bias, so it cannot copy the verifier's
        drafter.set_target_head(verifier.lm_head.weight)
    with torch.no_grad():
        for p in drafter.parameters():
            if p.dim() == 1:
                p.add_(0.1 * torch.randn_like(p))
    return drafter.eval()


def token_ids(batch, length, seed=0, vocab=VERIFIER["n_vocab"]):
    return torch.randint(0, vocab, (batch, length), generator=torch.Generator().manual_seed(seed))
