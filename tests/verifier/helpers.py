"""Shared helpers for the MLA verifier tests: tiny PyTorch MLA models and their exports.

The PyTorch model (torch_train/decoder_mla.py) is the reference the MiniTorch verifier
(specdec/mla_model.py) must reproduce. Test modules call pytest.importorskip("torch") before
importing this module.
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "torch_train"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from decoder_mla import DecoderLM, MLAConfig  # noqa: E402
from export_mla_verifier import export_state  # noqa: E402
from specdec.mla_model import MiniTorchMLAGPT2  # noqa: E402

BASE = dict(n_vocab=97, n_embd=32, n_head=4, n_layer=3, n_positions=64, kv_lora_rank=12,
            qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8, p_dropout=0.0)
CONFIGS = {
    # the trained verifier's architecture at toy scale: full-rank Q, no kv_norm, biases
    # (what torch_train/gpt2_mla_init.py produces from GPT-2)
    "gpt2_style": dict(q_lora_rank=None, use_kv_norm=False, bias=True),
    # the other options decoder_mla supports
    "lora_q_kv_norm_no_bias": dict(q_lora_rank=16, use_kv_norm=True, bias=False),
}


def make_torch_model(name="gpt2_style", seed=0):
    torch.manual_seed(seed)
    model = DecoderLM(MLAConfig(**BASE, **CONFIGS[name])).eval()
    with torch.no_grad():  # norm weights start at 1 and biases at 0: perturb them so they matter
        for p in model.parameters():
            if p.dim() == 1:
                p.add_(0.1 * torch.randn_like(p))
    return model


def export(model, out_dir):
    return export_state(model.state_dict(), dataclasses.asdict(model.cfg), out_dir)


def token_ids(n, seed=0, vocab=BASE["n_vocab"]):
    return [int(t) for t in np.random.default_rng(seed).integers(0, vocab, n)]


def torch_logits(model, ids):
    with torch.no_grad():
        return model(torch.tensor([list(ids)]))[0].numpy()


def torch_cache_and_hidden(model, ids, layers=()):
    """Full-sequence PyTorch forward: ([(c_kv, k_rope)] per layer, {layer: residual stream after it})."""
    hidden = {}
    hooks = [model.layers[m].register_forward_hook(
        lambda mod, inp, out, m=m: hidden.__setitem__(m, out[0][0].numpy().copy())) for m in layers]
    try:
        with torch.no_grad():
            _, cache = model(torch.tensor([list(ids)]), use_cache=True)
    finally:
        for h in hooks:
            h.remove()
    return [(c[0].numpy(), k[0].numpy()) for c, k in cache], hidden


def torch_greedy(model, prompt, n_new):
    """Greedy decoding with the PyTorch model's own (c_kv, k_rope) cache."""
    out = []
    with torch.no_grad():
        logits, cache = model(torch.tensor([list(prompt)]), use_cache=True)
        for _ in range(n_new):
            tok = int(logits[0, -1].argmax())
            out.append(tok)
            logits, cache = model(torch.tensor([[tok]]), past_key_values=cache, use_cache=True)
    return out


def numpy_verifier(out_dir):
    return MiniTorchMLAGPT2(str(out_dir))
