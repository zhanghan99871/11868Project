"""Plain-numpy float64 GPT-2 forward (no cache), used only as a parity reference.

Written independently of model.py: tanh-approximate GELU as in GPT-2, LayerNorm
with the checkpoint's eps (1e-5), full causal attention over the whole sequence.
"""
import json
import os

import numpy as np
from safetensors.numpy import load_file


def gpt2_logits(model_dir, ids, eps=None):
    """eps=None uses the checkpoint value (1e-5); pass 1e-8 to match MiniTorch's fused LayerNorm."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    w = {k.removeprefix("transformer."): v.astype(np.float64)
         for k, v in load_file(os.path.join(model_dir, "model.safetensors")).items()}
    H = cfg["n_head"]
    eps = cfg["layer_norm_epsilon"] if eps is None else eps

    def ln(x, g, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * g + b

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    T = len(ids)
    x = w["wte.weight"][ids] + w["wpe.weight"][:T]
    D = x.shape[1]
    dh = D // H
    causal = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(cfg["n_layer"]):
        p = f"h.{i}."
        h = ln(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        qkv = h @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
        q, k, v = (qkv[:, j * D:(j + 1) * D].reshape(T, H, dh).transpose(1, 0, 2) for j in range(3))
        s = q @ k.transpose(0, 2, 1) / np.sqrt(dh) + causal
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(T, D)
        x = x + o @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        h = ln(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        x = x + gelu(h @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"]) @ w[p + "mlp.c_proj.weight"] \
            + w[p + "mlp.c_proj.bias"]
    x = ln(x, w["ln_f.weight"], w["ln_f.bias"])
    return x @ w["wte.weight"].T
