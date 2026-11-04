"""GPT-2 inference in MiniTorch with a per-request KV cache.

Weights come from a Hugging Face `model.safetensors` checkpoint. GPT-2's Conv1D
stores weights as (in, out), which is already MiniTorch's Linear layout, so no
transposes are needed except for the tied LM head.

Scope: batch size 1, inference only (no autograd history is recorded because no
tensor requires grad). Every matmul / elementwise op runs through MiniTorch's
CudaKernelOps backend; the KV cache itself lives in host numpy buffers because
MiniTorch tensors are host-resident and copied to the device per op anyway.
With resident=True the matmul weights are uploaded to the GPU once
(cuda_kernel_ops.make_resident); activations still round-trip per op.
"""
import json
import os

import numpy as np
from safetensors.numpy import load_file

import minitorch
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.nn import softmax, GELU

MASK_VALUE = -1e9


class KVCache:
    """Preallocated K/V buffers of shape [n_layer, n_head, max_len, head_dim]."""

    def __init__(self, n_layer, n_head, head_dim, max_len):
        self.k = np.zeros((n_layer, n_head, max_len, head_dim), dtype=np.float32)
        self.v = np.zeros((n_layer, n_head, max_len, head_dim), dtype=np.float32)
        self.max_len = max_len
        self.length = 0

    def crop(self, length):
        """Roll back to the first `length` positions (rejected drafts are dropped)."""
        assert 0 <= length <= self.length, (length, self.length)
        self.length = length


class MiniTorchGPT2:
    def __init__(self, model_dir, backend, max_len=1024, resident=True):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        self.backend = backend
        self.n_layer = cfg["n_layer"]
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.head_dim = self.n_embd // self.n_head
        self.vocab_size = cfg["vocab_size"]
        self.max_len = min(max_len, cfg["n_positions"])

        w = load_file(os.path.join(model_dir, "model.safetensors"))

        def get(name):
            return w[name] if name in w else w["transformer." + name]

        def T(arr):
            return tensor_from_numpy(np.ascontiguousarray(arr, dtype=np.float32), backend=backend)

        self.wte = np.ascontiguousarray(get("wte.weight"), dtype=np.float32)  # (V, D), gathered on host
        self.wpe = np.ascontiguousarray(get("wpe.weight"), dtype=np.float32)  # (P, D)
        self.lm_head = T(self.wte.T)                                         # (D, V), tied
        self.layers = []
        for i in range(self.n_layer):
            p = f"h.{i}."
            self.layers.append({
                "ln1_g": T(get(p + "ln_1.weight")), "ln1_b": T(get(p + "ln_1.bias")),
                "attn_w": T(get(p + "attn.c_attn.weight")), "attn_b": T(get(p + "attn.c_attn.bias")),
                "proj_w": T(get(p + "attn.c_proj.weight")), "proj_b": T(get(p + "attn.c_proj.bias")),
                "ln2_g": T(get(p + "ln_2.weight")), "ln2_b": T(get(p + "ln_2.bias")),
                "fc_w": T(get(p + "mlp.c_fc.weight")), "fc_b": T(get(p + "mlp.c_fc.bias")),
                "mlp_w": T(get(p + "mlp.c_proj.weight")), "mlp_b": T(get(p + "mlp.c_proj.bias")),
            })
        self.lnf_g = T(get("ln_f.weight"))
        self.lnf_b = T(get("ln_f.bias"))
        self.resident = resident
        if resident:
            # upload every matmul weight once instead of copying it host->device per call
            from minitorch.cuda_kernel_ops import make_resident
            make_resident(self.lm_head)
            for L in self.layers:
                for name in ["attn_w", "proj_w", "fc_w", "mlp_w"]:
                    make_resident(L[name])

    def new_cache(self):
        return KVCache(self.n_layer, self.n_head, self.head_dim, self.max_len)

    def _t(self, arr):
        return tensor_from_numpy(np.ascontiguousarray(arr, dtype=np.float32), backend=self.backend)

    def forward(self, ids, cache, last_only=False):
        """Run `ids` (list of token ids) after the `cache.length` cached positions.

        Appends the new K/V to `cache` and returns host logits of shape
        (len(ids), V), or (1, V) for the last position if `last_only`.
        """
        T_new = len(ids)
        P = cache.length
        S = P + T_new
        assert T_new > 0 and S <= cache.max_len, (P, T_new, cache.max_len)
        H, dh, D = self.n_head, self.head_dim, self.n_embd

        x = self._t(self.wte[np.asarray(ids)] + self.wpe[P:S])  # (T, D)

        mask = None
        if T_new > 1:
            # query i sits at absolute position P + i and may see keys 0..P+i
            m = np.triu(np.full((T_new, S), MASK_VALUE, dtype=np.float32), k=P + 1)
            mask = self._t(np.broadcast_to(m, (H, T_new, S)))
        scale = 1.0 / float(np.sqrt(dh))

        for li, L in enumerate(self.layers):
            h = x.contiguous().layernorm(L["ln1_g"], L["ln1_b"])
            qkv = (h @ L["attn_w"] + L["attn_b"]).to_numpy()                 # (T, 3D)
            q, k, v = (qkv[:, j * D:(j + 1) * D].reshape(T_new, H, dh).transpose(1, 0, 2)
                       for j in range(3))                                     # (H, T, dh)
            cache.k[li, :, P:S] = k
            cache.v[li, :, P:S] = v

            att = (self._t(q) @ self._t(cache.k[li, :, :S].transpose(0, 2, 1))) * scale  # (H, T, S)
            if mask is not None:
                att = att + mask
            att = softmax(att, dim=2)
            out = att @ self._t(cache.v[li, :, :S])                           # (H, T, dh)
            out = out.permute(1, 0, 2).contiguous().view(T_new, D)
            x = x + (out @ L["proj_w"] + L["proj_b"])

            h = x.contiguous().layernorm(L["ln2_g"], L["ln2_b"])
            x = x + (GELU(h @ L["fc_w"] + L["fc_b"]) @ L["mlp_w"] + L["mlp_b"])

        cache.length = S
        if last_only:
            x = self._t(x.to_numpy()[-1:])
        x = x.contiguous().layernorm(self.lnf_g, self.lnf_b)
        return (x @ self.lm_head).to_numpy().copy()
