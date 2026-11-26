"""H-Spec-style drafter inference in MiniTorch: a port of HSpecDrafter.draft_block
(torch_train/hspec_drafter.py), which is where the drafter is defined and trained.

One call drafts a block of k tokens. The input is [anchor, MASK x k] at positions P..P+k
(P = number of tokens in the verifier's cache). Drafter layer i attends to the verifier's cached
MLA latents of layer target_layers[i] at positions < P, plus the block's own latents (causal),
through the fused MLA attention kernel with query_start = P: its causal mask is exactly the
drafter's mask (the whole prefix, causal within the block). The drafter keeps no cache.

Weights come from torch_train/export_hspec_drafter.py; the output head is the verifier's LM head
(MiniTorchMLAGPT2.head_weight). Backends as in specdec/mla_model.py: a MiniTorch
TensorBackend(CudaKernelOps), or backend=None for the float32 numpy reference (CPU).
"""
import json
import os
import sys
from dataclasses import dataclass, fields
from typing import Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from specdec.mla_model import apply_rope, layer_norm, make_ops  # noqa: E402

FORMAT = "hspec-drafter-minitorch-v1"  # torch_train/export_hspec_drafter.py


@dataclass(frozen=True)
class DrafterConfig:
    """The fields of torch_train/hspec_drafter.DrafterConfig (see there for their meaning)."""
    n_vocab: int
    n_embd: int
    n_layer: int
    n_positions: int
    middle_dim: int
    ln_eps: float
    bias: bool
    block_size: int
    n_head: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int
    target_n_embd: int
    target_layers: Tuple[int, ...]
    tap_layers: Tuple[int, ...]
    fusion_rank: int
    use_position_embeddings: bool
    summary_to_all_positions: bool

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        missing, unknown = names - set(d), set(d) - names
        if missing or unknown:
            raise ValueError(f"bad drafter config: missing {sorted(missing)}, unknown {sorted(unknown)}")
        d = dict(d, target_layers=tuple(d["target_layers"]), tap_layers=tuple(d["tap_layers"]))
        if len(d["target_layers"]) != d["n_layer"]:
            raise ValueError("need one target layer per drafter layer")
        return cls(**d)


def expected_shapes(cfg: DrafterConfig) -> dict:
    """PyTorch state_dict names and shapes of hspec_drafter.HSpecDrafter (target head excluded)."""
    d, Dt, H, R = cfg.n_embd, cfg.target_n_embd, cfg.n_head, cfg.kv_lora_rank
    Dn, Dr, Dv = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    shapes = {"token_embeddings.weight": (cfg.n_vocab, d), "mask_embedding": (d,),
              "summary.down.weight": (cfg.fusion_rank, len(cfg.tap_layers) * Dt),
              "summary.up.weight": (d, cfg.fusion_rank),
              "ln.weight": (d,), "ln.bias": (d,), "head_proj.weight": (Dt, d)}
    if cfg.use_position_embeddings:
        shapes["position_embeddings.weight"] = (cfg.n_positions, d)
    if cfg.bias:
        shapes["summary.up.bias"] = (d,)
    for i in range(len(cfg.tap_layers)):
        shapes[f"summary.norms.{i}.weight"] = shapes[f"summary.norms.{i}.bias"] = (Dt,)

    def linear(name, n_in, n_out):
        shapes[name + ".weight"] = (n_out, n_in)
        if cfg.bias:
            shapes[name + ".bias"] = (n_out,)

    for i in range(cfg.n_layer):
        p = f"layers.{i}."
        for ln in ("ln_1", "ln_2"):
            shapes[p + ln + ".weight"] = shapes[p + ln + ".bias"] = (d,)
        linear(p + "attention.q_proj", d, H * (Dn + Dr))
        linear(p + "attention.kv_down", d, R + Dr)
        linear(p + "attention.kv_up", R, H * (Dn + Dv))
        linear(p + "attention.out_projection", H * Dv, d)
        linear(p + "ff.linear_in", d, cfg.middle_dim)
        linear(p + "ff.linear_out", cfg.middle_dim, d)
    return shapes


class MiniTorchHSpecDrafter:
    def __init__(self, model_dir, target_head: np.ndarray, backend=None, resident=True):
        """target_head: the verifier's LM head weight [n_vocab, target_n_embd]."""
        from safetensors.numpy import load_file

        with open(os.path.join(model_dir, "config.json")) as f:
            meta = json.load(f)
        if meta.get("format") != FORMAT:
            raise ValueError(f"{model_dir}: expected format {FORMAT!r}, got {meta.get('format')!r}")
        self.cfg = cfg = DrafterConfig.from_dict(meta["config"])
        if tuple(target_head.shape) != (cfg.n_vocab, cfg.target_n_embd):
            raise ValueError(f"target head must be {(cfg.n_vocab, cfg.target_n_embd)}, got {tuple(target_head.shape)}")
        self.resident = backend is not None and resident
        self.ops = make_ops(backend, resident, cfg.ln_eps)
        self._load(load_file(os.path.join(model_dir, "model.safetensors")), target_head)

    @classmethod
    def from_verifier(cls, model_dir, verifier, backend=None, resident=True):
        """Drafter for a specdec.mla_model.MiniTorchMLAGPT2 verifier (checks they fit together)."""
        drafter = cls(model_dir, verifier.head_weight, backend=backend, resident=resident)
        cfg, vcfg = drafter.cfg, verifier.cfg
        pairs = {"target_n_embd": vcfg.n_embd, "n_head": vcfg.n_head, "kv_lora_rank": vcfg.kv_lora_rank,
                 "qk_nope_head_dim": vcfg.qk_nope_head_dim, "qk_rope_head_dim": vcfg.qk_rope_head_dim,
                 "v_head_dim": vcfg.v_head_dim}
        bad = {k: (getattr(cfg, k), v) for k, v in pairs.items() if getattr(cfg, k) != v}
        if bad or max(cfg.target_layers + cfg.tap_layers) >= vcfg.n_layer:
            raise ValueError(f"drafter does not fit the verifier: {bad or 'layer index out of range'}")
        return drafter

    def _load(self, w, target_head):
        cfg, ops = self.cfg, self.ops
        shapes = expected_shapes(cfg)
        missing, unexpected = shapes.keys() - w.keys(), w.keys() - shapes.keys()
        wrong = {k: (tuple(w[k].shape), shapes[k]) for k in shapes.keys() & w.keys()
                 if tuple(w[k].shape) != shapes[k]}
        if missing or unexpected or wrong:
            raise ValueError(f"weights do not match the config: missing {sorted(missing)}, "
                             f"unexpected {sorted(unexpected)}, wrong shapes {wrong}")

        def linear(name):  # nn.Linear [out, in] -> MiniTorch [in, out]
            bias = w.get(name + ".bias")
            return ops.param(w[name + ".weight"].T, matmul=True), None if bias is None else ops.param(bias)

        host = lambda name: np.ascontiguousarray(w[name], dtype=np.float32)  # noqa: E731
        self.wte = host("token_embeddings.weight")  # anchor lookup on the host
        self.wpe = host("position_embeddings.weight") if cfg.use_position_embeddings else None
        self.mask_embedding = host("mask_embedding")
        self.tap_norms = [(host(f"summary.norms.{i}.weight"), host(f"summary.norms.{i}.bias"))
                          for i in range(len(cfg.tap_layers))]  # per-tap LayerNorm, on the host
        self.summary_down = linear("summary.down")
        self.summary_up = linear("summary.up")
        self.layers = []
        for i in range(cfg.n_layer):
            p, a = f"layers.{i}.", f"layers.{i}.attention."
            self.layers.append({
                "ln1": (ops.param(w[p + "ln_1.weight"]), ops.param(w[p + "ln_1.bias"])),
                "ln2": (ops.param(w[p + "ln_2.weight"]), ops.param(w[p + "ln_2.bias"])),
                "q": linear(a + "q_proj"), "kv_down": linear(a + "kv_down"),
                "w_up": ops.param(w[a + "kv_up.weight"]),  # kernel layout [H*(Dn+Dv), R]
                "b_up": ops.param(w[a + "kv_up.bias"]) if cfg.bias else None,
                "out": linear(a + "out_projection"),
                "fc": linear(p + "ff.linear_in"), "proj": linear(p + "ff.linear_out")})
        self.ln_f = (ops.param(w["ln.weight"]), ops.param(w["ln.bias"]))
        self.head_proj = linear("head_proj")
        self.head = ops.param(np.asarray(target_head, dtype=np.float32).T, matmul=True)  # [Dt, V]

    def draft_logits(self, anchor: int, prefix_latents: Sequence[Tuple[np.ndarray, np.ndarray]],
                     summary: np.ndarray) -> np.ndarray:
        """Logits [block_size, n_vocab] for positions P+1..P+k.

        prefix_latents: one (c_kv [P, R], k_rope [P, Dr]) per drafter layer, from verifier layer
        target_layers[i]; summary [n_taps, target_n_embd]: the verifier's hidden state at P-1
        after each tap layer.
        """
        cfg, ops = self.cfg, self.ops
        H, R = cfg.n_head, cfg.kv_lora_rank
        Dn, Dr, Dv = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        k = cfg.block_size
        L = k + 1
        if len(prefix_latents) != cfg.n_layer:
            raise ValueError(f"need latents for {cfg.n_layer} layers, got {len(prefix_latents)}")
        P = prefix_latents[0][0].shape[0]
        if any(c.shape[0] != P or kr.shape[0] != P for c, kr in prefix_latents):
            raise ValueError("every layer's latents must cover the same prefix")
        if P + k >= cfg.n_positions:
            raise ValueError(f"block would pass n_positions={cfg.n_positions}")
        summary = np.asarray(summary, dtype=np.float32)
        if summary.shape != (len(cfg.tap_layers), cfg.target_n_embd):
            raise ValueError(f"summary must be {(len(cfg.tap_layers), cfg.target_n_embd)}, got {summary.shape}")

        z = np.concatenate([layer_norm(summary[i], g, b, cfg.ln_eps) for i, (g, b) in enumerate(self.tap_norms)])
        s = ops.numpy(ops.linear(ops.linear(ops.tensor(z[None]), *self.summary_down), *self.summary_up))[0]
        x = np.empty((L, cfg.n_embd), dtype=np.float32)
        x[0] = self.wte[int(anchor)] + s
        x[1:] = self.mask_embedding + (s if cfg.summary_to_all_positions else 0)
        if self.wpe is not None:
            x += self.wpe[P:P + L]
        x = ops.tensor(x)

        for Lw, (prefix_c, prefix_kr) in zip(self.layers, prefix_latents):
            h = ops.layernorm(x, *Lw["ln1"], cfg.ln_eps)
            q = ops.numpy(ops.linear(h, *Lw["q"])).reshape(L, H, Dn + Dr).transpose(1, 0, 2)
            q_nope = np.ascontiguousarray(q[..., :Dn])
            q_rope = apply_rope(np.ascontiguousarray(q[..., Dn:]), P)
            kv = ops.numpy(ops.linear(h, *Lw["kv_down"]))  # the block's own latents
            c_all = np.concatenate([prefix_c, kv[:, :R]]).astype(np.float32)
            kr_all = np.concatenate([prefix_kr, apply_rope(np.ascontiguousarray(kv[:, R:]), P)]).astype(np.float32)
            ctx = ops.mla(q_nope, q_rope, c_all, kr_all, Lw["w_up"], Lw["b_up"], P)  # [H, L, Dv]
            x = ops.add(x, ops.linear(ops.tensor(ctx.transpose(1, 0, 2).reshape(L, H * Dv)), *Lw["out"]))
            h = ops.layernorm(x, *Lw["ln2"], cfg.ln_eps)
            x = ops.add(x, ops.linear(ops.gelu(ops.linear(h, *Lw["fc"])), *Lw["proj"]))

        x = ops.tensor(ops.numpy(x)[1:])  # the MASK positions
        x = ops.linear(ops.layernorm(x, *self.ln_f, cfg.ln_eps), *self.head_proj)
        return np.array(ops.numpy(ops.linear(x, self.head, None)), dtype=np.float32)

    def draft_from_cache(self, anchor: int, cache, summary: np.ndarray) -> np.ndarray:
        """draft_logits for an anchor at position cache.length, reading a specdec LatentCache."""
        return self.draft_logits(anchor, [cache.latents(m) for m in self.cfg.target_layers], summary)

    def greedy(self, anchor: int, cache, summary: np.ndarray):
        """Greedy block draft: (k token ids, logits [k, n_vocab])."""
        logits = self.draft_from_cache(anchor, cache, summary)
        return [int(t) for t in logits.argmax(-1)], logits
