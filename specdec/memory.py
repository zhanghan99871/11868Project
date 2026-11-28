"""Memory of the verifier and its drafters: parameters and per-token caches, from the models'
config.json files (exact counts from the tensor shapes; no weights are loaded, no GPU needed).

  python -m specdec.memory --verifier-dir runs/mla-gpt2-medium/minitorch \
      --draft-dir runs/distil-drafter-gen-s32/best --hspec-drafter runs/hspec-drafter-gen-s32/minitorch \
      --out results/memory.json

What each setup keeps for one sequence of T cached tokens (MiniTorch stores everything in float32):

  verifier alone       verifier weights + MLA latent cache (c_kv + shared k_rope per layer) x T
  + distilgpt2         + its weights + its own K/V cache (2 x n_embd per layer) x T
  + H-Spec drafter     + its weights; it reads the verifier's latent cache and LM head in place and
                       keeps no cache (its block's latents, (k + 1) x layers x (rank + rope) floats,
                       live only during a pass)

The verifier's MHA counterpart (GPT-2 medium before the MLA conversion) would cache 2 x n_embd per
layer per token; it is reported for reference. Batching multiplies the caches, not the weights.
specdec.bench's `memory` config measures what the loaded MiniTorch models actually hold.
"""
import argparse
import json
import math
import os
import sys
from dataclasses import asdict, is_dataclass

from .mla_model import MLAConfig
from .mla_model import expected_shapes as verifier_shapes

BYTES_PER_FLOAT = 4  # MiniTorch keeps weights, activations and caches in float32
PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project")


def _drafter_module():
    if PROJECT not in sys.path:
        sys.path.insert(0, PROJECT)
    import minitorch_drafter
    return minitorch_drafter


def _as_dict(cfg):
    return asdict(cfg) if is_dataclass(cfg) else dict(cfg)


def n_params(shapes):
    return sum(math.prod(s) for s in shapes.values())


def mla_verifier(cfg):
    """The MLA verifier (specdec.mla_model; LM head tied to the token embeddings)."""
    c = MLAConfig.from_dict(_as_dict(cfg))
    return {"kind": "mla-verifier", "params": n_params(verifier_shapes(c)), "shared_params": 0,
            "cache_floats_per_token": c.n_layer * (c.kv_lora_rank + c.qk_rope_head_dim),
            "mha_cache_floats_per_token": 2 * c.n_layer * c.n_embd, "transient_floats_per_pass": 0,
            "n_layer": c.n_layer}


def gpt2_drafter(cfg):
    """A classic GPT-2-family draft model from its Hugging Face config.json (tied LM head)."""
    cfg = _as_dict(cfg)
    L, D, V, P = cfg["n_layer"], cfg["n_embd"], cfg["vocab_size"], cfg["n_positions"]
    inner = cfg.get("n_inner") or 4 * D
    per_layer = (2 * 2 * D                    # ln_1, ln_2
                 + D * 3 * D + 3 * D          # c_attn
                 + D * D + D                  # attn c_proj
                 + D * inner + inner          # mlp c_fc
                 + inner * D + D)             # mlp c_proj
    return {"kind": "gpt2", "params": V * D + P * D + L * per_layer + 2 * D, "shared_params": 0,
            "cache_floats_per_token": 2 * L * D, "transient_floats_per_pass": 0, "n_layer": L}


def hspec_drafter(cfg):
    """The H-Spec drafter (project/minitorch_drafter.py). shared_params: the verifier's LM head,
    which it reads in place."""
    mod = _drafter_module()
    c = mod.DrafterConfig.from_dict(_as_dict(cfg))
    return {"kind": "hspec", "params": n_params(mod.expected_shapes(c)),
            "shared_params": c.n_vocab * c.target_n_embd, "cache_floats_per_token": 0,
            "transient_floats_per_pass": (c.block_size + 1) * c.n_layer * (c.kv_lora_rank + c.qk_rope_head_dim),
            "n_layer": c.n_layer, "block_size": c.block_size}


def total_bytes(verifier, drafter=None, tokens=0):
    """Persistent bytes of the verifier (+ drafter) with `tokens` cached tokens (batch x context)."""
    floats = verifier["params"] + verifier["cache_floats_per_token"] * tokens
    if drafter:
        floats += drafter["params"] + drafter["cache_floats_per_token"] * tokens
    return BYTES_PER_FLOAT * floats


def drafter_bytes(drafter, tokens=0):
    """What a drafter adds to the verifier alone."""
    return BYTES_PER_FLOAT * (drafter["params"] + drafter["cache_floats_per_token"] * tokens)


def crossover_tokens(a, b):
    """Cached tokens at which drafters a and b add the same memory (None if they never cross)."""
    dc = a["cache_floats_per_token"] - b["cache_floats_per_token"]
    dp = b["params"] - a["params"]
    if dc == 0 or dp / dc < 0:
        return None
    return dp / dc


def read_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        meta = json.load(f)
    return meta.get("config", meta)  # specdec exports wrap it; Hugging Face config.json does not


def report(verifier, drafters, contexts=(128, 1024), batch=1):
    """Markdown tables: per model, and per setup at each context length."""
    mb = 1 / 2**20
    lines = ["| model | parameters | weights (MB, fp32) | shared with verifier | own cache per token | "
             "transient per pass |", "|---|---|---|---|---|---|"]
    for name, m in [("verifier (MLA)", verifier)] + list(drafters.items()):
        lines.append(f"| {name} | {m['params'] / 1e6:.1f}M | {BYTES_PER_FLOAT * m['params'] * mb:.0f} | "
                     f"{m['shared_params'] / 1e6:.1f}M | {m['cache_floats_per_token']:,} floats "
                     f"({BYTES_PER_FLOAT * m['cache_floats_per_token'] / 1024:.1f} KB) | "
                     f"{m['transient_floats_per_pass']:,} floats |")
    lines.append(f"\nThe verifier's MHA counterpart would cache {verifier['mha_cache_floats_per_token']:,} floats per "
                 f"token ({verifier['mha_cache_floats_per_token'] / verifier['cache_floats_per_token']:.2f}x its MLA "
                 "latent cache).\n")
    header = "| setup | " + " | ".join(f"total at {T} tokens (MB) | added by drafter at {T} (MB)" for T in contexts)
    lines += [f"Batch {batch}:\n", header + " |", "|---|" + "---|---|" * len(contexts)]
    for name, d in [("verifier alone", None)] + [(f"+ {n}", m) for n, m in drafters.items()]:
        cells = []
        for T in contexts:
            cells.append(f"{total_bytes(verifier, d, T * batch) * mb:.0f}")
            cells.append(f"{drafter_bytes(d, T * batch) * mb:.0f}" if d else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    names = list(drafters)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x = crossover_tokens(drafters[a], drafters[b])
            if x is not None:
                lines.append(f"\n{a} and {b} add the same memory at {x:,.0f} cached tokens (batch x context).")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verifier-dir", required=True, help="MLA verifier export (torch_train/export_mla_verifier.py)")
    ap.add_argument("--draft-dir", action="append", default=[], metavar="[LABEL=]DIR",
                    help="classic drafter (Hugging Face config.json), e.g. finetune_distilgpt2.py's best/")
    ap.add_argument("--hspec-drafter", action="append", default=[], metavar="[LABEL=]DIR",
                    help="H-Spec drafter export (torch_train/export_hspec_drafter.py)")
    ap.add_argument("--contexts", type=int, nargs="+", default=[128, 1024])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default=None, help="write the numbers as JSON here (and the tables to the same "
                                                "path with .md)")
    args = ap.parse_args(argv)

    def labelled(items, default):
        out = {}
        for item in items:
            label, sep, path = item.partition("=")
            label, path = (label, path) if sep else (default if default not in out else os.path.basename(item), item)
            out[label] = path
        return out

    verifier = mla_verifier(read_config(args.verifier_dir))
    drafters = {label: gpt2_drafter(read_config(path)) for label, path in labelled(args.draft_dir, "distilgpt2").items()}
    drafters.update({label: hspec_drafter(read_config(path))
                     for label, path in labelled(args.hspec_drafter, "H-Spec").items()})
    text = report(verifier, drafters, args.contexts, args.batch)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"bytes_per_float": BYTES_PER_FLOAT, "verifier": verifier, "drafters": drafters}, f, indent=2)
        with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
            f.write(text + "\n")
    return verifier, drafters


if __name__ == "__main__":
    main()
