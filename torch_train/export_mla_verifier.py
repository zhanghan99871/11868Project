"""Export a trained PyTorch MLA checkpoint for torch-free MiniTorch inference (specdec/mla_model.py).

Reads a checkpoint written by torch_train/train.py ({'model': state_dict, 'config': MLAConfig dict,
...}) and writes:
  <out_dir>/config.json        the MLAConfig fields, plus "format" and provenance
  <out_dir>/model.safetensors  fp32 weights, keys and layouts exactly as in the PyTorch state_dict

lm_head.weight is tied to token_embeddings.weight in decoder_mla.DecoderLM, so it is not stored;
the loader reuses token_embeddings.weight. An untied lm_head is kept as-is.

Usage:
  python torch_train/export_mla_verifier.py runs/mla-gpt2-medium/best.pt runs/mla-gpt2-medium/minitorch
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

FORMAT = "mla-gpt2-minitorch-v1"


def export_state(state_dict, config, out_dir, metadata=None):
    """Write config.json and model.safetensors for a decoder_mla.DecoderLM state_dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, value in state_dict.items():
        if name == "lm_head.weight" and torch.equal(value, state_dict["token_embeddings.weight"]):
            continue  # tied to the token embedding
        tensors[name] = np.ascontiguousarray(value.detach().to("cpu", torch.float32).numpy())
    save_file(tensors, str(out / "model.safetensors"))
    with open(out / "config.json", "w") as f:
        json.dump({"format": FORMAT, "config": dict(config), "metadata": metadata or {}}, f, indent=2)
    return out


def export_checkpoint(ckpt_path, out_dir):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not {"model", "config"} <= set(ckpt):
        raise ValueError(f"{ckpt_path} is not a torch_train/train.py MLA checkpoint")
    metadata = {"source": str(ckpt_path), "step": ckpt.get("step"),
                "training_metadata": ckpt.get("metadata", {})}
    return export_state(ckpt["model"], ckpt["config"], out_dir, metadata)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint", help="best.pt written by torch_train/train.py")
    ap.add_argument("out_dir", help="directory for config.json and model.safetensors")
    args = ap.parse_args()
    out = export_checkpoint(args.checkpoint, args.out_dir)
    print(f"Exported {args.checkpoint} -> {out}")


if __name__ == "__main__":
    main()
