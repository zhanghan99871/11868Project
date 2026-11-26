"""Export an H-Spec drafter (torch_train/hspec_drafter.py) for MiniTorch inference
(project/minitorch_drafter.py).

Writes <out_dir>/config.json (DrafterConfig fields, "format", provenance) and
<out_dir>/model.safetensors (fp32, PyTorch layout). The frozen target head is the verifier's LM
head and is not stored; the MiniTorch drafter takes it from the verifier.

Usage (a checkpoint holding {'model': drafter.state_dict(), 'config': asdict(drafter.cfg)}):
  python torch_train/export_hspec_drafter.py runs/hspec-drafter/best.pt runs/hspec-drafter/minitorch
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

FORMAT = "hspec-drafter-minitorch-v1"


def export_state(state_dict, config, out_dir, metadata=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {name: np.ascontiguousarray(value.detach().to("cpu", torch.float32).numpy())
               for name, value in state_dict.items() if name != "target_head"}
    save_file(tensors, str(out / "model.safetensors"))
    with open(out / "config.json", "w") as f:
        json.dump({"format": FORMAT, "config": dict(config), "metadata": metadata or {}}, f, indent=2)
    return out


def export_drafter(drafter, out_dir, metadata=None):
    return export_state(drafter.state_dict(), asdict(drafter.cfg), out_dir, metadata)


def export_checkpoint(ckpt_path, out_dir):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not {"model", "config"} <= set(ckpt):
        raise ValueError(f"{ckpt_path} is not a drafter checkpoint ({{'model', 'config'}})")
    metadata = {"source": str(ckpt_path), "step": ckpt.get("step"),
                "training_metadata": ckpt.get("metadata", {})}
    return export_state(ckpt["model"], ckpt["config"], out_dir, metadata)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    print(f"Exported {args.checkpoint} -> {export_checkpoint(args.checkpoint, args.out_dir)}")


if __name__ == "__main__":
    main()
