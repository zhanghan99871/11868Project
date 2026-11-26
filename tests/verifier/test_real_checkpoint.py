"""The trained MLA verifier, exported, against the PyTorch checkpoint it came from.

Needs a GPU and the checkpoint written by torch_train/train.py: runs/mla-gpt2-medium/best.pt,
or set MLA_CHECKPOINT=/path/to/best.pt. Skipped otherwise. Writes a ~1.4 GB export to pytest's
tmp dir and holds both models in memory.
"""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdec.decoding import autoregressive  # noqa: E402
from specdec.mla_model import MiniTorchMLAGPT2  # noqa: E402

from .helpers import ROOT, DecoderLM, MLAConfig, torch_greedy, torch_logits  # noqa: E402

CKPT = os.environ.get("MLA_CHECKPOINT", str(ROOT / "runs" / "mla-gpt2-medium" / "best.pt"))
pytestmark = pytest.mark.skipif(not os.path.isfile(CKPT),
                                reason=f"no trained MLA checkpoint at {CKPT} (set MLA_CHECKPOINT)")


def test_trained_verifier_matches_pytorch(cuda_backend, tmp_path):
    from export_mla_verifier import export_checkpoint

    out = export_checkpoint(CKPT, tmp_path / "verifier")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = DecoderLM(MLAConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    del ckpt

    with open(ROOT / "results" / "prompts_wikitext2.json") as f:
        prompt = json.load(f)["prompts"][0][:64]
    verifier = MiniTorchMLAGPT2(str(out), backend=cuda_backend)

    ref = torch_logits(model, prompt)
    got = verifier.forward(prompt, verifier.new_cache())
    err = np.abs(got - ref).max()
    assert err < 1e-2, f"max |logit diff| {err:.2e} (|logit| max {np.abs(ref).max():.1f})"
    assert (got.argmax(-1) == ref.argmax(-1)).all()

    # cached decoding, including the RoPE offsets of later positions (a near-tie could flip a token)
    assert autoregressive(verifier, prompt, 16)[0] == torch_greedy(model, prompt, 16)
