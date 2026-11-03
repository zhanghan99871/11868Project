"""GPT-2 small pretrained-weight parity tests against Hugging Face.

Place this file at tests/test_gpt2_pretrained.py and run from repo root:
    python -m pytest tests/test_gpt2_pretrained.py -v -s

Downloads openai-community/gpt2 once, unless already cached.
Requires transformers, torch, and the MiniTorch CUDA shared libraries.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# The existing converter is located in <repo>/project/gpt2_loader.py.
PROJECT_DIR = Path(__file__).resolve().parents[2] / "project"
sys.path.insert(0, str(PROJECT_DIR))

import minitorch  # noqa: E402
from minitorch.cuda_kernel_ops import CudaKernelOps  # noqa: E402
from gpt2_loader import load_gpt2_weights  # noqa: E402

MODEL_ID = "openai-community/gpt2"
PROMPTS = [
    "Hello world",
    "The capital of France is",
    "Artificial intelligence is",
    "Once upon a time",
    "def fibonacci(n):",
]

# Conservative float32 tolerance for end-to-end logits, not kernel-level parity.
MAX_ABS_ERROR = 1e-2
MEAN_ABS_ERROR = 1e-3


@pytest.fixture(scope="session")
def reference():
    torch.manual_seed(0)
    hf = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    hf.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return hf, tokenizer


@pytest.fixture(scope="session", params=[False, True], ids=["standard", "fused"])
def minitorch_model(request, reference):
    hf, _ = reference
    cfg = hf.config
    backend = minitorch.TensorBackend(CudaKernelOps)

    mt = minitorch.DecoderLM(
        n_vocab=cfg.vocab_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_positions=cfg.n_positions,
        n_layer=cfg.n_layer,
        p_dropout=0.0,
        ln_eps=cfg.layer_norm_epsilon,
        bias=True,
        backend=backend,
        use_fused_kernel=request.param,
    )
    load_gpt2_weights(mt, hf, backend)
    mt.eval()
    return mt, backend, request.param


@pytest.mark.parametrize("prompt", PROMPTS, ids=[
    "hello", "france", "ai", "story", "code",
])
def test_gpt2_pretrained_logits(reference, minitorch_model, prompt):
    hf, tokenizer = reference
    mt, backend, fused = minitorch_model

    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.inference_mode():
        expected = hf(ids).logits.detach().cpu().numpy()

    mt_ids = minitorch.tensor_from_numpy(
        ids.cpu().numpy().astype(np.float32),
        backend=backend,
        requires_grad=False,
    )
    actual = mt(mt_ids).to_numpy()

    assert actual.shape == expected.shape, (
        f"Logit shape mismatch: MiniTorch={actual.shape}, HF={expected.shape}"
    )
    assert np.isfinite(actual).all(), "MiniTorch logits contain NaN or Inf"

    differences = np.abs(actual - expected)
    max_diff = float(differences.max())
    mean_diff = float(differences.mean())
    hf_next = int(expected[0, -1].argmax())
    mt_next = int(actual[0, -1].argmax())

    print(
        f"\nmode={'fused' if fused else 'standard'} prompt={prompt!r} "
        f"max={max_diff:.7f} mean={mean_diff:.7f} "
        f"next_match={hf_next == mt_next} "
        f"HF={tokenizer.decode([hf_next])!r} "
        f"MT={tokenizer.decode([mt_next])!r}"
    )

    assert max_diff <= MAX_ABS_ERROR, (
        f"Max logit error {max_diff:.7f} > {MAX_ABS_ERROR}; "
        f"mode={'fused' if fused else 'standard'}, prompt={prompt!r}"
    )
    assert mean_diff <= MEAN_ABS_ERROR, (
        f"Mean logit error {mean_diff:.7f} > {MEAN_ABS_ERROR}; "
        f"mode={'fused' if fused else 'standard'}, prompt={prompt!r}"
    )
    assert mt_next == hf_next, (
        f"Next token mismatch for {prompt!r}: "
        f"HF={tokenizer.decode([hf_next])!r}, MT={tokenizer.decode([mt_next])!r}"
    )
