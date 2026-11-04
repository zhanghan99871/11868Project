"""Shared setup. The project depends on MiniTorch only: no torch, no transformers.

Tokenization uses the `tokenizers` library on the checkpoint's tokenizer.json;
checkpoints are read with safetensors.numpy. `assert_no_torch()` is called at the
end of every entry point to make sure nothing pulled torch in transitively.
"""
import json
import os
import sys

HF_HOME = os.environ.setdefault("HF_HOME", "/ocean/projects/cis260267p/yni3/hf")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.path.join(PROJ, "results", "prompts_wikitext2.json")
MODELS = {"gpt2": "openai-community/gpt2", "gpt2-medium": "openai-community/gpt2-medium",
          "distilgpt2": "distilbert/distilgpt2"}  # all share the GPT-2 BPE tokenizer


def model_dir(name):
    from huggingface_hub import snapshot_download
    return snapshot_download(MODELS[name], local_files_only=True)


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(model_dir("gpt2"), "tokenizer.json"))


def backend():
    import minitorch
    from minitorch.cuda_kernel_ops import CudaKernelOps
    assert minitorch.__file__.startswith(PROJ), "picked up a minitorch outside this project: " + minitorch.__file__
    return minitorch.TensorBackend(CudaKernelOps)


def load(name, max_len=1024, resident=True):
    from .model import MiniTorchGPT2
    return MiniTorchGPT2(model_dir(name), backend(), max_len=max_len, resident=resident)


def prompts(n=None):
    with open(PROMPTS) as f:
        p = json.load(f)["prompts"]
    return p if n is None else p[:n]


def assert_no_torch():
    assert "torch" not in sys.modules, "torch was imported somewhere"
    print("[check] torch not imported", flush=True)
